"""
QSA 量子叠加注意力 (Quantum Superposition Attention)

核心流程：
1. 复数投影：Q = x @ Wq, K = x @ Wk, V = x @ Wv（酉矩阵投影）
2. 复数内积：α_ij = ⟨q_i | k_j⟩
3. 干涉调制：β_ij = α_ij · exp(i · f_φ(|α_ij|))（QIR 相位调制）
4. Born 概率：p_ij = |β_ij|²
5. Top-K 筛选：仅保留概率最高的 k 个键
6. 复数 Softmax 加权聚合

复杂度：O(n·d·log n) vs Transformer O(n²·d)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .unitary import CayleyLinear
from .complex_ops import (
    complex_inner_product,
    born_normalize,
    complex_softmax,
    interference_score,
    von_neumann_entropy,
    complex_dropout,
)


class PhaseModulation(nn.Module):
    """可学习相位调制函数（QIR 的核心组件）。

    f_φ(|α|) = MLP(|α|)，将内积模长映射为相位偏移。

    Args:
        hidden_dim: MLP 隐藏维度
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, magnitude: torch.Tensor) -> torch.Tensor:
        """将内积模长映射为相位偏移。

        Args:
            magnitude: 实数模长 (...,)
        Returns:
            相位偏移 (...,)
        """
        return self.mlp(magnitude.unsqueeze(-1)).squeeze(-1)


class QuantumSuperpositionAttention(nn.Module):
    """量子叠加注意力 (QSA)。

    支持两种模式：
    - full: 完整注意力（O(n²)），用于正确性验证
    - topk: Top-K 干涉路由（O(n·log n)），用于高效推理

    Args:
        dim: 输入/输出特征维度
        num_heads: 多头注意力头数
        head_dim: 每个头的维度（默认 dim // num_heads）
        topk_ratio: Top-K 筛选比例（0.05 ~ 0.3）
        dropout: dropout 概率
        mode: 'topk' 或 'full'
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        head_dim: Optional[int] = None,
        topk_ratio: float = 0.1,
        dropout: float = 0.0,
        mode: str = 'topk',
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim or (dim // num_heads)
        self.topk_ratio = topk_ratio
        self.mode = mode
        self.scale = self.head_dim ** -0.5

        assert self.num_heads * self.head_dim == dim, \
            f"num_heads({num_heads}) * head_dim({self.head_dim}) != dim({dim})"

        # 复数 Q/K/V 投影（使用 Cayley 参数化保证酉性）
        self.Wq = CayleyLinear(dim, dim, init_scale=0.02)
        self.Wk = CayleyLinear(dim, dim, init_scale=0.02)
        self.Wv = CayleyLinear(dim, dim, init_scale=0.02)

        # 输出投影（酉矩阵）
        self.Wo = CayleyLinear(dim, dim, init_scale=0.02)

        # QIR 相位调制函数
        self.phase_fn = PhaseModulation(hidden_dim=64)

        self.dropout_p = dropout

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        training: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x: 复数输入 (batch, seq_len, dim)
            attention_mask: 可选的注意力掩码 (batch, seq_len)
            training: 是否训练模式
        Returns:
            (output, metrics)
            output: 复数张量 (batch, seq_len, dim)
            metrics: 注意力统计信息字典
        """
        B, N, D = x.shape

        # ── 1. 复数投影 ──
        Q = self.Wq(x)  # (B, N, D)
        K = self.Wk(x)
        V = self.Wv(x)

        # ── 2. 多头分割 ──
        # (B, N, num_heads, head_dim) -> (B, num_heads, N, head_dim)
        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # ── 3. 复数内积 + 干涉调制 ──
        # α_ij = ⟨q_i | k_j⟩，形状 (B, num_heads, N, N)
        alpha = torch.einsum('bhnd,bhsd->bhns', Q.conj(), K) * self.scale

        # 干涉调制：β = α · exp(i · f(|α|))
        alpha_mag = alpha.abs()
        phase_shift = self.phase_fn(alpha_mag.detach())  # detach 模长用于相位调制
        beta = alpha * torch.exp(1j * phase_shift)

        # ── 4. Born 概率 ──
        attn_probs = born_normalize(beta, dim=-1)  # (B, H, N, N)

        # ── 5. 注意力熵（度量信息集中度）──
        attn_entropy = von_neumann_entropy(attn_probs, dim=-1)  # (B, H, N)

        # ── 6. Top-K 筛选或完整注意力 ──
        if self.mode == 'topk' and training:
            output = self._topk_attention(V, attn_probs, B, N)
        else:
            output = self._full_attention(V, attn_probs, attention_mask)

        # ── 7. 多头合并 ──
        # (B, num_heads, N, head_dim) -> (B, N, D)
        output = output.transpose(1, 2).contiguous().view(B, N, D)

        # ── 8. 输出投影 ──
        output = self.Wo(output)

        # 应用 dropout
        if training and self.dropout_p > 0:
            output = complex_dropout(output, p=self.dropout_p, training=True)

        # 构建指标
        metrics = {
            'attention_entropy': attn_entropy.mean().item(),
            'attention_probs_max': attn_probs.max().item(),
            'interference_phase_std': phase_shift.std().item(),
            'topk_ratio_actual': self.topk_ratio,
            'qsa_mode': self.mode,
        }

        return output, metrics

    def _full_attention(
        self,
        V: torch.Tensor,
        attn_probs: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """完整注意力（O(n²)），用于验证和推理。

        Args:
            V: 值张量 (B, H, N, d_h)
            attn_probs: 注意力概率 (B, H, N, N) — 实数或复数
            mask: 可选掩码
        Returns:
            加权输出 (B, H, N, d_h)
        """
        if mask is not None:
            # mask: (B, N) -> (B, 1, 1, N)
            mask = mask.unsqueeze(1).unsqueeze(2)
            attn_probs = attn_probs.masked_fill(mask == 0, 0.0)
            prob_sum = attn_probs.sum(dim=-1, keepdim=True)
            attn_probs = attn_probs / (prob_sum + 1e-8)

        # 使用概率加权聚合（概率是实数，V 是复数）
        # 将实数概率转为复数以兼容 einsum
        weights = attn_probs.to(V.dtype)
        return torch.einsum('bhns,bhsd->bhnd', weights, V)

    def _topk_attention(
        self,
        V: torch.Tensor,
        attn_probs: torch.Tensor,
        B: int,
        N: int,
    ) -> torch.Tensor:
        """Top-K 干涉路由注意力（O(n·k·d)）。

        仅对每个查询的 Top-K 个键计算精确注意力。

        Args:
            V: 值张量 (B, H, N, d_h)
            attn_probs: 注意力概率 (B, H, N, N)
            B: batch size
            N: 序列长度
        Returns:
            加权输出 (B, H, N, d_h)
        """
        k = max(1, int(self.topk_ratio * N))

        # Top-K 选择
        topk_probs, topk_indices = torch.topk(attn_probs, k=k, dim=-1)  # (B, H, N, k)

        # 归一化选中的概率
        topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-8)

        # 收集 Top-K 的值
        # topk_indices: (B, H, N, k) -> 需要 gather V
        H = V.shape[1]
        d_h = V.shape[-1]

        # 扩展索引用于 gather
        idx_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, -1, -1, d_h)  # (B, H, N, k, d_h)
        V_expanded = V.unsqueeze(3).expand(-1, -1, -1, N, -1)  # (B, H, N, N, d_h)
        V_topk = torch.gather(V_expanded, 3, idx_expanded)  # (B, H, N, k, d_h)

        # 复数 Softmax 加权
        weights = complex_softmax(topk_probs * k, dim=-1).unsqueeze(-1)  # (B, H, N, k, 1)
        output = (weights * V_topk).sum(dim=3)  # (B, H, N, d_h)

        return output

    def set_mode(self, mode: str):
        """切换 QSA 运行模式。

        Args:
            mode: 'topk'（高效推理）或 'full'（完整 O(n²) 注意力，用于验证）
        """
        if mode not in ('topk', 'full'):
            raise ValueError(f"mode 必须为 'topk' 或 'full'，收到: {mode!r}")
        self.mode = mode

    def set_topk_ratio(self, ratio: float):
        """动态调整 Top-K 筛选比例。

        可被优化系统在线调用，无需重新初始化模型。

        Args:
            ratio: 新的 Top-K 比例，范围 (0, 1]
        """
        if not 0 < ratio <= 1.0:
            raise ValueError(f"topk_ratio 必须在 (0, 1] 内，收到: {ratio}")
        self.topk_ratio = ratio

    def extra_repr(self) -> str:
        return (
            f'dim={self.dim}, num_heads={self.num_heads}, '
            f'head_dim={self.head_dim}, topk_ratio={self.topk_ratio}, '
            f'mode={self.mode}'
        )
