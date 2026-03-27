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
        mode: str = "topk",
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim or (dim // num_heads)
        self.topk_ratio = topk_ratio
        self.mode = mode
        self.scale = self.head_dim**-0.5

        assert (
            self.num_heads * self.head_dim == dim
        ), f"num_heads({num_heads}) * head_dim({self.head_dim}) != dim({dim})"

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
        causal: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x: 复数输入 (batch, seq_len, dim)
            attention_mask: 可选的 padding 掩码 (batch, seq_len)，0 表示忽略位置
            training: 是否训练模式
            causal: 是否使用因果掩码（下三角注意力），适配自回归语言模型。
                    启用时 token i 只能关注 j <= i 的位置，防止信息泄漏。
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
        alpha = torch.einsum("bhnd,bhsd->bhns", Q.conj(), K) * self.scale

        # 干涉调制：β = α · exp(i · f(|α|))
        alpha_mag = alpha.abs()
        phase_shift = self.phase_fn(alpha_mag.detach())  # detach 模长用于相位调制
        beta = alpha * torch.exp(1j * phase_shift)

        # ── 4. 因果掩码（自回归模式）──
        # 构造下三角掩码矩阵，确保位置 i 只关注 j <= i（防止未来信息泄漏）
        # causal_mask[i, j] = True 表示位置 i 可以关注位置 j
        if causal:
            causal_mask = torch.tril(
                torch.ones(N, N, device=x.device, dtype=torch.bool)
            )  # (N, N)
            # 将下三角外的位置（未来位置）的 beta 置为极小值，使 Born 概率接近 0
            mask_value = torch.finfo(torch.float32).min
            beta = beta.masked_fill(
                ~causal_mask.unsqueeze(0).unsqueeze(0),  # (1, 1, N, N)
                complex(mask_value, 0.0),
            )

        # ── 5. Born 概率 ──
        attn_probs = born_normalize(beta, dim=-1)  # (B, H, N, N)

        # ── 6. 注意力熵（度量信息集中度）──
        attn_entropy = von_neumann_entropy(attn_probs, dim=-1)  # (B, H, N)

        # ── 7. Top-K 筛选或完整注意力 ──
        # topk 模式：训练和推理均使用 Top-K 干涉路由（O(n·k·d)），只有
        # full 模式才强制使用完整 O(n²) 注意力（用于正确性验证/对比实验）。
        # 修复：旧版仅在 training=True 时使用 topk，导致推理阶段退化为
        # O(n²) full attention，丧失了 topk 的推理加速优势。
        if self.mode == "topk":
            output = self._topk_attention(V, attn_probs, B, N)
        else:
            output = self._full_attention(V, attn_probs, attention_mask)

        # ── 8. 多头合并 ──
        # (B, num_heads, N, head_dim) -> (B, N, D)
        output = output.transpose(1, 2).contiguous().view(B, N, D)

        # ── 9. 输出投影 ──
        output = self.Wo(output)

        # 应用 dropout
        if training and self.dropout_p > 0:
            output = complex_dropout(output, p=self.dropout_p, training=True)

        # 构建指标
        metrics = {
            "attention_entropy": attn_entropy.mean().item(),
            "attention_probs_max": attn_probs.max().item(),
            "interference_phase_std": phase_shift.std().item(),
            "topk_ratio_actual": self.topk_ratio,
            "qsa_mode": self.mode,
            "causal": causal,
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
        return torch.einsum("bhns,bhsd->bhnd", weights, V)

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
        if mode not in ("topk", "full"):
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

    @torch.no_grad()
    def get_attention_patterns(
        self,
        x: torch.Tensor,
        causal: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """提取完整的注意力模式，用于可视化与诊断。

        在不更新模型参数的情况下，计算所有头的注意力热图：
        - Born 概率矩阵（归一化为合法概率分布）
        - 干涉相位矩阵（量子相位调制结果）
        - 各头注意力熵（度量注意力集中/分散程度）

        典型使用场景：
        1. 注意力可视化：观察不同层不同头关注的位置模式
        2. 稀疏性分析：验证 Top-K 筛选是否过滤了正确的低概率位置
        3. 相位干涉调试：检测相长/相消干涉是否产生有意义的路由

        Args:
            x: 复数输入 (batch, seq_len, dim)
            causal: 是否使用因果掩码（与 forward 一致）
        Returns:
            dict 包含:
                - ``attn_probs``: 注意力概率矩阵 (B, H, N, N)，实数
                - ``phase_matrix``: 干涉相位矩阵 (B, H, N, N)，实数 [弧度]
                - ``attn_entropy``: 每个查询的注意力熵 (B, H, N)，实数
                - ``topk_mask``: Top-K 稀疏掩码 (B, H, N, N)，bool
                  （True = 被 Top-K 保留，False = 被筛除）
        """
        B, N, D = x.shape

        # 复数投影
        Q = self.Wq(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.Wk(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # 复数内积 α = ⟨Q|K⟩
        alpha = torch.einsum("bhnd,bhsd->bhns", Q.conj(), K) * self.scale

        # 干涉相位调制 β = α · e^{i·f(|α|)}
        alpha_mag = alpha.abs()
        phase_shift = self.phase_fn(alpha_mag.detach())
        beta = alpha * torch.exp(1j * phase_shift)

        # 因果掩码（可选）
        if causal:
            causal_mask = torch.tril(torch.ones(N, N, device=x.device, dtype=torch.bool))
            mask_value = torch.finfo(torch.float32).min
            beta = beta.masked_fill(
                ~causal_mask.unsqueeze(0).unsqueeze(0),
                complex(mask_value, 0.0),
            )

        # Born 概率
        attn_probs = born_normalize(beta, dim=-1)  # (B, H, N, N)，实数

        # 注意力熵
        attn_entropy = von_neumann_entropy(attn_probs, dim=-1)  # (B, H, N)

        # Top-K 稀疏掩码（标记哪些位置被 Top-K 筛选保留）
        k = max(1, int(self.topk_ratio * N))
        _, topk_indices = torch.topk(attn_probs, k=k, dim=-1)  # (B, H, N, k)
        topk_mask = torch.zeros(B, self.num_heads, N, N, dtype=torch.bool, device=x.device)
        topk_mask.scatter_(3, topk_indices, True)  # (B, H, N, N)

        return {
            "attn_probs": attn_probs,           # (B, H, N, N) 实数概率
            "phase_matrix": phase_shift,         # (B, H, N, N) 干涉相位
            "attn_entropy": attn_entropy,         # (B, H, N) 每查询熵
            "topk_mask": topk_mask,               # (B, H, N, N) bool 稀疏掩码
        }

    def multi_head_entropy_summary(
        self,
        x: torch.Tensor,
        causal: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """多头注意力熵分布分析（诊断工具）。

        分析所有注意力头的熵分布特征，识别以下异常模式：
        1. **头崩溃（Head Collapse）**：多个头的注意力模式高度相似（熵差 < ε）
        2. **均匀化头（Uniform Head）**：熵接近最大值（max_entropy），注意力均匀分散
        3. **尖锐头（Sharp Head）**：熵接近 0，注意力集中在少数位置（可能退化）
        4. **头多样性（Head Diversity）**：各头熵的标准差，高多样性 = 各头学到不同特征

        对于 L 层 H 头的量子架构，理想状态是：
        - 每层不同头关注不同语义层次（熵多样性高）
        - 同一层相邻头之间互信息低（头独立性好）
        - 深层头比浅层头具有更集中的注意力（熵随层数降低）

        此方法用于训练时的诊断和超参数调整，不影响梯度计算。

        Args:
            x: 复数输入 (batch, seq_len, dim)
            causal: 是否使用因果掩码
        Returns:
            dict 包含:
                - ``per_head_entropy``: 每头的平均注意力熵 (num_heads,) 实数
                - ``entropy_mean``: 所有头的熵均值（标量）
                - ``entropy_std``: 所有头的熵标准差（标量，度量头多样性）
                - ``entropy_min``: 最小头熵（标量，标识最集中的头）
                - ``entropy_max``: 最大头熵（标量，标识最分散的头）
                - ``uniform_head_mask``: bool (num_heads,)，True = 均匀化头（熵 > 0.9*max）
                - ``sharp_head_mask``: bool (num_heads,)，True = 尖锐头（熵 < 0.1*max）
                - ``max_entropy_ref``: 当前序列长度下的最大可能熵（实数标量）
                - ``head_diversity_score``: 头多样性分数 ∈ [0,1]（归一化的熵标准差）
        """
        import math as _math

        patterns = self.get_attention_patterns(x, causal=causal)
        # attn_probs: (B, H, N, N) — 实数概率矩阵
        attn_probs = patterns["attn_probs"]
        B, H, N, _ = attn_probs.shape

        # 每个查询位置的注意力熵 (B, H, N)
        entropy_per_query = von_neumann_entropy(attn_probs, dim=-1)  # (B, H, N)

        # 对 batch 和 query 位置取平均 → 每头的代表性熵 (H,)
        per_head_entropy = entropy_per_query.mean(dim=[0, 2])  # (H,)

        # 统计量
        entropy_mean = per_head_entropy.mean()
        entropy_std = per_head_entropy.std() if H > 1 else torch.tensor(0.0)
        entropy_min = per_head_entropy.min()
        entropy_max = per_head_entropy.max()

        # 当前序列长度下的最大熵（均匀分布时）
        max_entropy_ref = _math.log(N) if N > 1 else 0.0

        # 均匀化头：熵超过 90% 最大熵 → 注意力过于分散，缺少选择性
        uniform_threshold = 0.9 * max_entropy_ref
        uniform_head_mask = per_head_entropy > uniform_threshold

        # 尖锐头：熵低于 10% 最大熵 → 注意力过于集中，可能退化为固定模式
        sharp_threshold = 0.1 * max_entropy_ref
        sharp_head_mask = per_head_entropy < sharp_threshold

        # 头多样性分数 ∈ [0, 1]：归一化的头间熵标准差
        # max_std ≤ max_entropy_ref / 2（理论最大标准差）
        max_possible_std = max_entropy_ref / 2.0 if max_entropy_ref > 0 else 1.0
        head_diversity_score = float(entropy_std.item() / max(max_possible_std, 1e-8))
        head_diversity_score = min(1.0, head_diversity_score)  # clamp 到 [0, 1]

        return {
            "per_head_entropy": per_head_entropy,           # (H,) 每头平均熵
            "entropy_mean": entropy_mean,                   # 标量
            "entropy_std": entropy_std,                     # 标量（头多样性）
            "entropy_min": entropy_min,                     # 标量
            "entropy_max": entropy_max,                     # 标量
            "uniform_head_mask": uniform_head_mask,         # (H,) bool
            "sharp_head_mask": sharp_head_mask,             # (H,) bool
            "max_entropy_ref": max_entropy_ref,             # 最大可能熵
            "head_diversity_score": head_diversity_score,   # [0,1] 多样性分数
        }

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, num_heads={self.num_heads}, "
            f"head_dim={self.head_dim}, topk_ratio={self.topk_ratio}, "
            f"mode={self.mode}"
        )
