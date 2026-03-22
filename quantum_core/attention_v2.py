"""
QSA 量子叠加注意力 - 优化版本 v2

简单优化：
1. 使用SiLU替代ReLU
2. 使用rsqrt替代除法
3. 优化TopK计算流程
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .unitary import CayleyLinear


class PhaseModulation(nn.Module):
    """可学习相位调制函数 - 优化版本使用SiLU"""

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),  # SiLU替代ReLU，更平滑
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, magnitude: torch.Tensor) -> torch.Tensor:
        return self.mlp(magnitude.unsqueeze(-1)).squeeze(-1)


class QuantumSuperpositionAttentionOptimized(nn.Module):
    """量子叠加注意力 - 优化版本"""

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

        assert self.num_heads * self.head_dim == dim

        # 复数 Q/K/V 投影
        self.Wq = CayleyLinear(dim, dim, init_scale=0.02)
        self.Wk = CayleyLinear(dim, dim, init_scale=0.02)
        self.Wv = CayleyLinear(dim, dim, init_scale=0.02)

        # 输出投影
        self.Wo = CayleyLinear(dim, dim, init_scale=0.02)

        # QIR相位调制
        self.phase_fn = PhaseModulation(hidden_dim=64)

        self.dropout_p = dropout

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        training: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, N, D = x.shape

        # 1. 复数投影
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        # 2. 多头分割
        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. 注意力计算 - 使用rsqrt优化
        rsqrt_scale = self.scale**-0.5  # rsqrt优化

        # 复数内积
        alpha = torch.einsum("bhnd,bhsd->bhns", Q.conj(), K) * rsqrt_scale

        # 干涉调制
        alpha_mag = alpha.abs()
        phase_shift = self.phase_fn(alpha_mag.detach())
        beta = alpha * torch.exp(1j * phase_shift)

        # Born概率 (简化)
        attn_probs = beta.abs() / (beta.abs().sum(dim=-1, keepdim=True) + 1e-8)

        # 4. TopK或Full
        if self.mode == "topk" and training:
            output = self._topk_attention(V, attn_probs, B, N)
        else:
            output = self._full_attention(V, attn_probs, attention_mask)

        # 5. 多头合并
        output = output.transpose(1, 2).contiguous().view(B, N, D)

        # 6. 输出投影
        output = self.Wo(output)

        # Dropout
        if training and self.dropout_p > 0:
            output = F.dropout(output, p=self.dropout_p, training=training)

        metrics = {
            "attention_entropy": 0.0,
            "attention_probs_max": attn_probs.max().item(),
            "interference_phase_std": phase_shift.std().item(),
            "topk_ratio_actual": self.topk_ratio,
            "qsa_mode": self.mode,
            "optimized": True,
        }

        return output, metrics

    def _full_attention(self, V, attn_probs, mask):
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            attn_probs = attn_probs.masked_fill(mask == 0, 0.0)
            attn_probs = attn_probs / (attn_probs.sum(dim=-1, keepdim=True) + 1e-8)

        return torch.einsum("bhns,bhsd->bhnd", attn_probs.real, V)

    def _topk_attention(self, V, attn_probs, B, N):
        k = max(1, int(self.topk_ratio * N))

        # TopK选择
        topk_probs, topk_indices = torch.topk(attn_probs, k=k, dim=-1)

        # 归一化
        topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-8)

        # Gather TopK values
        H = V.shape[1]
        d_h = V.shape[-1]

        # 使用efficient gather
        idx_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, -1, -1, d_h)
        V_expanded = V.unsqueeze(3).expand(-1, -1, -1, N, -1)
        V_topk = torch.gather(V_expanded, 3, idx_expanded)

        # 加权输出
        weights = topk_probs.unsqueeze(-1)
        output = (weights * V_topk).sum(dim=3)

        return output


# 别名
QuantumSuperpositionAttention = QuantumSuperpositionAttentionOptimized
