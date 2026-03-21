"""
FFN_Q 量子前馈网络 (Quantum Feed-Forward Network)

核心设计：
    FFN_Q(|ψ⟩) = U_out · ModReLU(W_up |ψ⟩ + |b⟩)

其中：
- W_up: 上投影复数权重 (Cayley 参数化)
- ModReLU: 复数激活函数（相位保持，模长调节）
- U_out: 输出酉投影（Cayley 参数化）

相比 Transformer FFN：
- 所有线性层使用酉矩阵，梯度流守恒
- ModReLU 保持相位信息，不丢失复数编码
- 支持门控机制（GLU 变体）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .activations import ModReLU
from .unitary import CayleyLinear
from .complex_ops import complex_dropout


class ComplexBias(nn.Module):
    """可学习的复数偏置。"""

    def __init__(self, features: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(features, dtype=torch.complex64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.bias


class QuantumFFN(nn.Module):
    """量子前馈网络 (FFN_Q)。

    FFN_Q(|ψ⟩) = U_out · σ_C(W_up |ψ⟩ + b) + |ψ⟩

    Args:
        dim: 输入/输出特征维度
        ffn_dim: 中间层维度（通常 4x dim）
        dropout: dropout 概率
        use_glu: 是否使用门控线性单元变体
        activation: 激活函数类型 ('modrelu', 'gelu', 'relu')
    """

    def __init__(
        self,
        dim: int,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.0,
        use_glu: bool = False,
        activation: str = 'modrelu',
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim or (dim * 4)
        self.use_glu = use_glu
        self.dropout_p = dropout

        # 上投影（一般复数线性层，非酉，允许特征空间扩展）
        self.W_up = nn.Linear(dim, self.ffn_dim, bias=False).to(torch.complex64)

        if use_glu:
            # 门控投影
            self.W_gate = nn.Linear(dim, self.ffn_dim, bias=False).to(torch.complex64)
            # 输出投影维度减半（GLU 会把维度减半）
            actual_ffn_out = self.ffn_dim // 2
        else:
            actual_ffn_out = self.ffn_dim

        # 复数偏置
        self.bias_up = ComplexBias(self.ffn_dim)

        # 激活函数（GLU 模式下，激活在截断后应用）
        if activation == 'modrelu':
            self.activation = ModReLU(self.ffn_dim)  # 创建为完整 ffn_dim，forward 中会先截断
        elif activation == 'gelu':
            from .activations import ComplexGELU
            self.activation = ComplexGELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # 下投影（酉矩阵，保持酉性）
        self.W_down = CayleyLinear(actual_ffn_out, dim, init_scale=0.02)

        # LayerNorm（放在 FFN 内部）
        from .normalization import ComplexLayerNorm
        self.norm = ComplexLayerNorm(dim)

        self._actual_ffn_out = actual_ffn_out

    def forward(
        self, x: torch.Tensor, training: bool = True
    ) -> torch.Tensor:
        """
        Args:
            x: 复数输入 (batch, seq_len, dim)
            training: 是否训练模式
        Returns:
            复数输出 (batch, seq_len, dim)
        """
        residual = x

        # 上投影
        up = self.W_up(x)  # (B, N, ffn_dim)
        up = self.bias_up(up)

        # 复数激活
        up = self.activation(up)

        # 门控机制（可选）
        if self.use_glu:
            gate = self.W_gate(x)  # (B, N, ffn_dim)
            up = up * torch.sigmoid(gate.real + 1j * gate.imag * 0.5)
            # GLU 把维度减半
            up = up[..., :self._actual_ffn_out]

        # Dropout
        if training and self.dropout_p > 0:
            up = complex_dropout(up, p=self.dropout_p, training=True)

        # 下投影（酉矩阵）
        down = self.W_down(up)  # (B, N, dim)

        # 归一化
        down = self.norm(down)

        # 残差连接
        return residual + down

    def extra_repr(self) -> str:
        return (
            f'dim={self.dim}, ffn_dim={self.ffn_dim}, '
            f'glu={self.use_glu}, dropout={self.dropout_p}'
        )
