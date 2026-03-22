"""
复数嵌入层 + 量子位置编码

- ComplexEmbedding: 将离散 token 映射为复数向量（量子态）
- QuantumPositionalEncoding: 酉旋转位置编码（RzRx 变换）
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class ComplexEmbedding(nn.Module):
    """复数嵌入层。

    将离散 token ID 映射为复数向量（量子态）：
        |x_i⟩ = E[i]  (自动归一化为单位向量)

    Args:
        vocab_size: 词汇表大小
        dim: 嵌入维度
        normalize: 是否归一化为单位向量（量子态要求）
        scale: 初始化缩放因子
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        normalize: bool = True,
        scale: float = 0.02,
    ):
        super().__init__()
        self.dim = dim
        self.normalize = normalize

        # 复数嵌入矩阵
        self.embedding = nn.Parameter(torch.randn(vocab_size, dim, dtype=torch.complex64) * scale)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: 整数张量 (batch, seq_len) 或 (seq_len,)
        Returns:
            复数嵌入 (..., dim)
        """
        z = self.embedding[token_ids]

        if self.normalize:
            norm = z.abs().pow(2).sum(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)
            z = z / norm

        return z

    def extra_repr(self) -> str:
        return f"vocab_size={self.embedding.shape[0]}, dim={self.dim}, normalize={self.normalize}"


class QuantumPositionalEncoding(nn.Module):
    """量子位置编码（酉旋转）。

    不同于 Transformer 的 sin/cos 位置编码，量子架构使用酉旋转：
        |x_i⟩ → R(pos_i) |x_i⟩

    其中 R(pos) = Rz(ω_z·pos) · Rx(ω_x·pos) 是参数化的酉旋转。

    Args:
        dim: 特征维度
        max_len: 最大序列长度
        dropout: dropout 概率
    """

    def __init__(
        self,
        dim: int,
        max_len: int = 5120,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.dropout_p = dropout

        # 可学习位置编码参数
        # 频率参数 ω_z 和 ω_x
        self.omega_z = nn.Parameter(torch.randn(1, max_len, dim) * 0.01)
        self.omega_x = nn.Parameter(torch.randn(1, max_len, dim) * 0.01)

    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        Args:
            x: 复数输入 (batch, seq_len, dim)
            training: 是否训练模式
        Returns:
            位置编码后的复数张量 (batch, seq_len, dim)
        """
        B, N, D = x.shape
        N = min(N, self.omega_z.shape[1])

        # Rz 旋转：相位偏移 exp(i * ω_z * pos)
        phase_z = self.omega_z[:, :N, :D]
        rz = torch.exp(1j * phase_z)  # (1, N, D)

        # Rx 旋转：混合实部和虚部（类似 Pauli-X 门）
        # |0⟩ → cos(ω) |0⟩ + i·sin(ω) |1⟩
        phase_x = self.omega_x[:, :N, :D]
        cos_x = torch.cos(phase_x)
        sin_x = torch.sin(phase_x)

        # 应用 Rz
        x_rotated = x * rz

        # 应用 Rx：混合实部和虚部
        real_part = x_rotated.real * cos_x - x_rotated.imag * sin_x
        imag_part = x_rotated.real * sin_x + x_rotated.imag * cos_x
        x_rotated = torch.complex(real_part, imag_part)

        if training and self.dropout_p > 0:
            mask = (torch.rand_like(x.real) > self.dropout_p).float()
            x_rotated = x_rotated * mask

        return x_rotated

    def extra_repr(self) -> str:
        return f"dim={self.dim}, max_len={self.omega_z.shape[1]}"


class LearnedPositionalEncoding(nn.Module):
    """可学习位置编码（简单版，兼容性更好）。

    直接在复数空间学习位置偏置：
        output = input + pos_embedding
    """

    def __init__(self, dim: int, max_len: int = 5120, dropout: float = 0.0):
        super().__init__()
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_len, dim, dtype=torch.complex64) * 0.02
        )
        self.dropout_p = dropout

    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        N = x.shape[1]
        N = min(N, self.pos_embedding.shape[1])
        x = x + self.pos_embedding[:, :N, :]

        if training and self.dropout_p > 0:
            from .complex_ops import complex_dropout

            x = complex_dropout(x, p=self.dropout_p, training=True)

        return x
