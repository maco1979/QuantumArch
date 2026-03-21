"""
复数归一化层

- ComplexLayerNorm: 复数 LayerNorm（对实部和虚部分别归一化后合成）
- ComplexBatchNorm: 复数 BatchNorm（对模长做 BN，保持相位不变）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ComplexLayerNorm(nn.Module):
    """复数 Layer Normalization。

    对复数张量做归一化，使用可学习的仿射变换（复数增益 + 偏置）。

    归一化方式：
    z_norm = (z - mean) / std
    其中 mean = E[Re(z)] + i*E[Im(z)]，std 基于模长计算

    参考：Deep Complex Networks (Trabelsi et al., 2018) 的复数 BN 思路，
    但适配 LayerNorm 的归一化策略。

    Args:
        normalized_shape: 归一化的特征维度大小
        eps: 数值稳定 epsilon
        elementwise_affine: 是否使用可学习的仿射参数
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
    ):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            # 复数仿射参数：gamma（增益）和 beta（偏置）均为复数
            self.gamma = nn.Parameter(
                torch.ones(normalized_shape, dtype=torch.complex64)
            )
            self.beta = nn.Parameter(
                torch.zeros(normalized_shape, dtype=torch.complex64)
            )
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: 复数张量 (..., normalized_shape)
        Returns:
            归一化后的复数张量 (..., normalized_shape)
        """
        # 分别计算实部和虚部的均值和方差
        mean = z.mean(dim=-1, keepdim=True)
        # 使用模长方差作为复数方差度量
        variance = (z - mean).abs().pow(2).mean(dim=-1, keepdim=True)
        z_norm = (z - mean) / torch.sqrt(variance + self.eps)

        if self.elementwise_affine:
            z_norm = self.gamma * z_norm + self.beta

        return z_norm

    def extra_repr(self) -> str:
        return f'normalized_shape={self.normalized_shape}, eps={self.eps}'


class ComplexLayerNormV2(nn.Module):
    """替代方案：基于复数均值的 LayerNorm。

    将复数张量视为 2d 维实数向量做 LayerNorm，
    然后重新组装为复数。
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
    ):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(2 * normalized_shape))
            self.bias = nn.Parameter(torch.zeros(2 * normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # 拆分实部和虚部
        z_flat = torch.cat([z.real, z.imag], dim=-1)

        # 标准 LayerNorm
        mean = z_flat.mean(dim=-1, keepdim=True)
        var = z_flat.var(dim=-1, keepdim=True, unbiased=False)
        z_norm = (z_flat - mean) / torch.sqrt(var + self.eps)

        if self.weight is not None:
            z_norm = self.weight * z_norm + self.bias

        # 重新组装为复数
        d = self.normalized_shape
        return torch.complex(z_norm[..., :d], z_norm[..., d:])


class ComplexBatchNorm(nn.Module):
    """复数 Batch Normalization。

    对复数张量的实部和虚部分别做 BatchNorm，
    保持复数运算的统计特性。

    参考：Deep Complex Networks (Trabelsi et al., 2018)

    Args:
        num_features: 特征维度大小
        eps: 数值稳定 epsilon
        momentum: BN 动量
        affine: 是否使用仿射变换
        track_running_stats: 是否追踪运行统计量
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        # 分别对实部和虚部做 BN
        self.bn_real = nn.BatchNorm1d(num_features, eps, momentum, affine=False, track_running_stats=track_running_stats)
        self.bn_imag = nn.BatchNorm1d(num_features, eps, momentum, affine=False, track_running_stats=track_running_stats)

        if affine:
            # 复数仿射参数
            self.gamma = nn.Parameter(
                torch.ones(num_features, dtype=torch.complex64)
            )
            self.beta = nn.Parameter(
                torch.zeros(num_features, dtype=torch.complex64)
            )
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: 复数张量 (batch, num_features) 或 (batch, num_features, seq_len)
        Returns:
            归一化后的复数张量
        """
        orig_shape = z.shape

        # BatchNorm1d 需要 (batch, features) 或 (batch, features, length)
        if z.dim() == 3:
            # (batch, seq_len, features) -> (batch, features, seq_len)
            z = z.transpose(1, 2)

        # 分别归一化实部和虚部
        z_real = self.bn_real(z.real)
        z_imag = self.bn_imag(z.imag)
        z_norm = torch.complex(z_real, z_imag)

        # 恢复原始形状
        if len(orig_shape) == 3:
            z_norm = z_norm.transpose(1, 2)

        # 应用复数仿射变换
        if self.affine:
            z_norm = self.gamma * z_norm + self.beta

        return z_norm

    def extra_repr(self) -> str:
        return (
            f'num_features={self.num_features}, eps={self.eps}, '
            f'momentum={self.momentum}, affine={self.affine}'
        )


class MagnitudeBatchNorm(nn.Module):
    """基于模长的 BatchNorm（保持相位不变）。

    对复数张量的模长做 BN，然后重新缩放：

    1. 分解 z = |z| * exp(i*arg(z))
    2. 对 |z| 做 BN: |z|_norm = BN(|z|)
    3. 重新合成: z_norm = |z|_norm * exp(i*arg(z))

    Args:
        num_features: 特征维度大小
        eps: 数值稳定 epsilon
        momentum: BN 动量
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
    ):
        super().__init__()
        self.eps = eps
        self.bn_mag = nn.BatchNorm1d(num_features, eps, momentum, affine=True)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: 复数张量 (batch, num_features, seq_len) 或 (batch, num_features)
        Returns:
            模长归一化后的复数张量，相位保持不变
        """
        magnitude = z.abs()
        phase = z.angle()

        orig_shape = z.shape
        if z.dim() == 3:
            magnitude = magnitude.transpose(1, 2)

        mag_norm = self.bn_mag(magnitude)

        if len(orig_shape) == 3:
            mag_norm = mag_norm.transpose(1, 2)

        return magnitude.new_complex(mag_norm, 0) * torch.exp(1j * phase)
