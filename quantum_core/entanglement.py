"""
QEL 量子纠缠层 (Quantum Entanglement Layer)

核心思想：用张量积纠缠替代 Transformer 的残差连接。

操作流程：
1. 局部纠缠：相邻 token 对应用参数化纠缠门 U_ent(θ)
2. 全局纠缠：通过 QFT（量子傅里叶变换）建立长程关联
3. 自适应强度：纠缠强度 θ 根据输入动态调整

纠缠门：
    U_ent(θ) = | cosθ   i·sinθ |
               | i·sinθ  cosθ  |
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple

from .complex_ops import (
    normalize_quantum_state,
    interference_score,
    born_normalize,
)


class EntanglementGate(nn.Module):
    """参数化纠缠门 U_ent(θ)。

    2x2 酉矩阵，作用于两个量子态（token）：
        U_ent(θ) = | cosθ   i·sinθ |
                   | i·sinθ  cosθ  |

    满足 U_ent · U_ent† = I（酉性自动保证）。

    Args:
        init_theta: 初始纠缠角度
    """

    def __init__(self, init_theta: float = 0.1):
        super().__init__()
        # 可学习的纠缠角度（每个 token 对独立）
        self.logit_theta = nn.Parameter(
            torch.tensor(0.0)  # 通过 sigmoid 映射到 [0, π/2]
        )

    @property
    def theta(self) -> torch.Tensor:
        return torch.sigmoid(self.logit_theta) * (torch.pi / 2)

    def get_gate_matrix(self) -> torch.Tensor:
        """返回 2x2 纠缠门矩阵。

        Returns:
            U_ent: (2, 2) 复数酉矩阵
        """
        t = self.theta.item()
        cos_t = math.cos(t)
        sin_t = math.sin(t)
        # | cosθ    i·sinθ |
        # | i·sinθ  cosθ   |
        U = torch.tensor([
            [cos_t, 1j * sin_t],
            [1j * sin_t, cos_t],
        ], dtype=torch.complex64)
        return U

    def forward(
        self, a: torch.Tensor, b: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """对两个量子态应用纠缠门。

        Args:
            a: 量子态 (..., d)
            b: 量子态 (..., d)
        Returns:
            (a_out, b_out): 纠缠后的量子态
        """
        t = self.theta
        cos_t = torch.cos(t)
        sin_t = torch.sin(t)

        a_out = cos_t * a + 1j * sin_t * b
        b_out = 1j * sin_t * a + cos_t * b

        return a_out, b_out


class AdaptiveEntanglementGate(nn.Module):
    """自适应纠缠门：纠缠强度根据输入动态调整。

    θ_i = sigmoid(MLP([x_i; x_{i+1}])) * θ_max

    相关 token 对 -> 高纠缠；无关 token 对 -> 低纠缠。
    """

    def __init__(self, dim: int, theta_max: float = 1.0):
        super().__init__()
        self.theta_max = theta_max
        # 使用实数 MLP（从复数模长中提取强度）
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, a: torch.Tensor, b: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """自适应纠缠操作。

        Args:
            a: (B, N, d) 或 (..., d)
            b: 同上
        Returns:
            (a_out, b_out, strength): 纠缠后的态和纠缠强度
        """
        # 拼接两个 token 的模长作为 MLP 输入（实数）
        combined = torch.cat([a.abs(), b.abs()], dim=-1)  # (..., 2d) 实数

        # 动态计算纠缠强度
        strength = self.mlp(combined) * self.theta_max  # (..., 1) 实数

        cos_t = torch.cos(strength)
        sin_t = torch.sin(strength)

        a_out = cos_t * a + 1j * sin_t * b
        b_out = 1j * sin_t * a + cos_t * b

        return a_out, b_out, strength


class QuantumEntanglementLayer(nn.Module):
    """量子纠缠层 (QEL)。

    对序列中的 token 建立量子纠缠关联：
    1. 局部纠缠：相邻 token 对通过纠缠门耦合
    2. 全局纠缠：QFT 建立全局长程关联
    3. 自适应强度：MLP 控制纠缠强度

    Args:
        dim: 特征维度
        use_adaptive: 是否使用自适应纠缠强度
        use_global_qft: 是否使用 QFT 全局纠缠
        theta_max: 最大纠缠角度
    """

    def __init__(
        self,
        dim: int,
        use_adaptive: bool = True,
        use_global_qft: bool = True,
        theta_max: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.use_adaptive = use_adaptive
        self.use_global_qft = use_global_qft

        if use_adaptive:
            self.entangle_gate = AdaptiveEntanglementGate(dim, theta_max)
        else:
            self.entangle_gate = EntanglementGate(init_theta=0.1)

    def forward(
        self, x: torch.Tensor, training: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            x: 复数输入 (batch, seq_len, dim)
            training: 是否训练模式
        Returns:
            (output, metrics)
            output: 纠缠后的复数张量 (batch, seq_len, dim)
            metrics: 纠缠统计信息
        """
        B, N, D = x.shape
        metrics = {}

        # ── 1. 局部纠缠（相邻 token 对）──
        if N < 2:
            entangled = x
            avg_strength = torch.tensor(0.0)
        else:
            if self.use_adaptive:
                # 自适应纠缠
                x_even = x[:, 0::2, :]  # (B, N/2, D)
                x_odd = x[:, 1::2, :]   # (B, N/2, D)

                min_len = min(x_even.shape[1], x_odd.shape[1])
                x_even = x_even[:, :min_len, :]
                x_odd = x_odd[:, :min_len, :]

                x_even_out, x_odd_out, strengths = self.entangle_gate(x_even, x_odd)

                # 交错合并
                entangled = x.clone()
                entangled[:, 0::2, :][:, :min_len, :] = x_even_out
                entangled[:, 1::2, :][:, :min_len, :] = x_odd_out

                avg_strength = strengths.mean().item()
                metrics['entanglement_strength'] = avg_strength
            else:
                # 固定强度纠缠
                entangled = x.clone()
                for i in range(0, N - 1, 2):
                    a = x[:, i, :]
                    b = x[:, i + 1, :]
                    a_out, b_out = self.entangle_gate(a, b)
                    entangled[:, i, :] = a_out
                    entangled[:, i + 1, :] = b_out

                avg_strength = self.entangle_gate.theta.item()
                metrics['entanglement_strength'] = avg_strength

        # ── 2. 全局纠缠（QFT）──
        if self.use_global_qft and N > 1:
            entangled = self._qft_entangle(entangled)
            metrics['qft_applied'] = True

        # ── 3. 残差连接（信息融合）──
        # 设计文档中 QEL 替代残差连接，但实际用纠缠耦合：
        # |ψ_out⟩ = U_couple(|ψ_input⟩ ⊗ |ψ_entangled⟩) 的简化版
        # 使用加权平均融合
        output = (entangled + x) / (2 ** 0.5)

        return output, metrics

    def _qft_entangle(self, x: torch.Tensor) -> torch.Tensor:
        """通过 QFT 建立全局长程纠缠。

        使用 torch.fft.fft 在特征维度上应用离散傅里叶变换。

        Args:
            x: 复数张量 (batch, seq_len, dim)
        Returns:
            QFT 纠缠后的复数张量 (batch, seq_len, dim)
        """
        # 对特征维度做 FFT（等价于在 d 维希尔伯特空间中做 QFT）
        x_qft = torch.fft.fft(x, dim=-1)

        # 归一化（FFT 不改变模长的平方和，但 QFT 需要 1/sqrt(d) 缩放）
        d = x.shape[-1]
        x_qft = x_qft / (d ** 0.5)

        # 混合原始和 QFT 结果（部分纠缠）
        alpha = 0.5  # 混合系数
        return alpha * x + (1 - alpha) * x_qft

    def extra_repr(self) -> str:
        return (
            f'dim={self.dim}, adaptive={self.use_adaptive}, '
            f'global_qft={self.use_global_qft}'
        )
