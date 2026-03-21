"""
QCI 量子坍缩推理 (Quantum Collapse Inference)

核心机制：
1. 不确定性度量：通过冯诺依曼熵量化量子态的不确定性
2. 自适应早退：高确定性快速退出，低确定性继续计算
3. POVM 测量：正定算子值测量算子实现概率性推理

早退规则：
- H < τ_low：高置信度，提前坍缩输出
- τ_low ≤ H ≤ τ_high：正常处理
- H > τ_high：低置信度，继续完整前向传播
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .complex_ops import (
    born_normalize,
    von_neumann_entropy,
    normalize_quantum_state,
    max_entropy,
)


class POVMProjector(nn.Module):
    """POVM 测量投影器。

    将高维量子态投影到低维确定态，模拟波函数坍缩。

    M_m = |e_m⟩⟨e_m|  →  P(m) = |⟨e_m|ψ⟩|²

    通过可学习的测量基实现：
    - 测量基为复数向量 {φ_m}
    - P(m) = |⟨φ_m|ψ⟩|²

    Args:
        in_dim: 输入量子态维度
        out_dim: 输出（坍缩后）维度
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # 可学习的测量基（正交化）
        self.measurement_basis = nn.Parameter(
            torch.randn(out_dim, in_dim, dtype=torch.complex64) * 0.02
        )

    def forward(
        self, psi: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """对量子态应用 POVM 测量。

        Args:
            psi: 量子态 (..., in_dim)
        Returns:
            (collapsed_state, probabilities)
            collapsed_state: 坍缩后的态 (..., in_dim)
            probabilities: 测量概率 (..., out_dim)
        """
        # 测量：⟨φ_m | ψ⟩
        inner = psi @ self.measurement_basis.conj().T  # (..., out_dim)

        # 概率：P(m) = |⟨φ_m|ψ⟩|² / Σ|⟨φ_m|ψ⟩|²
        probs = born_normalize(inner, dim=-1)  # (..., out_dim) 实数

        # 坍缩：将概率最高的基矢作为投影方向
        # collapsed = Σ_m p_m * φ_m（实数概率 × 复数基矢 = 复数态）
        collapsed = (probs.unsqueeze(-1) * self.measurement_basis).sum(dim=-2)  # (..., in_dim)

        # 重新归一化
        collapsed = normalize_quantum_state(collapsed, dim=-1)

        return collapsed, probs

    def extra_repr(self) -> str:
        return f'in_dim={self.in_dim}, out_dim={self.out_dim}'


class QuantumCollapseInference(nn.Module):
    """量子坍缩推理 (QCI)。

    基于不确定性的自适应推理模块：
    1. 计算量子态的冯诺依曼熵（不确定性度量）
    2. 根据阈值决定是否早退
    3. 早退时通过 POVM 测量坍缩到低维态

    Args:
        dim: 特征维度
        tau_low: 低阈值（低于此值早退）
        tau_high: 高阈值（高于此值需要更多计算）
        collapse_dim: 坍缩后的维度（None 表示不降维）
    """

    def __init__(
        self,
        dim: int,
        tau_low: float = 0.5,
        tau_high: float = 1.5,
        collapse_dim: Optional[int] = None,
    ):
        super().__init__()
        self.dim = dim
        self.tau_low = tau_low
        self.tau_high = tau_high
        self.collapse_dim = collapse_dim or dim

        # POVM 测量算子（用于坍缩）
        self.povm = POVMProjector(dim, self.collapse_dim)

    @torch.no_grad()
    def compute_uncertainty(self, x: torch.Tensor) -> torch.Tensor:
        """计算量子态的不确定性（冯诺依曼熵）。

        Args:
            x: 复数张量 (..., dim)
        Returns:
            熵值 (...)，范围 [0, log(dim)]
        """
        probs = born_normalize(x, dim=-1)
        return von_neumann_entropy(probs, dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        training: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            x: 复数输入 (batch, seq_len, dim)
            training: 是否训练模式
        Returns:
            (output, metrics)
            output: (batch, seq_len, dim) 或 (batch, seq_len, collapse_dim)
            metrics: 坍缩统计信息
        """
        B, N, D = x.shape

        # 计算每个 token 的不确定性
        entropy = self.compute_uncertainty(x)  # (B, N)

        avg_entropy = entropy.mean().item()
        max_entropy_val = max_entropy(D)

        # 决定每个 token 的处理方式
        if training:
            # 训练时：使用 soft 版本（所有 token 都继续，但高确定性 token 赋予更小权重）
            confidence = torch.sigmoid(-(entropy - self.tau_low) * 5)  # 高确定性 -> 高置信度

            # Soft 坍缩：混合原始态和坍缩态
            collapsed, probs = self.povm(x)
            # 使用置信度作为混合权重
            output = confidence.unsqueeze(-1) * x + (1 - confidence.unsqueeze(-1)) * collapsed

            # 如果 collapse_dim != dim，投影回原始维度
            if self.collapse_dim != D:
                output = F.pad(output, (0, D - self.collapse_dim))

            early_exit_rate = (entropy < self.tau_low).float().mean().item()
        else:
            # 推理时：硬阈值决定
            should_collapse = entropy < self.tau_low  # (B, N)
            early_exit_mask = should_collapse.unsqueeze(-1)  # (B, N, 1)

            # 坍缩
            collapsed, probs = self.povm(x)
            if self.collapse_dim != D:
                collapsed = F.pad(collapsed, (0, D - self.collapse_dim))

            # 根据掩码选择
            output = torch.where(early_exit_mask, collapsed, x)
            early_exit_rate = should_collapse.float().mean().item()

        metrics = {
            'collapse_entropy': avg_entropy,
            'collapse_entropy_max': entropy.max().item(),
            'collapse_entropy_min': entropy.min().item(),
            'collapse_early_exit_rate': early_exit_rate,
            'collapse_tau_low': self.tau_low,
            'collapse_tau_high': self.tau_high,
            'collapse_max_entropy': max_entropy_val,
        }

        return output, metrics

    def update_thresholds(self, tau_low: float, tau_high: float):
        """更新坍缩阈值（由优化系统的 CollapseThresholdLearner 调用）。"""
        self.tau_low = tau_low
        self.tau_high = tau_high

    def extra_repr(self) -> str:
        return (
            f'dim={self.dim}, collapse_dim={self.collapse_dim}, '
            f'tau_low={self.tau_low}, tau_high={self.tau_high}'
        )
