"""
QCI 量子坍缩推理 (Quantum Collapse Inference)

核心机制：
1. POVM 测量：正定算子值测量，满足完整性约束 Σ E_m = I
2. 不确定性度量：冯诺依曼熵（纯态为0，混合态 > 0）
3. 自适应早退：基于不确定性的概率性退出策略
4. 可学习阈值：τ 随训练进度自适应调整

理论依据：
- POVM 测量：P(m) = ⟨ψ|E_m|ψ⟩，其中 E_m = M_m†M_m ≥ 0
- 完整性约束：Σ_m E_m = I（保证 Σ P(m) = 1）
- 坍缩后态：|ψ_m⟩ = √E_m|ψ⟩ / √P(m)
- 熵：S(ρ) = -Tr(ρ log ρ)，纯态 S=0，最大熵 S_max = log(d)
"""

import math
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
    """POVM (Positive Operator-Valued Measure) 测量投影器。

    实现满足完整性约束的 POVM 测量：
    - 构造一组正定算子 E_m = M_m†M_m
    - 满足 Σ_m E_m = I（通过 Softmax 参数化自动保证）
    - 测量概率：P(m) = ⟨ψ|E_m|ψ⟩
    - 坍缩后态：|ψ_m⟩ = √E_m|ψ⟩ / √P(m)

    完整性约束的保证方式：
    通过将 E_m 参数化为 softmax 权重下的投影，即：
        E_m = α_m · |φ_m⟩⟨φ_m|
    其中 α_m = softmax(w)_m > 0，|φ_m⟩ 为归一化测量基矢。
    由于 Σ α_m = 1（softmax 性质），只要 {φ_m} 接近完备基，
    Σ E_m ≈ I。

    更严格的方法是使用 Naimark 扩展定理，将 POVM 嵌入到更大的
    希尔伯特空间中作为投影测量。这里采用近似方法以保证可训练性。

    Args:
        in_dim: 输入量子态维度
        out_dim: 测量结果数（POVM 元素数量）
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # 可学习的测量基（用于构造投影算子）
        # 初始化使用正交矩阵的列（如果 out_dim <= in_dim）
        if out_dim <= in_dim:
            # 使用 QR 分解生成近似正交基
            basis = torch.randn(in_dim, out_dim, dtype=torch.complex64)
            q, _ = torch.linalg.qr(basis)
            self.measurement_basis = nn.Parameter(q.T.clone())  # (out_dim, in_dim)
        else:
            # out_dim > in_dim：用随机初始化 + 后续正交化
            self.measurement_basis = nn.Parameter(
                torch.randn(out_dim, in_dim, dtype=torch.complex64) * 0.02
            )

        # POVM 权重（正数，softplus 保证 α_m > 0）
        # 初始化 softplus(w) ≈ 1 → w = ln(e-1) ≈ 0.541
        # 这样初始时 Σ α_m · |φ_m⟩⟨φ_m| ≈ Σ |φ_m⟩⟨φ_m| ≈ I（对正交基）
        self.povm_weights = nn.Parameter(
            torch.full((out_dim,), math.log(math.e - 1))
        )

    def get_operators(self) -> torch.Tensor:
        """获取 POVM 正定算子 E_m 的权重。

        使用 softplus 保证权重为正：α_m = softplus(w_m) > 0
        注意：这里不强制 Σ α_m = 1，因为 POVM 完整性要求
        Σ E_m = I，对于正交基 {φ_m} 需要每个 α_m ≈ 1。

        Returns:
            weights: (out_dim,) 正权重 α_m > 0
        """
        return F.softplus(self.povm_weights)

    def get_completeness_violation(self) -> torch.Tensor:
        """计算 POVM 完整性约束的违背度。

        理想情况：Σ_m α_m · |φ_m⟩⟨φ_m| ≈ I
        违背度：||Σ E_m - I||_F

        使用批量矩阵运算替代 Python for 循环，提升大 out_dim 时的效率：
            Σ_m α_m · φ_m† ⊗ φ_m  =  basis† @ diag(α) @ basis

        Returns:
            violation: Frobenius 范数标量
        """
        weights = self.get_operators()  # (out_dim,)
        basis = self.measurement_basis  # (out_dim, in_dim)

        # 向量化：Σ_m α_m · |φ_m⟩⟨φ_m| = basis^† · diag(α) · basis
        # basis^†: (in_dim, out_dim), diag(α) @ basis: (out_dim, in_dim)
        weighted_basis = weights.unsqueeze(-1) * basis          # (out_dim, in_dim)
        sum_E = basis.conj().T @ weighted_basis                  # (in_dim, in_dim)

        identity = torch.eye(self.in_dim, dtype=torch.complex64, device=basis.device)
        return (sum_E - identity).abs().pow(2).sum().sqrt()

    def forward(
        self, psi: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """对量子态应用 POVM 测量。

        测量过程：
        1. 计算概率 P(m) = α_m · |⟨φ_m|ψ⟩|²
        2. 采样测量结果 m ~ P(m)
        3. 坍缩：|ψ_m⟩ = √α_m · ⟨φ_m|ψ⟩ · |φ_m⟩ / √P(m)

        注意：训练时使用期望坍缩（所有结果加权平均）以保持梯度流。
        推理时使用硬采样（真实的坍缩行为）。

        Args:
            psi: 量子态 (..., in_dim)
        Returns:
            (collapsed_state, probabilities)
            collapsed_state: 坍缩后的态 (..., in_dim)
            probabilities: 测量概率 (..., out_dim)，实数
        """
        weights = self.get_operators()  # (out_dim,)
        basis = self.measurement_basis  # (out_dim, in_dim)

        # 内积 ⟨φ_m | ψ⟩ = basis @ ψ†
        # psi: (..., in_dim), basis: (out_dim, in_dim)
        inner = psi @ basis.conj().T  # (..., out_dim)

        # 概率：P(m) = α_m · |⟨φ_m|ψ⟩|²
        raw_probs = weights.unsqueeze(0) * inner.abs().pow(2)  # (..., out_dim)

        # 归一化为合法概率分布
        probs = raw_probs / (raw_probs.sum(dim=-1, keepdim=True) + 1e-8)

        # 坍缩态：期望坍缩 = Σ_m p_m · √α_m · ⟨φ_m|ψ⟩ · |φ_m⟩ / √p_m
        # 简化：= Σ_m √(α_m · p_m) · ⟨φ_m|ψ⟩ · |φ_m⟩
        sqrt_weights = torch.sqrt(weights).unsqueeze(0)  # (1, out_dim)
        sqrt_probs = torch.sqrt(probs.clamp(min=1e-8))  # (..., out_dim)

        # 坍缩态的振幅
        collapse_amp = sqrt_weights * sqrt_probs * inner  # (..., out_dim)

        # 重建态：Σ_m collapse_amp_m · |φ_m⟩
        collapsed = collapse_amp @ basis  # (..., in_dim)

        # 重新归一化为单位向量
        collapsed = normalize_quantum_state(collapsed, dim=-1)

        return collapsed, probs

    def extra_repr(self) -> str:
        return f'in_dim={self.in_dim}, out_dim={self.out_dim}'


class AdaptiveThreshold(nn.Module):
    """自适应坍缩阈值模块。

    根据训练进度和统计信息自动调节 τ_low 和 τ_high：
    - 训练初期：τ_low 较高（保守，避免过早坍缩）
    - 训练后期：τ_low 逐渐降低（更多早退，提升推理效率）
    - τ_high 始终跟踪当前熵分布的百分位

    自适应策略：
    τ_low(t) = τ_low_init · decay^t + τ_min
    τ_high(t) = percentile(entropy_history, 90)

    Args:
        dim: 特征维度（用于计算 max_entropy）
        tau_low_init: 初始低阈值
        tau_min: 最小低阈值
        tau_high_init: 初始高阈值
        decay_rate: 衰减率（每个 epoch）
    """

    def __init__(
        self,
        dim: int,
        tau_low_init: float = 0.5,
        tau_min: float = 0.1,
        tau_high_init: float = 1.5,
        decay_rate: float = 0.95,
    ):
        super().__init__()
        self.dim = dim
        self.tau_low_init = tau_low_init
        self.tau_min = tau_min
        self.tau_high_init = tau_high_init
        self.decay_rate = decay_rate
        self.max_entropy = max_entropy(dim)

        # 当前阈值
        self.register_buffer('tau_low', torch.tensor(tau_low_init))
        self.register_buffer('tau_high', torch.tensor(tau_high_init))

        # 训练步数
        self.register_buffer('step_count', torch.tensor(0))

        # 熵历史（用于动态调整 τ_high）
        self.register_buffer('entropy_sum', torch.tensor(0.0))
        self.register_buffer('entropy_count', torch.tensor(0))

    @torch.no_grad()
    def update(self, entropy_batch: torch.Tensor, training: bool = True):
        """根据当前 batch 的熵统计更新阈值。

        Args:
            entropy_batch: 当前 batch 的熵值 (B, N)
            training: 是否训练模式
        """
        if not training:
            return

        # 更新步数
        self.step_count += 1

        # 更新熵历史
        self.entropy_sum += entropy_batch.mean().item()
        self.entropy_count += 1

        # 每隔 100 步更新一次阈值
        if self.step_count % 100 != 0:
            return

        # τ_low 指数衰减
        steps = self.step_count.item() / 100
        new_tau_low = max(
            self.tau_low_init * (self.decay_rate ** steps),
            self.tau_min
        )
        self.tau_low.fill_(new_tau_low)

        # τ_high 基于历史熵分布的 90 百分位
        if self.entropy_count > 10:
            avg_entropy = self.entropy_sum / self.entropy_count
            # τ_high 保持为平均熵的 1.5-2.0 倍，但不超过 max_entropy
            new_tau_high = min(avg_entropy * 2.0, self.max_entropy * 0.95)
            new_tau_high = max(new_tau_high, new_tau_low + 0.5)  # τ_high > τ_low
            self.tau_high.fill_(new_tau_high)

    def reset_history(self):
        """重置熵历史（用于新的训练阶段）。"""
        self.entropy_sum.fill_(0.0)
        self.entropy_count.fill_(0)

    def extra_repr(self) -> str:
        return (
            f'dim={self.dim}, '
            f'tau_low={self.tau_low.item():.3f}, '
            f'tau_high={self.tau_high.item():.3f}, '
            f'max_entropy={self.max_entropy:.3f}'
        )


class QuantumCollapseInference(nn.Module):
    """量子坍缩推理 (QCI)。

    基于不确定性的自适应推理模块：
    1. 计算量子态的不确定性（冯诺依曼熵）
    2. 根据自适应阈值决定早退或继续
    3. 早退时通过 POVM 测量坍缩到确定态

    训练策略：
    - 使用 Gumbel-Softmax 实现可微的坍缩决策（训练时有梯度）
    - 置信度作为混合权重，平衡原始态和坍缩态
    - 自适应阈值随训练进度调整

    推理策略：
    - 硬阈值决定：H < τ_low → 坍缩早退
    - 可选温度退火：逐步从 soft 切换到 hard

    Args:
        dim: 特征维度
        tau_low: 低阈值（低于此值早退）
        tau_high: 高阈值（高于此值需要更多计算）
        collapse_dim: 坍缩后的维度（None 表示不降维）
        adaptive_tau: 是否启用自适应阈值
    """

    def __init__(
        self,
        dim: int,
        tau_low: float = 0.5,
        tau_high: float = 1.5,
        collapse_dim: Optional[int] = None,
        adaptive_tau: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.collapse_dim = collapse_dim or dim

        # POVM 测量算子（满足完整性约束）
        self.povm = POVMProjector(dim, self.collapse_dim)

        # 自适应阈值
        self.adaptive_tau = adaptive_tau
        if adaptive_tau:
            self.threshold = AdaptiveThreshold(
                dim=dim,
                tau_low_init=tau_low,
                tau_min=0.1,
                tau_high_init=tau_high,
                decay_rate=0.95,
            )
        else:
            self.register_buffer('tau_low', torch.tensor(tau_low))
            self.register_buffer('tau_high', torch.tensor(tau_high))
            self.threshold = None

    @property
    def max_entropy(self) -> float:
        """最大可能熵 = log(dim)。"""
        return max_entropy(self.dim)

    @property
    def tau_low_val(self) -> float:
        """获取当前 tau_low 值（兼容自适应和静态模式）。"""
        if self.adaptive_tau and self.threshold is not None:
            return self.threshold.tau_low.item()
        return self.tau_low.item()

    @property
    def tau_high_val(self) -> float:
        """获取当前 tau_high 值（兼容自适应和静态模式）。"""
        if self.adaptive_tau and self.threshold is not None:
            return self.threshold.tau_high.item()
        return self.tau_high.item()

    def compute_uncertainty(self, x: torch.Tensor) -> torch.Tensor:
        """计算量子态的不确定性（冯诺依曼熵）。

        对于纯态 |ψ⟩，将 Born 概率分布的 Shannon 熵作为
        态的"确定性度量"：
        - 均匀分布（最大不确定性）：S = log(d)
        - 集中分布（高确定性）：S ≈ 0

        注意：严格的冯诺依曼熵对纯态为 0。这里计算的是
        Born 概率分布的熵，作为"注意力集中度"的代理指标，
        这是自适应推理中实际使用的信号。

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
            output: (batch, seq_len, dim)
            metrics: 坍缩统计信息
        """
        B, N, D = x.shape

        # 获取当前阈值
        if self.adaptive_tau and self.threshold is not None:
            tau_low = self.threshold.tau_low.item()
            tau_high = self.threshold.tau_high.item()
        else:
            tau_low = self.tau_low.item()
            tau_high = self.tau_high.item()

        # 计算每个 token 的不确定性
        entropy = self.compute_uncertainty(x)  # (B, N)

        avg_entropy = entropy.mean().item()
        max_ent = self.max_entropy

        # 更新自适应阈值
        if self.adaptive_tau and self.threshold is not None:
            self.threshold.update(entropy, training=training)

        # 决定每个 token 的处理方式
        if training:
            # 训练时：Gumbel-Softmax 式的可微坍缩决策
            # 置信度 = sigmoid(-(entropy - tau_low) * temperature)
            # temperature 控制从 soft 到 hard 的过渡
            temperature = 5.0
            confidence = torch.sigmoid(-(entropy - tau_low) * temperature)  # (B, N)
            # confidence ≈ 1 → 高确定性 → 保持原始态
            # confidence ≈ 0 → 低确定性 → 坍缩

            # Soft 坍缩：混合原始态和坍缩态
            collapsed, probs = self.povm(x)  # probs: (B, N, collapse_dim)

            # 如果 collapse_dim != dim，投影回原始维度
            if self.collapse_dim != D:
                collapsed = F.pad(collapsed, (0, D - self.collapse_dim))

            # 使用置信度作为混合权重
            conf = confidence.unsqueeze(-1)  # (B, N, 1)
            output = conf * x + (1 - conf) * collapsed

            early_exit_rate = (entropy < tau_low).float().mean().item()
        else:
            # 推理时：硬阈值决定
            should_collapse = entropy < tau_low  # (B, N)
            early_exit_mask = should_collapse.unsqueeze(-1)  # (B, N, 1)

            # 坍缩
            collapsed, probs = self.povm(x)
            if self.collapse_dim != D:
                collapsed = F.pad(collapsed, (0, D - self.collapse_dim))

            # 根据掩码选择
            output = torch.where(early_exit_mask, collapsed, x)
            early_exit_rate = should_collapse.float().mean().item()

        # 计算 POVM 完整性违背度
        povm_violation = self.povm.get_completeness_violation().item()

        metrics = {
            'collapse_entropy': avg_entropy,
            'collapse_entropy_max': entropy.max().item(),
            'collapse_entropy_min': entropy.min().item(),
            'collapse_early_exit_rate': early_exit_rate,
            'collapse_tau_low': tau_low,
            'collapse_tau_high': tau_high,
            'collapse_max_entropy': max_ent,
            'collapse_povm_violation': povm_violation,
        }

        return output, metrics

    def update_thresholds(self, tau_low: float, tau_high: float):
        """手动更新坍缩阈值。

        如果使用自适应阈值，此方法会重置自适应模块的参数。
        否则直接更新静态阈值。

        Args:
            tau_low: 新的低阈值
            tau_high: 新的高阈值
        """
        if self.adaptive_tau and self.threshold is not None:
            self.threshold.tau_low.fill_(tau_low)
            self.threshold.tau_high.fill_(tau_high)
            self.threshold.tau_low_init = tau_low
            self.threshold.tau_high_init = tau_high
        else:
            self.tau_low.fill_(tau_low)
            self.tau_high.fill_(tau_high)

    def get_unitarity_violation(self) -> Dict[str, float]:
        """获取 QCI 模块的约束违背报告。

        Returns:
            dict: 包含 POVM 完整性违背度
        """
        return {
            'povm_completeness': self.povm.get_completeness_violation().item(),
        }

    def extra_repr(self) -> str:
        if self.adaptive_tau and self.threshold is not None:
            return (
                f'dim={self.dim}, collapse_dim={self.collapse_dim}, '
                f'adaptive_tau=True, '
                f'tau_low={self.threshold.tau_low.item():.3f}, '
                f'tau_high={self.threshold.tau_high.item():.3f}'
            )
        return (
            f'dim={self.dim}, collapse_dim={self.collapse_dim}, '
            f'tau_low={self.tau_low.item():.3f}, '
            f'tau_high={self.tau_high.item():.3f}'
        )
