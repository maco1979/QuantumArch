"""
复数基础运算工具
提供量子架构所需的所有复数数学操作。
"""

import torch
import torch.nn.functional as F
from typing import Optional


# ──────────────────────────────────────────────
# 复数分解与合成
# ──────────────────────────────────────────────


def complex_to_polar(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """将复数张量分解为模长和相位。

    Args:
        z: 复数张量 (*shape)
    Returns:
        (magnitude, phase): 模长和相位，均为实数张量 (*shape)
    """
    return z.abs(), z.angle()


def polar_to_complex(magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    """将模长和相位合成为复数张量。

    Args:
        magnitude: 实数模长 (*shape)
        phase: 实数相位 (*shape)
    Returns:
        复数张量 (*shape)
    """
    return magnitude * torch.exp(1j * phase)


def separate_mod_phase(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """分离可训练的模长和相位参数（用于QGD优化器）。

    与 complex_to_polar 相同，但强调语义为"参数分离"。
    """
    return z.abs(), z.angle()


# ──────────────────────────────────────────────
# 量子态归一化
# ──────────────────────────────────────────────


def normalize_quantum_state(z: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """将复数张量归一化为量子态（单位向量）。

    对指定维度做 L2 归一化：||z||_2 = 1

    Args:
        z: 复数张量 (..., d)
        dim: 归一化维度
        eps: 数值稳定 epsilon
    Returns:
        归一化后的复数张量，满足 sum(|z_i|^2) = 1
    """
    norm = z.abs().pow(2).sum(dim=dim, keepdim=True).sqrt().clamp(min=eps)
    return z / norm


def born_probability(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Born 法则：计算量子态的概率分布。

    P(i) = |α_i|² 其中 α_i 是复数振幅

    Args:
        z: 复数振幅张量 (..., d)
        dim: 概率计算维度
    Returns:
        实数概率张量 (..., d)，sum(dim=dim) 可能不等于1（需要归一化后使用）
    """
    return z.abs().pow(2)


def born_normalize(z: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """Born 概率 + 归一化：得到合法概率分布。

    P(i) = |α_i|² / Σ_j |α_j|²

    Args:
        z: 复数振幅张量 (..., d)
        dim: 概率计算维度
        eps: 数值稳定 epsilon
    Returns:
        归一化概率分布 (..., d)，sum(dim=dim) == 1
    """
    probs = z.abs().pow(2)
    return probs / probs.sum(dim=dim, keepdim=True).clamp(min=eps)


# ──────────────────────────────────────────────
# 熵与不确定性
# ──────────────────────────────────────────────


def von_neumann_entropy(probs: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """计算冯诺依曼熵（Shannon熵形式，用于概率分布）。

    S = -Σ P(i) log P(i)

    注意：对于纯态密度矩阵 ρ = |ψ⟩⟨ψ|，冯诺依曼熵为0。
    这里用于量化注意力分布或坍缩概率的"确定性"。

    Args:
        probs: 概率分布 (..., d)，需为正实数且和为1
        dim: 求和维度
        eps: 防止 log(0)
    Returns:
        标量熵值 (...)
    """
    log_probs = torch.log(probs.clamp(min=eps))
    entropy = -(probs * log_probs).sum(dim=dim)
    return entropy


def entropy_from_state(z: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """从复数量子态直接计算熵。

    等价于 von_neumann_entropy(born_normalize(z))

    Args:
        z: 复数振幅 (..., d)
        dim: 概率维度
        eps: 数值稳定 epsilon
    Returns:
        熵值 (...)
    """
    probs = born_normalize(z, dim=dim, eps=eps)
    return von_neumann_entropy(probs, dim=dim, eps=eps)


def max_entropy(dim: int) -> float:
    """均匀分布的最大熵 = log(dim)"""
    return float(torch.log(torch.tensor(dim, dtype=torch.float32)).item())


# ──────────────────────────────────────────────
# 复数内积与干涉
# ──────────────────────────────────────────────


def complex_inner_product(a: torch.Tensor, b: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """复数内积 ⟨a|b⟩ = Σ conj(a_i) * b_i

    Args:
        a: 复数张量 (..., d)
        b: 复数张量 (..., d)
        dim: 内积维度
    Returns:
        复数内积 (...)
    """
    return (a.conj() * b).sum(dim=dim)


def interference_score(
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """计算两个复数振幅之间的干涉强度。

    |α + β|² - (|α|² + |β|²) = 2 Re(conj(α)β)

    正值：相长干涉；负值：相消干涉

    Args:
        alpha: 复数振幅 (...)
        beta: 复数振幅 (...)
    Returns:
        实数干涉强度 (...)
    """
    return 2 * (alpha.conj() * beta).real


# ──────────────────────────────────────────────
# 复数 Softmax
# ──────────────────────────────────────────────


def complex_softmax(z: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """复数 Softmax。

    softmax_C(z)_i = exp(z_i) / Σ_j exp(z_j)

    性质：
    - |softmax_C(z)_i|² = softmax(Re(z))_i（概率分布由实部决定）
    - 保持原始相位信息

    Args:
        z: 复数张量 (..., d)
        dim: softmax 维度
        eps: 数值稳定
    Returns:
        复数归一化权重 (..., d)
    """
    # 数值稳定：减去最大实部
    z_real_max = z.real.max(dim=dim, keepdim=True).values
    z_stable = z - z_real_max
    exp_z = torch.exp(z_stable)
    return exp_z / (exp_z.abs().pow(2).sum(dim=dim, keepdim=True).sqrt().clamp(min=eps))


# ──────────────────────────────────────────────
# 复数 Dropout
# ──────────────────────────────────────────────


def complex_dropout(z: torch.Tensor, p: float = 0.1, training: bool = True) -> torch.Tensor:
    """复数 Dropout：同时对实部和虚部施加相同的 mask。

    Args:
        z: 复数张量
        p: drop 概率
        training: 是否训练模式
    Returns:
        dropout 后的复数张量
    """
    if not training or p == 0.0:
        return z
    mask = F.dropout(torch.ones_like(z.real), p=p, training=True)
    return z * mask


# ──────────────────────────────────────────────
# 酉性验证
# ──────────────────────────────────────────────


def complex_phase_shift(z: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """对复数张量施加相位旋转 e^{iθ}。

    等价于将每个振幅的相位增加 θ，模长不变。
    常用于量子电路中的 R_z(θ) 单量子比特旋转门。

    Args:
        z: 复数张量 (*shape)
        theta: 实数相位偏移 (*shape 或可广播)
    Returns:
        旋转后的复数张量 z · e^{iθ}
    """
    return z * torch.exp(1j * theta)


def real_to_complex(x: torch.Tensor) -> torch.Tensor:
    """将实数张量转为复数张量（虚部置零）。

    快捷工具，避免每处都写 torch.complex(x, torch.zeros_like(x))。

    Args:
        x: 实数张量 (*shape)
    Returns:
        复数张量 (*shape)，虚部为零
    """
    return torch.complex(x, torch.zeros_like(x))


def check_unitarity(W: torch.Tensor, eps: float = 1e-4) -> dict:
    """检查矩阵的酉性 W†W = I。

    Args:
        W: 方阵 (d, d)
        eps: 允许的偏差
    Returns:
        dict with keys: is_unitary, violation_norm, max_deviation
    """
    d = W.shape[-1]
    identity = torch.eye(d, dtype=W.dtype, device=W.device)
    product = W.conj().transpose(-2, -1) @ W
    deviation = product - identity
    violation = deviation.abs().pow(2).sum().sqrt().item()
    max_dev = deviation.abs().max().item()

    return {
        "is_unitary": violation < eps,
        "violation_norm": violation,
        "max_deviation": max_dev,
    }
