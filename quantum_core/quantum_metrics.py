"""
量子度量工具集 (Quantum Metrics)

汇集 QuantumArch 各模块所需的量子信息度量函数，提供统一接口：

1. **纠缠度量（Entanglement Measures）**
   - Concurrence（纠缠并发度）
   - Schmidt 数估计（纠缠维度）
   - 纠缠熵

2. **信息度量（Information Measures）**
   - von Neumann 熵（量子不确定性）
   - 量子 KL 散度
   - 量子互信息（两态联合信息）

3. **量子相似性（Quantum Similarity）**
   - 量子保真度 (Fidelity)
   - 迹距离 (Trace Distance)
   - Bures 距离

4. **注意力质量度量（Attention Quality）**
   - 注意力熵（集中度）
   - 有效注意力宽度（等效关注 token 数）
   - 注意力稀疏率

所有函数均支持批量操作（batch 维度透明），且使用 `@torch.no_grad()`
标记可选的纯推理模式，方便在训练循环中以零梯度开销收集诊断信息。

数学参考：
    Nielsen & Chuang, "Quantum Computation and Quantum Information" (2000)
    第9章：量子信息论
"""

import math
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


# ──────────────────────────────────────────────
# 纠缠度量
# ──────────────────────────────────────────────


def concurrence_from_amplitudes(
    a: torch.Tensor,
    b: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """从两个量子态振幅计算纠缠并发度（Concurrence 近似）。

    对于两个纯态 |a⟩, |b⟩ ∈ ℂ^d，使用以下近似公式：
        C(a, b) ≈ 1 - |⟨a|b⟩|²

    - C = 0：两态相同（无纠缠，可分离）
    - C = 1：两态正交（最大纠缠）

    注：严格的 Concurrence 定义仅适用于双量子比特系统，
    本函数给出高维情况下的 overlap-based 近似。

    Args:
        a: 量子态 (..., d)，复数
        b: 量子态 (..., d)，复数
        dim: 内积计算维度
        eps: 归一化数值稳定量

    Returns:
        纠缠度量 (...)，范围 [0, 1]，越大表示越纠缠
    """
    # 归一化
    a_norm_sq = a.abs().pow(2).sum(dim=dim, keepdim=True).clamp(min=eps).sqrt()
    b_norm_sq = b.abs().pow(2).sum(dim=dim, keepdim=True).clamp(min=eps).sqrt()
    a_n = a / a_norm_sq
    b_n = b / b_norm_sq

    # 复数内积 ⟨a|b⟩
    inner = (a_n.conj() * b_n).sum(dim=dim)  # (...) 复数
    overlap = inner.abs().pow(2)  # |⟨a|b⟩|² 实数

    return 1.0 - overlap.clamp(0.0, 1.0)


def schmidt_number_estimate(
    state: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """估计量子态的 Schmidt 数（有效纠缠维度）。

    Schmidt 数 K 衡量纠缠涉及的有效子空间维度：
    - K = 1：可分离态（无纠缠）
    - K = d：最大纠缠（均匀叠加）

    使用逆参与比（Inverse Participation Ratio, IPR）的倒数近似：
        K ≈ 1 / Σ_i P_i²

    其中 P_i = |α_i|² / Σ_j |α_j|² 是 Born 概率。

    Args:
        state: 量子态振幅 (..., d)，复数
        dim: 振幅维度
        eps: 数值稳定量

    Returns:
        Schmidt 数估计 (...)，范围 [1, d]（实数）
    """
    probs = state.abs().pow(2)
    probs = probs / probs.sum(dim=dim, keepdim=True).clamp(min=eps)
    ipr = probs.pow(2).sum(dim=dim)  # Σ P_i²
    return 1.0 / ipr.clamp(min=eps)


# ──────────────────────────────────────────────
# 信息度量
# ──────────────────────────────────────────────


def quantum_entropy(
    probs: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """von Neumann 熵（Shannon 熵近似）。

    S(ρ) = -Σ_i p_i · log(p_i)

    对于 Born 概率分布 {p_i = |α_i|²}，此函数计算其 Shannon 熵，
    作为量子不确定性的度量。

    范围：[0, log(d)]
    - S = 0：纯态，完全确定
    - S = log(d)：最大混合态，完全不确定

    Args:
        probs: 概率分布 (..., d)，实数，应满足 sum=1
        dim: 概率维度
        eps: 避免 log(0) 的偏置量

    Returns:
        熵值 (...)，实数
    """
    safe_probs = probs.clamp(min=eps)
    return -(safe_probs * safe_probs.log()).sum(dim=dim)


def quantum_mutual_information(
    a: torch.Tensor,
    b: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """计算两个量子态之间的量子互信息近似。

    量子互信息 I(A:B) = S(A) + S(B) - S(AB)

    本实现使用以下近似：
    - S(A) 和 S(B)：各自 Born 概率的 Shannon 熵
    - S(AB)：联合态 |a⟩ ⊗ |b⟩ 的 Born 概率熵（取前 min(da, db) 维近似）

    注意：这是一个近似度量，完整的量子互信息需要密度矩阵计算。

    Args:
        a: 量子态 (..., da)，复数
        b: 量子态 (..., db)，复数
        dim: 振幅维度（a 和 b 需相同维度）
        eps: 数值稳定量

    Returns:
        互信息近似 (...)，≥ 0
    """
    from .complex_ops import born_normalize

    probs_a = born_normalize(a, dim=dim, eps=eps)
    probs_b = born_normalize(b, dim=dim, eps=eps)

    s_a = quantum_entropy(probs_a, dim=dim, eps=eps)
    s_b = quantum_entropy(probs_b, dim=dim, eps=eps)

    # 联合态近似：取 |a|²·|b|² 的外积对角近似（元素积）
    joint_probs_approx = probs_a * probs_b  # (..., d)，近似联合分布
    joint_probs_approx = joint_probs_approx / joint_probs_approx.sum(
        dim=dim, keepdim=True
    ).clamp(min=eps)
    s_ab = quantum_entropy(joint_probs_approx, dim=dim, eps=eps)

    # I(A:B) = S(A) + S(B) - S(AB)，非负
    return (s_a + s_b - s_ab).clamp(min=0.0)


# ──────────────────────────────────────────────
# 量子相似性
# ──────────────────────────────────────────────


def quantum_fidelity(
    rho: torch.Tensor,
    sigma: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """计算两个量子态之间的保真度（Fidelity）。

    对于纯态 |ψ⟩ 和 |φ⟩：
        F(ψ, φ) = |⟨ψ|φ⟩|²

    保真度范围 [0, 1]：
    - F = 1：两态相同
    - F = 0：两态正交

    Args:
        rho: 第一个量子态振幅 (..., d)，复数
        sigma: 第二个量子态振幅 (..., d)，复数
        dim: 内积计算维度
        eps: 归一化数值稳定量

    Returns:
        保真度 (...)，范围 [0, 1]，实数
    """
    # 归一化
    rho_n = rho / rho.abs().pow(2).sum(dim=dim, keepdim=True).clamp(min=eps).sqrt()
    sigma_n = sigma / sigma.abs().pow(2).sum(dim=dim, keepdim=True).clamp(min=eps).sqrt()

    inner = (rho_n.conj() * sigma_n).sum(dim=dim)
    return inner.abs().pow(2).clamp(0.0, 1.0)


def trace_distance(
    rho: torch.Tensor,
    sigma: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """计算两个量子态之间的迹距离（Trace Distance）。

    对于纯态，迹距离与保真度的关系：
        T(ψ, φ) = √(1 - F(ψ, φ))

    迹距离范围 [0, 1]：
    - T = 0：两态相同
    - T = 1：两态正交（完全可区分）

    物理意义：最优量子测量能以 (1 + T) / 2 的成功概率区分两个态。

    Args:
        rho: 第一个量子态振幅 (..., d)，复数
        sigma: 第二个量子态振幅 (..., d)，复数
        dim: 内积计算维度
        eps: 数值稳定量

    Returns:
        迹距离 (...)，范围 [0, 1]，实数
    """
    F = quantum_fidelity(rho, sigma, dim=dim, eps=eps)
    return (1.0 - F).clamp(min=0.0).sqrt()


def bures_distance(
    rho: torch.Tensor,
    sigma: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """计算两个量子态之间的 Bures 距离。

    Bures 距离是量子信息中最自然的量子态空间距离：
        B(ρ, σ) = √(2 - 2√F(ρ, σ))
                = √(2(1 - √F))

    范围 [0, √2]：
    - B = 0：两态相同
    - B = √2：两态正交

    Bures 距离满足量子版本的 Cramér-Rao 界，
    是量子参数估计中的基础度量。

    Args:
        rho: 第一个量子态振幅 (..., d)，复数
        sigma: 第二个量子态振幅 (..., d)，复数
        dim: 内积计算维度
        eps: 数值稳定量

    Returns:
        Bures 距离 (...)，范围 [0, √2]，实数
    """
    F = quantum_fidelity(rho, sigma, dim=dim, eps=eps)
    return (2.0 * (1.0 - F.clamp(min=0.0).sqrt())).clamp(min=0.0).sqrt()


# ──────────────────────────────────────────────
# 注意力质量度量
# ──────────────────────────────────────────────


def attention_effective_width(
    attn_probs: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """计算注意力分布的有效宽度（等效关注 token 数）。

    使用指数形式的 Shannon 熵（熵的指数）作为有效宽度：
        W_eff = exp(H(p))   where H(p) = -Σ p_i log p_i

    解释：
    - W_eff = 1：注意力完全集中在单个 token（delta 分布）
    - W_eff = N：注意力均匀分布在所有 token（均匀分布）

    这比直接用 "top-K 覆盖率" 更平滑，适合用于损失函数。

    Args:
        attn_probs: 注意力概率矩阵 (..., N)，实数
        dim: 概率维度
        eps: 数值稳定量

    Returns:
        有效宽度 (...)，范围 [1, N]，实数
    """
    h = quantum_entropy(attn_probs, dim=dim, eps=eps)
    return h.exp()


def attention_sparsity_ratio(
    attn_probs: torch.Tensor,
    threshold: float = 0.01,
    dim: int = -1,
) -> torch.Tensor:
    """计算注意力分布的稀疏率。

    稀疏率 = 概率低于 threshold 的位置比例：
        sparsity = |{i : p_i < threshold}| / N

    - sparsity → 1.0：注意力极度稀疏（仅关注少数几个 token）
    - sparsity → 0.0：注意力非常均匀

    Args:
        attn_probs: 注意力概率矩阵 (..., N)，实数
        threshold: 稀疏阈值
        dim: 概率维度

    Returns:
        稀疏率 (...)，范围 [0, 1]，实数
    """
    sparse_mask = (attn_probs < threshold).float()
    return sparse_mask.mean(dim=dim)


# ──────────────────────────────────────────────
# 综合诊断
# ──────────────────────────────────────────────


@torch.no_grad()
def compute_model_quantum_health(
    hidden_states: torch.Tensor,
    attn_probs: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """计算模型量子态的整体健康度摘要。

    对给定的隐藏状态张量，计算一系列量子信息度量，
    用于训练监控和异常检测。

    Args:
        hidden_states: 模型隐藏状态 (B, N, D)，复数
        attn_probs: 可选注意力概率矩阵 (B, H, N, N)，实数
        eps: 数值稳定量

    Returns:
        dict 包含:
            - ``state_entropy_mean``: 隐藏状态的平均 Born 概率熵
            - ``state_entropy_std``: 隐藏状态熵的标准差
            - ``state_mag_mean``: 隐藏状态模长均值（量子态能量）
            - ``state_mag_std``: 隐藏状态模长标准差
            - ``schmidt_number_mean``: 平均 Schmidt 数估计
            - ``attn_effective_width_mean``: 平均注意力有效宽度（若提供 attn_probs）
            - ``attn_sparsity_mean``: 平均注意力稀疏率（若提供 attn_probs）
    """
    from .complex_ops import born_normalize

    B, N, D = hidden_states.shape
    result: Dict[str, float] = {}

    # Born 概率熵（量子不确定性）
    probs = born_normalize(hidden_states, dim=-1, eps=eps)  # (B, N, D)
    entropies = quantum_entropy(probs, dim=-1, eps=eps)  # (B, N)
    result["state_entropy_mean"] = entropies.mean().item()
    result["state_entropy_std"] = entropies.std().item()

    # 模长统计
    magnitudes = hidden_states.abs().mean(dim=-1)  # (B, N)
    result["state_mag_mean"] = magnitudes.mean().item()
    result["state_mag_std"] = magnitudes.std().item()

    # Schmidt 数估计
    k_est = schmidt_number_estimate(hidden_states, dim=-1, eps=eps)  # (B, N)
    result["schmidt_number_mean"] = k_est.mean().item()

    # 注意力度量（可选）
    if attn_probs is not None:
        # attn_probs: (B, H, N, N)
        eff_width = attention_effective_width(attn_probs, dim=-1, eps=eps)  # (B, H, N)
        result["attn_effective_width_mean"] = eff_width.mean().item()

        sparsity = attention_sparsity_ratio(attn_probs, threshold=0.01, dim=-1)  # (B, H, N)
        result["attn_sparsity_mean"] = sparsity.mean().item()
    else:
        result["attn_effective_width_mean"] = float("nan")
        result["attn_sparsity_mean"] = float("nan")

    return result


# ──────────────────────────────────────────────
# 公开接口
# ──────────────────────────────────────────────

__all__ = [
    # 纠缠度量
    "concurrence_from_amplitudes",
    "schmidt_number_estimate",
    # 信息度量
    "quantum_entropy",
    "quantum_mutual_information",
    # 量子相似性
    "quantum_fidelity",
    "trace_distance",
    "bures_distance",
    # 注意力质量
    "attention_effective_width",
    "attention_sparsity_ratio",
    # 综合诊断
    "compute_model_quantum_health",
]
