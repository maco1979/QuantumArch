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


def phase_coherence(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """计算复数张量沿指定维度的相位相干性（Phase Coherence）。

    相位相干性衡量一组复数振幅的相位是否"指向同一方向"，
    是量子干涉强度的重要指标：

        C = |mean(exp(i·φ_k))| = |Σ exp(i·φ_k)| / N

    其中 φ_k = arg(z_k) 为各振幅的相位。

    取值范围 [0, 1]：
    - C ≈ 1：高相干性，所有相位接近一致（相长干涉为主）
    - C ≈ 0：低相干性，相位随机分布（相消干涉为主）

    与 von_neumann_entropy 的对比：
    - 熵度量：概率分布的集中度（Born 概率 |z|²）
    - 相干性：相位分布的集中度（角度信息）

    Args:
        z: 复数张量 (..., d)
        dim: 求均值的维度
    Returns:
        实数相干性标量 (...)，范围 [0, 1]

    Example:
        >>> z = torch.tensor([1+0j, 0+1j, -1+0j], dtype=torch.complex64)
        >>> phase_coherence(z)  # ≈ 0.0（三相位均匀分布，相消）
        >>> z2 = torch.ones(4, dtype=torch.complex64)
        >>> phase_coherence(z2)  # ≈ 1.0（所有相位为 0，完全相长）
    """
    # 归一化到单位圆（仅保留相位信息）
    z_unit = z / (z.abs().clamp(min=1e-8))  # exp(i·φ_k)

    # 相位向量的平均模长
    coherence = z_unit.mean(dim=dim).abs()
    return coherence


def phase_gradient(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """计算相邻元素的相位差（相位梯度），用于检测相位连续性。

    相位梯度反映量子态相位的"变化速率"，
    在 QIR 路由和 QEL 纠缠质量分析中用于识别相位跳变。

    Args:
        z: 复数张量 (..., d)，d >= 2
        dim: 计算差分的维度（必须 >= 1 才有相邻元素）
    Returns:
        相位梯度张量 (..., d-1)，实数，单位：弧度

    Raises:
        ValueError: 如果指定维度长度 < 2
    """
    phase = z.angle()  # (..., d)
    # 沿 dim 计算相邻相位差
    dphase = torch.diff(phase, dim=dim)
    # 将相位差映射到 [-π, π]（主值）
    dphase = torch.remainder(dphase + torch.pi, 2 * torch.pi) - torch.pi
    return dphase


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


# ──────────────────────────────────────────────
# 量子信息度量
# ──────────────────────────────────────────────


def quantum_fidelity(psi: torch.Tensor, phi: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """计算两个纯量子态之间的保真度 (Fidelity)。

    保真度是量子信息中衡量两个量子态"相似程度"的核心指标：

        F(|ψ⟩, |φ⟩) = |⟨ψ|φ⟩|²

    取值范围 [0, 1]：
    - F = 1：两态完全相同（相同方向，可有全局相位差）
    - F = 0：两态完全正交（完全不同）
    - 0 < F < 1：部分重叠

    在量子架构中的用途：
    1. **QEL 监控**：纠缠前后的态保真度，低保真度意味着纠缠操作改变了态
    2. **QCI 评估**：坍缩前后的保真度，高保真度意味着坍缩代价低
    3. **KD 损失**：知识蒸馏时对齐教师/学生模型的量子态

    Args:
        psi: 量子态 (..., d)，复数或实数
        phi: 量子态 (..., d)，复数或实数
        dim: 内积维度（状态空间维度）
    Returns:
        保真度 (...)，实数，范围 [0, 1]

    Example:
        >>> psi = torch.tensor([1, 0], dtype=torch.complex64)  # |0⟩
        >>> phi = torch.tensor([0, 1], dtype=torch.complex64)  # |1⟩
        >>> quantum_fidelity(psi, phi)  # 0.0（正交态）
        >>> phi2 = torch.tensor([1, 0], dtype=torch.complex64)  # |0⟩
        >>> quantum_fidelity(psi, phi2)  # 1.0（相同态）
    """
    # 归一化（确保是合法的量子态）
    psi_norm = normalize_quantum_state(psi, dim=dim)
    phi_norm = normalize_quantum_state(phi, dim=dim)

    # 内积 ⟨ψ|φ⟩ = Σ conj(ψ_i) * φ_i
    inner = (psi_norm.conj() * phi_norm).sum(dim=dim)  # (...) 复数

    # 保真度 = |⟨ψ|φ⟩|²
    fidelity = inner.abs().pow(2)
    return fidelity.real  # 保真度为实数（虚部数值误差）


def trace_distance(psi: torch.Tensor, phi: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """计算两个纯量子态的迹距离 (Trace Distance)。

    迹距离是量子态区分能力的操作性度量：

        T(|ψ⟩, |φ⟩) = √(1 - F(|ψ⟩, |φ⟩)) = √(1 - |⟨ψ|φ⟩|²)

    与保真度的关系：
    - T = 0 ↔ F = 1（相同态）
    - T = 1 ↔ F = 0（正交态，完全可区分）

    迹距离满足三角不等式，是量子态空间上的合法度量（Bures 度量的简化形式）。

    在量子架构中的用途：
    1. **训练监控**：两层之间量子态的平均迹距离反映信息流动程度
    2. **正则化**：约束相邻层量子态不过度变化（防止梯度爆炸的语义）
    3. **早退判据**：迹距离 < ε 时可以安全跳过后续层

    Args:
        psi: 量子态 (..., d)
        phi: 量子态 (..., d)
        dim: 内积维度
    Returns:
        迹距离 (...)，实数，范围 [0, 1]
    """
    fid = quantum_fidelity(psi, phi, dim=dim)
    return torch.sqrt((1.0 - fid).clamp(min=0.0))


def quantum_mutual_information(
    psi: torch.Tensor,
    phi: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """计算两个量子态之间的量子互信息（Quantum Mutual Information）。

    量子互信息衡量两个量子子系统之间的总关联（包含经典关联和量子纠缠）：

        I(A:B) = S(A) + S(B) - S(AB)

    其中 S(X) 是子系统 X 的 von Neumann 熵。

    对于由 Born 概率表示的纯态近似：
        I(A:B) ≈ H(p_A) + H(p_B) - H(p_A ⊗ p_B)
                = H(p_A) + H(p_B) - H(p_A) - H(p_B)
                = 0  （纯态无经典混合关联）

    注意：对于可分态 ρ_AB = ρ_A ⊗ ρ_B，互信息为 0。
    当 I(A:B) > 0 时，表明两子系统之间存在关联（纠缠或经典关联）。

    本实现采用乘积分布近似：
        - 计算各态的 Born 概率分布 p(ψ), p(φ)
        - 乘积分布 p(ψ) ⊗ p(φ) 作为无关联参考
        - 互信息 = D_KL(联合分布 || 乘积分布) 的近似

    在量子架构中的用途：
    1. **QEL 监控**：纠缠前后的互信息变化，衡量纠缠门的"关联注入"效果
    2. **QSA 诊断**：不同 head 之间的互信息，检测 head 冗余度
    3. **损失函数**：最大化互信息可作为纠缠学习的辅助损失

    Args:
        psi: 量子态 A (..., d_A)
        phi: 量子态 B (..., d_B)
        dim: 内积维度
        eps: 防止 log(0) 的数值稳定 epsilon
    Returns:
        互信息估计值 (...)，实数，非负

    Example:
        >>> psi = torch.randn(4, 64, dtype=torch.complex64)
        >>> phi = torch.randn(4, 64, dtype=torch.complex64)
        >>> I = quantum_mutual_information(psi, phi)  # (...,) ≥ 0
    """
    # Born 概率
    p_A = born_normalize(psi, dim=dim, eps=eps)   # (..., d_A) 实数
    p_B = born_normalize(phi, dim=dim, eps=eps)   # (..., d_B) 实数

    # 子系统熵 S(A), S(B)
    S_A = von_neumann_entropy(p_A, dim=dim, eps=eps)  # (...,)
    S_B = von_neumann_entropy(p_B, dim=dim, eps=eps)  # (...,)

    # 联合分布近似：p_AB(i,j) = p_A(i) * p_B(j)（乘积分布 = 无关联参考）
    # S(AB) ≈ S(A) + S(B)（乘积态熵等于各子系统熵之和）
    # 因此互信息 I(A:B) = S(A) + S(B) - S(AB) ≈ 0（无额外关联）
    #
    # 更实用的做法：用 KL 散度衡量联合 Born 概率与乘积分布的偏差
    # 将两态在特征维度上进行外积，度量关联程度
    # (..., d_A, 1) × (..., 1, d_B) → (..., d_A, d_B)
    p_A_expanded = p_A.unsqueeze(-1)   # (..., d_A, 1)
    p_B_expanded = p_B.unsqueeze(-2)   # (..., 1, d_B)
    p_product = p_A_expanded * p_B_expanded  # (..., d_A, d_B) 独立分布

    # 联合 Born 概率：用外积计算（近似）
    # 通过模长乘积构造联合振幅，然后 Born 归一化
    psi_expanded = psi.unsqueeze(-1)   # (..., d_A, 1)
    phi_expanded = phi.unsqueeze(-2)   # (..., 1, d_B)
    joint_amplitude = psi_expanded * phi_expanded  # (..., d_A, d_B) 外积态

    # 联合 Born 概率（展平后归一化）
    joint_shape = joint_amplitude.shape
    joint_flat = joint_amplitude.reshape(*joint_shape[:-2], -1)  # (..., d_A*d_B)
    p_joint = born_normalize(joint_flat, dim=-1, eps=eps)  # (..., d_A*d_B)
    p_product_flat = p_product.reshape(*joint_shape[:-2], -1)  # (..., d_A*d_B)

    # 互信息 = KL(p_joint || p_product)
    log_ratio = torch.log(p_joint.clamp(min=eps)) - torch.log(p_product_flat.clamp(min=eps))
    mutual_info = (p_joint * log_ratio).sum(dim=-1)  # (...,)

    return mutual_info.clamp(min=0.0)  # 理论非负，数值上 clamp 防止浮点误差


def quantum_relative_entropy(
    psi: torch.Tensor,
    phi: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """计算量子相对熵（KL 散度的量子类比）。

    对于由 Born 法则导出的概率分布：
        p_i = |ψ_i|² / Σ_j |ψ_j|²
        q_i = |φ_i|² / Σ_j |φ_j|²

    量子相对熵（在 Born 概率分布意义下）：
        D(p||q) = Σ_i p_i · log(p_i / q_i)

    注意：严格量子相对熵 S(ρ||σ) = Tr(ρ(log ρ - log σ)) 需要密度矩阵，
    这里使用 Born 概率分布的经典 KL 散度作为近似，适用于纯态情形。

    在 QKD（量子知识蒸馏）和 QSA 对齐损失中可直接替代 KL 散度。

    Args:
        psi: 量子态（参考分布）(..., d)
        phi: 量子态（目标分布）(..., d)
        dim: 状态维度
        eps: 防止 log(0)
    Returns:
        相对熵 (...)，实数，非负
    """
    p = born_normalize(psi, dim=dim, eps=eps)  # Born 概率
    q = born_normalize(phi, dim=dim, eps=eps)

    # KL 散度 D(p||q)
    log_ratio = torch.log(p.clamp(min=eps)) - torch.log(q.clamp(min=eps))
    return (p * log_ratio).sum(dim=dim)
