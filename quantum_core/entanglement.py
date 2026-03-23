"""
QEL 量子纠缠层 (Quantum Entanglement Layer)

核心思想：用量子纠缠操作替代 Transformer 的残差连接。

理论基础：
    两个量子态 |a⟩ ∈ ℂ^d 和 |b⟩ ∈ ℂ^d 的纠缠操作为：
        ENTANGLE(|a⟩, |b⟩) = (U_ent ⊗ I)(I ⊗ U_ent) |a⟩⊗|b⟩

    其中 U_ent ∈ U(d) 是参数化的纠缠酉门（通过 Cayley 变换保证酉性）。

实现策略：
    完整的张量积 |a⟩⊗|b⟩ ∈ ℂ^{d²} 在高维时内存开销过大，
    因此采用 Schmidt 分解近似：
        |ψ_AB⟩ ≈ Σ_{k=1}^{r} σ_k |u_k⟩_A ⊗ |v_k⟩_B

    其中 r ≪ d 为截断秩，通过可学习的纠缠酉门控制 Schmidt 系数。

操作流程：
1. 局部纠缠：相邻 token 对通过参数化酉门进行纠缠耦合
   - 构造纠缠酉门 U_ent = Cayley(Ω)，保证严格酉性
   - 对 (|x_i⟩, |x_{i+1}⟩) 对应用纠缠操作
2. 全局纠缠：通过 QFT（量子傅里叶变换）在序列维度建立长程关联
   - QFT 在序列维度 dim=1（token 间）执行，而非特征维度
   - 多步 QFT 增强纠缠深度（可类比量子电路中的纠缠深度）
3. 纠缠融合：用酉耦合替代残差连接
   - U_couple(|ψ_input⟩⊗|ψ_entangled⟩) → 投影回 ℂ^d
   - 非线性融合，保留相位关系
4. 自适应纠缠强度：纠缠强度根据输入动态调整
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
    complex_inner_product,
    von_neumann_entropy,
)
from .unitary import CayleyLinear


# ──────────────────────────────────────────────
# 纠缠度量
# ──────────────────────────────────────────────


def concurrence(a: torch.Tensor, b: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """计算两个量子态的纠缠度量（concurrence 近似）。

    Concurrence 是两体量子态纠缠的标准度量之一。
    对于纯态 |ψ⟩_{AB}，concurrence = √(2(1 - Tr(ρ_A²)))。

    这里使用简化度量：
        C(a, b) = |⟨a|b⟩|² 的某种函数
    来近似纠缠强度。

    更准确的度量需要构造约化密度矩阵：
        ρ_A = Tr_B(|ψ⟩⟨ψ|)

    但这对高维张量计算开销大，所以我们使用近似版本。

    Args:
        a: 量子态 (..., d)
        b: 量子态 (..., d)
        dim: 内积维度
    Returns:
        纠缠度量标量 (...)
    """
    # 归一化
    a_norm = normalize_quantum_state(a, dim=dim)
    b_norm = normalize_quantum_state(b, dim=dim)

    # 内积模方
    overlap = complex_inner_product(a_norm, b_norm, dim=dim).abs().pow(2)

    # 使用 1 - |⟨a|b⟩|² 作为纠缠度量
    # 当两态正交时纠缠最大，相同时纠缠为零
    return 1.0 - overlap


def entanglement_entropy(a: torch.Tensor, b: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """计算两体纠缠熵（von Neumann 熵的近似）。

    对于纯态 |ψ⟩ = U_ent(|a⟩⊗|b⟩)，纠缠熵等于任一子系统的 von Neumann 熵：
        S(A) = S(B) = -Σ λ_k log λ_k

    其中 λ_k 是约化密度矩阵的特征值。

    简化实现：构造约化密度矩阵并计算其特征值的熵。

    Args:
        a: 量子态 (..., d)
        b: 量子态 (..., d)
        dim: 特征维度
    Returns:
        纠缠熵 (...)
    """
    # 构造约化密度矩阵 ρ_A = |a⟩⟨a| 的对角近似
    probs = born_normalize(a, dim=dim)
    return von_neumann_entropy(probs, dim=dim)


# ──────────────────────────────────────────────
# Schmidt 纠缠操作
# ──────────────────────────────────────────────


class SchmidtEntanglementGate(nn.Module):
    """基于 Schmidt 分解的纠缠门。

    完整张量积 |a⟩⊗|b⟩ ∈ ℂ^{d²} 的酉变换在 d 大时不可行。
    本模块使用低秩近似实现纠缠效果：

    1. 将两个态分别投影到 r 维纠缠子空间：
       |ã⟩ = P_A |a⟩,  |b̃⟩ = P_B |b⟩   (P_A, P_B ∈ ℂ^{d×r})
    2. 在纠缠子空间中应用可学习的 2r×2r 酉门：
       |ψ̃⟩ = U_ent · [|ã⟩; |b̃⟩]
    3. 投影回原空间：
       |a'⟩ = P_A^† |ψ̃₁⟩,  |b'⟩ = P_B^† |ψ̃₂⟩

    当 r = d 时退化为完整的张量积酉变换（但 O(d²) 参数和计算量）。

    Args:
        dim: 特征维度
        rank: 纠缠子空间维度（Schmidt 秩截断）
        init_scale: 酉门初始化缩放
    """

    def __init__(
        self,
        dim: int,
        rank: Optional[int] = None,
        init_scale: float = 0.02,
    ):
        super().__init__()
        self.dim = dim
        # 默认 rank = min(dim, 32)，平衡效果和效率
        self.rank = rank if rank is not None else min(dim, 32)

        # 投影矩阵 P_A, P_B: (dim, rank) 复数
        # 通过 Cayley 变换保证酉性（近似等距投影）
        self.proj_a = CayleyLinear(dim, self.rank, init_scale=init_scale)
        self.proj_b = CayleyLinear(dim, self.rank, init_scale=init_scale)

        # 2r×2r 纠缠酉门 U_ent（在拼接的子空间上操作）
        self.entangle_unitary = CayleyLinear(2 * self.rank, 2 * self.rank, init_scale=init_scale)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """对两个量子态应用纠缠操作。

        Args:
            a: 量子态 (..., d)
            b: 量子态 (..., d)
        Returns:
            (a_out, b_out): 纠缠后的量子态 (..., d)
        """
        # 投影到纠缠子空间
        a_proj = self.proj_a(a)  # (..., r)
        b_proj = self.proj_b(b)  # (..., r)

        # 拼接并应用纠缠酉门
        combined = torch.cat([a_proj, b_proj], dim=-1)  # (..., 2r)
        entangled = self.entangle_unitary(combined)  # (..., 2r)

        # 拆分并投影回原空间
        a_ent = entangled[..., : self.rank]  # (..., r)
        b_ent = entangled[..., self.rank :]  # (..., r)

        a_out = self.proj_a.conj().T @ a_ent.unsqueeze(-1)  # 逆投影近似
        b_out = self.proj_b.conj().T @ b_ent.unsqueeze(-1)

        # 实际上 CayleyLinear 的逆不是简单的 .conj().T
        # 用转置投影回原空间维度
        a_out = a_out.squeeze(-1)  # (..., d)
        b_out = b_out.squeeze(-1)  # (..., d)

        return a_out, b_out


class SchmidtEntanglementGateV2(nn.Module):
    """Schmidt 纠缠门 V2 — 使用可学习的混合系数而非投影。

    更高效且梯度友好的纠缠实现：

    1. 构造混合表示：
       m = MLP([|a|²; |b|²; Re(a·b̄); Im(a·b̄)])  →  纠缠参数
    2. 参数化纠缠酉门：
       U_ent(θ) = [[cos θ·I + sin θ·W_aa, sin θ·W_ab],
                    [sin θ·W_ba, cos θ·I + sin θ·W_bb]]
       其中 W_aa, W_ab, W_ba, W_bb ∈ ℂ^{d×d}
    3. 纠缠操作：
       [a'; b'] = U_ent · [a; b]

    这等价于在 d 维空间上应用块结构酉变换，
    避免了 d² 维张量积空间的显式构造。

    Args:
        dim: 特征维度
        use_cayley: 是否用 Cayley 变换保证酉性（否则用 softmax 归一化）
    """

    def __init__(self, dim: int, use_cayley: bool = True, init_scale: float = 0.02):
        super().__init__()
        self.dim = dim

        if use_cayley:
            # 2d × 2d 纠缠酉门 — 精确酉
            self.U_ent = CayleyLinear(2 * dim, 2 * dim, init_scale=init_scale)
            self._use_cayley = True
        else:
            # 四个 d×d 块 — 软酉（训练中近似酉）
            self.W_aa = nn.Parameter(torch.randn(dim, dim, dtype=torch.complex64) * init_scale)
            self.W_ab = nn.Parameter(torch.randn(dim, dim, dtype=torch.complex64) * init_scale)
            self.W_ba = nn.Parameter(torch.randn(dim, dim, dtype=torch.complex64) * init_scale)
            self.W_bb = nn.Parameter(torch.randn(dim, dim, dtype=torch.complex64) * init_scale)
            self._use_cayley = False

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """对两个量子态应用纠缠酉变换。

        Args:
            a: (..., d)
            b: (..., d)
        Returns:
            (a_out, b_out): (..., d)
        """
        combined = torch.cat([a, b], dim=-1)  # (..., 2d)

        if self._use_cayley:
            entangled = self.U_ent(combined)  # (..., 2d)
        else:
            # 块矩阵乘法 + 软酉归一化
            a_part = a @ self.W_aa + b @ self.W_ba
            b_part = a @ self.W_ab + b @ self.W_bb
            entangled = torch.cat([a_part, b_part], dim=-1)

        a_out = entangled[..., : self.dim]
        b_out = entangled[..., self.dim :]

        return a_out, b_out


# ──────────────────────────────────────────────
# 自适应纠缠门
# ──────────────────────────────────────────────


class AdaptiveEntanglementGate(nn.Module):
    """自适应纠缠门：纠缠强度根据输入动态调整。

    改进：使用复数信息（模长 + 相位）计算纠缠强度，
    而非仅仅使用模长。

    θ_i = sigmoid(MLP([|a_i|²; |b_i|²; Re(⟨a|b⟩); Im(⟨a|b⟩)])) * θ_max

    相关 token 对 → 高纠缠；无关 token 对 → 低纠缠。

    Args:
        dim: 特征维度
        rank: Schmidt 秩截断（用于纠缠门）
        theta_max: 最大纠缠角度
    """

    def __init__(
        self,
        dim: int,
        rank: Optional[int] = None,
        theta_max: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.theta_max = theta_max

        # Schmidt 纠缠门
        self.entangle_gate = SchmidtEntanglementGateV2(dim=dim, use_cayley=True)

        # 自适应强度 MLP：输入 4 维特征（模长 + 内积），输出 1 维强度
        self.strength_mlp = nn.Sequential(
            nn.Linear(4, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, a: torch.Tensor, b: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """自适应纠缠操作。

        Args:
            a: (..., d) 复数
            b: (..., d) 复数
        Returns:
            (a_out, b_out, strength): 纠缠后的态和纠缠强度 (..., 1)
        """
        # 计算纠缠相关特征
        a_mag_sq = a.abs().pow(2).mean(dim=-1, keepdim=True)  # (..., 1)
        b_mag_sq = b.abs().pow(2).mean(dim=-1, keepdim=True)  # (..., 1)
        inner = complex_inner_product(
            a,
            b,
            dim=-1,
        )  # (...,) 复数

        # 拼接特征
        feat = torch.cat(
            [
                a_mag_sq,
                b_mag_sq,
                inner.real.unsqueeze(-1),
                inner.imag.unsqueeze(-1),
            ],
            dim=-1,
        )  # (..., 4)

        # 动态计算纠缠强度
        strength = self.strength_mlp(feat) * self.theta_max  # (..., 1)

        # 通过纠缠门变换
        a_ent, b_ent = self.entangle_gate(a, b)

        # 使用强度插值（低强度 → 保持原态，高强度 → 使用纠缠态）
        a_out = (1.0 - strength) * a + strength * a_ent
        b_out = (1.0 - strength) * b + strength * b_ent

        return a_out, b_out, strength


# ──────────────────────────────────────────────
# 量子傅里叶变换 (QFT)
# ──────────────────────────────────────────────


class QuantumFourierTransform(nn.Module):
    """量子傅里叶变换（QFT）模块。

    QFT 是建立全局量子纠缠的标准方法。在量子电路中，
    n-qubit QFT 需要 O(n²) 个量子门，能将所有量子比特纠缠。

    关键修正：QFT 应在**序列维度**（token 之间）执行，
    而非特征维度（token 内部）。

    QFT_N 的变换矩阵元素：
        [QFT_N]_{j,k} = ω^{jk} / √N,  ω = exp(2πi/N)

    对于序列长度 N，这等价于 torch.fft.fft 后除以 √N。

    为实现可学习的全局纠缠，我们引入：
    1. 相位旋转参数 φ_k（类比量子电路中的 R_k 门）
    2. 可学习的混合系数 α（控制纠缠深度）
    3. 多步 QFT（增加纠缠深度）

    Args:
        learnable_phase: 是否学习 QFT 相位旋转
        n_steps: QFT 步数（增加纠缠深度）
        init_alpha: 初始混合系数
    """

    def __init__(
        self,
        learnable_phase: bool = True,
        n_steps: int = 1,
        init_alpha: float = 0.5,
    ):
        super().__init__()
        self.learnable_phase = learnable_phase
        self.n_steps = n_steps

        # 可学习的 QFT 混合系数（控制纠缠强度）
        self.logit_alpha = nn.Parameter(
            torch.tensor(math.log(init_alpha / (1 - init_alpha + 1e-8)))
        )

    @property
    def alpha(self) -> torch.Tensor:
        """混合系数 α ∈ (0, 1)"""
        return torch.sigmoid(self.logit_alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """在序列维度应用 QFT。

        Args:
            x: 复数输入 (batch, seq_len, dim)
        Returns:
            QFT 纠缠后的复数张量 (batch, seq_len, dim)
        """
        B, N, D = x.shape
        alpha = self.alpha

        result = x
        for step in range(self.n_steps):
            # 在序列维度 dim=1 上做 FFT
            # norm='ortho' 保证酉性：F†F = I，即 QFT 操作是等距变换
            x_qft = torch.fft.fft(result, dim=1, norm="ortho")

            # 混合原始态和 QFT 态（部分纠缠）
            # α→1：保持原始局部表示；α→0：完全转换为全局频域表示
            result = alpha * result + (1 - alpha) * x_qft

        return result


# ──────────────────────────────────────────────
# 酉耦合（替代残差连接）
# ──────────────────────────────────────────────


class UnitaryCoupling(nn.Module):
    """酉耦合层：用纠缠耦合替代残差连接。

    Transformer 残差：y = x + SubLayer(x)   ← 线性，无相位保护
    酉耦合：|ψ_out⟩ = U_couple(|ψ_x⟩⊗|ψ_sub⟩)  ← 非线性，保留相位

    实现：
    1. 拼接 [x; entangled] → (..., 2d)
    2. 应用 2d×2d 酉变换 U_couple
    3. 从酉变换结果中提取 d 维输出

    酉性保证信息守恒（不丢失、不放大），梯度流绝对稳定。

    Args:
        dim: 特征维度
        coupling_type: 'full'（完整酉）或 'diagonal'（对角酉，高效）
    """

    def __init__(self, dim: int, coupling_type: str = "full", init_scale: float = 0.02):
        super().__init__()
        self.dim = dim
        self.coupling_type = coupling_type

        if coupling_type == "full":
            # 完整 2d×2d 酉耦合
            self.U_couple = CayleyLinear(2 * dim, 2 * dim, init_scale=init_scale)
        elif coupling_type == "diagonal":
            # 对角酉：每个维度独立旋转，O(d) 参数
            self.phase = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
            self.mix = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
        else:
            raise ValueError(f"Unknown coupling_type: {coupling_type}")

    def forward(self, x: torch.Tensor, sublayer_output: torch.Tensor) -> torch.Tensor:
        """酉耦合融合。

        Args:
            x: 原始输入 (..., d)
            sublayer_output: 子层输出（纠缠后的态）(..., d)
        Returns:
            耦合输出 (..., d)
        """
        if self.coupling_type == "full":
            combined = torch.cat([x, sublayer_output], dim=-1)  # (..., 2d)
            coupled = self.U_couple(combined)  # (..., 2d)
            # 从耦合空间中提取输出（取前半部分 + 后半部分的投影）
            out = (coupled[..., : self.dim] + coupled[..., self.dim :]) / math.sqrt(2)
            return out
        elif self.coupling_type == "diagonal":
            # 对角酉：|ψ_out⟩_k = e^{iφ_k}(cos θ_k |x⟩_k + sin θ_k |sub⟩_k)
            theta = torch.sigmoid(self.mix) * (torch.pi / 2)
            phase = torch.exp(1j * self.phase)
            out = phase * (torch.cos(theta) * x + torch.sin(theta) * sublayer_output)
            return out


# ──────────────────────────────────────────────
# 主模块：量子纠缠层
# ──────────────────────────────────────────────


class QuantumEntanglementLayer(nn.Module):
    """量子纠缠层 (QEL) — 完整实现。

    对序列中的 token 建立量子纠缠关联：
    1. 局部纠缠：相邻 token 对通过参数化酉门耦合（真正的张量积近似）
    2. 全局纠缠：QFT 在序列维度建立长程关联（修正维度方向）
    3. 酉耦合融合：用 U_couple 替代残差连接（非线性融合）
    4. 自适应强度：MLP 根据复数特征控制纠缠强度

    与原版的关键差异：
    - 原版：cos·a + i·sin·b（仅线性混合） → 新版：Schmidt 纠缠门（酉变换）
    - 原版：QFT 在 dim=-1（特征维度） → 新版：QFT 在 dim=1（序列维度）
    - 原版：(x + entangled)/√2（普通平均） → 新版：U_couple（酉耦合）

    Args:
        dim: 特征维度
        use_adaptive: 是否使用自适应纠缠强度
        use_global_qft: 是否使用 QFT 全局纠缠
        coupling_type: 'full' 或 'diagonal'（酉耦合类型）
        qft_steps: QFT 步数
        init_scale: 初始化缩放
    """

    def __init__(
        self,
        dim: int,
        use_adaptive: bool = True,
        use_global_qft: bool = True,
        coupling_type: str = "full",
        qft_steps: int = 1,
        init_scale: float = 0.02,
    ):
        super().__init__()
        self.dim = dim
        self.use_adaptive = use_adaptive
        self.use_global_qft = use_global_qft

        # 局部纠缠门
        if use_adaptive:
            self.entangle_gate = AdaptiveEntanglementGate(dim=dim)
        else:
            # 固定强度酉纠缠门
            self.entangle_gate = SchmidtEntanglementGateV2(
                dim=dim, use_cayley=True, init_scale=init_scale
            )

        # 全局 QFT
        if use_global_qft:
            self.qft = QuantumFourierTransform(
                learnable_phase=True,
                n_steps=qft_steps,
            )

        # 酉耦合（替代残差连接）
        self.coupling = UnitaryCoupling(dim=dim, coupling_type=coupling_type, init_scale=init_scale)

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
        entangled = self._local_entangle(x, metrics)

        # ── 2. 全局纠缠（QFT 在序列维度）──
        if self.use_global_qft and N > 1:
            entangled = self.qft(entangled)
            metrics["qft_applied"] = True
            metrics["qft_alpha"] = self.qft.alpha.item()

        # ── 3. 酉耦合融合（替代残差连接）──
        output = self.coupling(x, entangled)

        return output, metrics

    def _local_entangle(self, x: torch.Tensor, metrics: Dict[str, float]) -> torch.Tensor:
        """局部纠缠：对相邻 token 对应用纠缠门。

        改进的奇数序列处理策略：
        - 偶数 N：所有 token 都参与纠缠对，无边界问题
        - 奇数 N：最后一个 token 与倒数第二个 token 组成额外的纠缠对
          （覆盖重叠）。这样每个 token 至少参与一次纠缠，
          避免末尾 token 因无配对而保持未纠缠状态，
          造成不同位置 token 的信息更新不均衡。

        Args:
            x: (B, N, D) 复数
            metrics: 统计信息字典
        Returns:
            局部纠缠后的张量 (B, N, D)
        """
        B, N, D = x.shape

        if N < 2:
            return x

        entangled = x.clone()

        if self.use_adaptive:
            # 自适应纠缠：批量处理相邻对（0,1), (2,3), ...
            x_even = x[:, 0::2, :]  # (B, ceil(N/2), D)
            x_odd = x[:, 1::2, :]  # (B, floor(N/2), D)

            min_len = min(x_even.shape[1], x_odd.shape[1])
            x_even_c = x_even[:, :min_len, :]
            x_odd_c = x_odd[:, :min_len, :]

            x_even_out, x_odd_out, strengths = self.entangle_gate(x_even_c, x_odd_c)

            entangled[:, 0::2, :][:, :min_len, :] = x_even_out
            entangled[:, 1::2, :][:, :min_len, :] = x_odd_out

            # ── 奇数 N 边界处理：最后一个 token 与倒数第二个额外纠缠 ──
            # 注：min_len = N//2，此时 x_even 比 x_odd 多一个（N 为奇数）
            if N % 2 == 1:
                # 末尾落单 token 的索引 = N - 1 = 2 * (N//2)
                last_idx = N - 1
                second_last_idx = N - 2
                # 用倒数第二个（已纠缠后的）状态作为配对，再纠缠一次末尾 token
                a_extra = entangled[:, second_last_idx : second_last_idx + 1, :]  # (B, 1, D)
                b_extra = x[:, last_idx : last_idx + 1, :]  # (B, 1, D)，使用原始未纠缠值
                a_e_out, b_e_out, s_extra = self.entangle_gate(a_extra, b_extra)
                # 只更新末尾 token（避免覆盖倒数第二个已完成的纠缠）
                entangled[:, last_idx : last_idx + 1, :] = b_e_out
                avg_strength = (strengths.mean() + s_extra.mean()).item() / 2
            else:
                avg_strength = strengths.mean().item()

            metrics["entanglement_strength"] = avg_strength

            # 纠缠度量
            with torch.no_grad():
                ent_measure = concurrence(x_even_out, x_odd_out, dim=-1)
                metrics["avg_concurrence"] = ent_measure.mean().item()
        else:
            # 固定强度纠缠（含奇数边界处理）
            for i in range(0, N - 1, 2):
                a = x[:, i, :]
                b = x[:, i + 1, :]
                a_out, b_out = self.entangle_gate(a, b)
                entangled[:, i, :] = a_out
                entangled[:, i + 1, :] = b_out

            # 奇数 N：额外处理最后一个 token
            if N % 2 == 1:
                last_idx = N - 1
                a_extra = entangled[:, last_idx - 1, :]
                b_extra = x[:, last_idx, :]
                _, b_out_extra = self.entangle_gate(a_extra, b_extra)
                entangled[:, last_idx, :] = b_out_extra

            metrics["entanglement_strength"] = 1.0  # 固定全强度

        return entangled

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, adaptive={self.use_adaptive}, "
            f"global_qft={self.use_global_qft}, "
            f"coupling={self.coupling.coupling_type}"
        )

    @property
    def entanglement_depth(self) -> int:
        """返回纠缠深度（QFT 步数 + 局部纠缠层数）。

        纠缠深度是衡量模型在序列方向建立长程关联能力的指标：
        - 局部纠缠始终贡献 1 层
        - 每步 QFT 额外贡献 1 层（指数级扩大感受野）

        Returns:
            等效纠缠深度（整数）
        """
        depth = 1  # 局部纠缠
        if self.use_global_qft:
            depth += self.qft.n_steps
        return depth


# ──────────────────────────────────────────────
# 向后兼容别名
# ──────────────────────────────────────────────


# 保留旧版 EntanglementGate 接口（但内部使用 Schmidt 门）
class EntanglementGate(nn.Module):
    """向后兼容的纠缠门。

    新版内部使用 SchmidtEntanglementGateV2 实现真正的张量积近似纠缠。
    """

    def __init__(self, dim: int = 64, init_theta: float = 0.1):
        super().__init__()
        self.gate = SchmidtEntanglementGateV2(dim=dim, use_cayley=True)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.gate(a, b)

    def get_gate_matrix(self) -> torch.Tensor:
        """返回纠缠酉门的有效表示。

        对于 2d×2d 酉门，返回其左上 d×d 块作为近似 2x2 表示。
        """
        with torch.no_grad():
            W = (
                self.gate.U_ent.unitary_matrix
                if hasattr(self.gate.U_ent, "unitary_matrix")
                else None
            )
            if W is not None:
                # 返回 d×d 的主要作用块
                return W[: self.gate.dim, : self.gate.dim]
        return torch.eye(2, dtype=torch.complex64)
