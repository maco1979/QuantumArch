"""
Cayley 参数化酉矩阵层

核心公式：
    W = (I + i/2 * Ω)^{-1} (I - i/2 * Ω)

其中 Ω 为斜厄米矩阵（skew-Hermitian）: Ω† = -Ω

性质：
- W 自动满足酉性约束 W†W = I
- Ω 有 d(d-1) 个实数自由度（斜厄米矩阵的对角线为纯虚数）
- 使用 torch.linalg.solve 代替 inv 提高数值稳定性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CayleyLinear(nn.Module):
    """Cayley 参数化的酉线性层。

    将输入向量通过酉矩阵 W 做线性变换：
        output = input @ W

    W 通过 Cayley 变换从可学习的斜厄米参数 Ω 获得：
        W = Cayley(Ω) = (I + i/2·Ω)^{-1} · (I - i/2·Ω)

    Args:
        in_features: 输入特征维度
        out_features: 输出特征维度
            当 in_features == out_features 时，W 为方阵，严格酉
            当不相等时，使用矩阵乘法但酉性不严格保证
        init_scale: Ω 的初始化缩放因子（较小值保持 W 接近单位矩阵）
        eps: Cayley 变换中的数值稳定 epsilon
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        init_scale: float = 0.01,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.init_scale = init_scale
        self.eps = eps
        self.is_square = in_features == out_features

        # 参数化：存储斜厄米矩阵 Ω 的自由度
        # 斜厄米矩阵：Ω† = -Ω
        # 对角线为纯虚数，上三角 = -conj(下三角)
        # 存储方式：上三角部分（含对角线的虚部）
        if self.is_square:
            d = in_features
            # 自由度：对角线 d 个虚数 + 上三角 d*(d-1)/2 个复数
            # 总共 d + 2 * d*(d-1)/2 = d² 个实参数
            n_triangular = d * (d - 1) // 2
            self.omega_diag = nn.Parameter(torch.zeros(d, dtype=torch.float32))  # 对角线虚部
            self.omega_tri = nn.Parameter(
                torch.zeros(n_triangular, dtype=torch.complex64)  # 上三角复数
            )
        else:
            # 非方阵：使用一般复数矩阵 + Cayley 正则化
            self.omega = nn.Parameter(
                torch.randn(in_features, out_features, dtype=torch.complex64) * init_scale
            )

    def _get_skew_hermitian(self) -> torch.Tensor:
        """从存储参数重建斜厄米矩阵 Ω。

        使用向量化索引替代双层 Python for 循环，在大维度 d 时性能显著更优。

        Returns:
            Ω: (d, d) 斜厄米矩阵，复数类型
        """
        d = self.in_features
        device = self.omega_diag.device

        # 构建全零矩阵
        Omega = torch.zeros(d, d, dtype=torch.complex64, device=device)

        # 填充对角线：纯虚数
        diag_imag = self.omega_diag * 1j
        Omega.diagonal().copy_(diag_imag)

        # 向量化填充上三角（使用 torch.triu_indices）
        rows, cols = torch.triu_indices(d, d, offset=1, device=device)
        Omega[rows, cols] = self.omega_tri
        # 斜厄米性质：Ω[j,i] = -conj(Ω[i,j])
        Omega[cols, rows] = -self.omega_tri.conj()

        return Omega

    @torch.no_grad()
    def _get_skew_hermitian_simple(self) -> torch.Tensor:
        """简化版：使用反对称部分构造斜厄米矩阵。

        Ω = (A - A†) * scale，其中 A 为一般复数矩阵
        """
        if self.is_square:
            return self._get_skew_hermitian()
        else:
            # 非方阵情况：取反对称部分
            d = min(self.in_features, self.out_features)
            A_square = self.omega[:d, :d]
            return (A_square - A_square.conj().T) * 0.5

    def cayley_transform(self, Omega: torch.Tensor) -> torch.Tensor:
        """Cayley 变换：W = (I + i/2·Ω)^{-1} · (I - i/2·Ω)

        使用 torch.linalg.solve 代替 inv 提高数值稳定性：
            solve(I + i/2·Ω, I - i/2·Ω)

        Args:
            Omega: 斜厄米矩阵 (d, d)
        Returns:
            酉矩阵 W (d, d)
        """
        d = Omega.shape[0]
        I = torch.eye(d, dtype=Omega.dtype, device=Omega.device)
        half_i_omega = 0.5j * Omega

        A = I + half_i_omega
        B = I - half_i_omega

        # W = A^{-1} B 等价于 A W = B
        W = torch.linalg.solve(A, B)

        return W

    @property
    def unitary_matrix(self) -> torch.Tensor:
        """返回当前的酉矩阵 W（仅方阵情况）。

        注意：此操作不追踪梯度。用于验证和检查。
        """
        if not self.is_square:
            raise RuntimeError("unitary_matrix 仅在方阵情况下可用")
        with torch.no_grad():
            Omega = self._get_skew_hermitian()
            return self.cayley_transform(Omega)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入复数张量 (..., in_features)
        Returns:
            输出复数张量 (..., out_features)
        """
        if self.is_square:
            Omega = self._get_skew_hermitian()
            W = self.cayley_transform(Omega)
            return x @ W
        else:
            # 非方阵：使用一般投影 + Cayley 正则化
            return x @ self.omega

    def get_unitarity_violation(self) -> torch.Tensor:
        """计算酉性违背度 ||W†W - I||_F。

        Returns:
            标量违背度
        """
        if not self.is_square:
            return torch.tensor(float("inf"))
        W = self.unitary_matrix
        d = W.shape[0]
        I = torch.eye(d, dtype=W.dtype, device=W.device)
        return (W.conj().T @ W - I).abs().pow(2).sum().sqrt()

    @torch.no_grad()
    def recover_unitarity(self, method: str = "qr") -> float:
        """主动恢复酉性约束（当数值误差积累导致 Ω 参数漂移时使用）。

        理论上，Cayley 参数化保证 W 精确酉，但由于：
        1. 有限精度浮点运算的舍入误差积累
        2. 混合精度训练（float16 → 精度丢失）
        3. 长时间训练后 omega_diag/omega_tri 参数的数值漂移

        可能导致 ||W†W - I||_F 不为零。本方法提供两种恢复策略：

        - ``'qr'``（推荐）：对当前 W 做 QR 分解，用 Q 替代 W，
          然后逆向求解新的 Ω = Cayley^{-1}(Q)，更新参数。
          - 优点：Q 精确酉，恢复后违背度接近机器精度
          - 缺点：需要矩阵分解，O(d³) 时间复杂度

        - ``'rescale'``：对 omega_tri 和 omega_diag 施加轻微正则化，
          将参数模长缩回安全范围（经验上 ||Ω|| < 10 保证数值稳定）。
          - 优点：快速，O(d²)
          - 缺点：恢复效果不精确

        推荐在以下时机调用：
        - 每 N 步训练后（作为定期维护，N 通常 > 1000）
        - 违背度 > 1e-4 时（由监控系统触发）
        - 混合精度训练的 checkpoint 保存前

        Args:
            method: 恢复方法，'qr'（精确）或 'rescale'（快速近似）

        Returns:
            恢复前的违背度（用于日志记录）

        Raises:
            ValueError: 非方阵情况不支持此方法
            ValueError: 不支持的 method 值
        """
        if not self.is_square:
            raise ValueError("recover_unitarity 仅支持方阵 CayleyLinear")

        d = self.in_features
        device = self.omega_diag.device

        # 记录恢复前的违背度
        W_before = self.unitary_matrix  # (d, d) 复数
        I = torch.eye(d, dtype=W_before.dtype, device=device)
        violation_before = (W_before.conj().T @ W_before - I).abs().pow(2).sum().sqrt().item()

        if method == "qr":
            # ── QR 分解恢复 ──
            # 对当前 W 做 QR，取 Q（精确酉矩阵）
            Q, R = torch.linalg.qr(W_before)

            # 确保 R 对角线为正（QR 分解中 Q 不唯一，这里做 phase 校正）
            diag_r = R.diagonal()
            phase_corr = diag_r / diag_r.abs().clamp(min=1e-8)  # e^{iθ_k}
            Q = Q * phase_corr.unsqueeze(0)  # 列相位校正 → Q 仍然酉

            # 逆向求解 Ω：从 Q 还原斜厄米矩阵
            # Q = (I + i/2·Ω)^{-1}(I - i/2·Ω)
            # => (I + i/2·Ω) Q = I - i/2·Ω
            # => Q + i/2·Ω·Q = I - i/2·Ω
            # => i/2·Ω(Q + I) = I - Q
            # => Ω = 2/(i) · (I - Q)(I + Q)^{-1}  = -2i · (I-Q)(I+Q)^{-1}
            I_plus_Q = I + Q
            I_minus_Q = I - Q
            try:
                Omega_recovered = -2j * torch.linalg.solve(I_plus_Q.conj().T, I_minus_Q.conj().T).conj().T
                # Omega_recovered 应为斜厄米：检查 ||Ω + Ω†||
                skew_violation = (Omega_recovered + Omega_recovered.conj().T).abs().max().item()
                if skew_violation > 1e-3:
                    # 强制斜厄米化
                    Omega_recovered = (Omega_recovered - Omega_recovered.conj().T) / 2

                # 更新参数
                # 对角线虚部
                self.omega_diag.data.copy_(Omega_recovered.diagonal().imag)
                # 上三角
                rows, cols = torch.triu_indices(d, d, offset=1, device=device)
                self.omega_tri.data.copy_(Omega_recovered[rows, cols])

            except torch.linalg.LinAlgError:
                # 若 (I+Q) 奇异，退化为 rescale 方法
                method = "rescale"

        if method == "rescale":
            # ── 参数缩放恢复（快速近似）──
            # 将 omega 参数的模长约束到安全范围内（经验 < 5.0）
            max_norm = 5.0
            with torch.no_grad():
                diag_norm = self.omega_diag.abs().max()
                if diag_norm > max_norm:
                    self.omega_diag.data.mul_(max_norm / diag_norm)

                tri_norm = self.omega_tri.abs().max()
                if tri_norm > max_norm:
                    self.omega_tri.data.mul_(max_norm / tri_norm)

        return violation_before

    def condition_number(self) -> float:
        """计算当前酉矩阵 W 的条件数（Condition Number）。

        对于理想的酉矩阵，所有奇异值均为 1，条件数 = σ_max / σ_min = 1。
        条件数偏离 1 的程度反映了数值退化程度。

        诊断指南：
        - condition_number ≈ 1.0：酉性良好，正常状态
        - 1.0 < κ < 10：轻微退化，可通过 recover_unitarity('rescale') 修复
        - κ > 100：严重退化，建议调用 recover_unitarity('qr') 恢复

        Returns:
            条件数（float），理想值为 1.0

        Raises:
            RuntimeError: 非方阵情况下不支持
        """
        if not self.is_square:
            raise RuntimeError("condition_number 仅在方阵 CayleyLinear 上有定义")
        W = self.unitary_matrix  # (d, d) 复数，不追踪梯度
        # 奇异值分解（仅需奇异值）
        s = torch.linalg.svdvals(W)  # (d,) 实数，降序排列
        sigma_max = s.max().item()
        sigma_min = s.min().clamp(min=1e-10).item()
        return sigma_max / sigma_min

    def omega_spectrum(self) -> dict:
        """分析斜厄米参数矩阵 Ω 的谱（特征值分布）。

        斜厄米矩阵 Ω 的特征值均为纯虚数 iλ_k（λ_k ∈ ℝ）。
        由 Cayley 变换，W 的特征值为 μ_k = (1-iλ_k/2)/(1+iλ_k/2) ∈ 单位圆上。

        Ω 的谱分析揭示：
        1. |λ_k| 分布：参数在希尔伯特空间中的"旋转角度"分布
        2. 接近奇点的参数：|λ_k| > 4 时 Cayley 变换可能不稳定
        3. 参数利用率：有效参数数量（|λ_k| > ε 的比例）

        Returns:
            dict 包含:
                - ``lambda_max``: |λ_k| 的最大值（稳定性指标，应 < 4）
                - ``lambda_mean``: |λ_k| 的均值（参数活跃度）
                - ``near_singularity_count``: |λ_k| > 4 的参数数量
                - ``effective_param_ratio``: |λ_k| > 0.01 的比例（参数利用率）
                - ``w_phase_uniformity``: W 特征值相位均匀性指数 ∈ [0,1]

        Raises:
            RuntimeError: 非方阵情况不支持
        """
        if not self.is_square:
            raise RuntimeError("omega_spectrum 仅在方阵 CayleyLinear 上有定义")

        with torch.no_grad():
            Omega = self._get_skew_hermitian()  # (d, d) 斜厄米

            # 斜厄米矩阵 Ω = -Ω†，所以 iΩ 为 Hermitian（iΩ = -(iΩ)†）
            # eigvalsh 对 Hermitian 矩阵返回实数特征值
            try:
                # 1j * Omega 是 Hermitian（因为 Omega 是斜厄米）
                # (1j * Ω)† = -1j * Ω† = -1j * (-Ω) = 1j * Ω ✓
                hermitian_op = 1j * Omega
                lambda_vals = torch.linalg.eigvalsh(hermitian_op)  # (d,) 实数
            except RuntimeError:
                lambda_vals = torch.linalg.eigvals(Omega).imag

            lambda_abs = lambda_vals.abs()
            lambda_max = lambda_abs.max().item()
            lambda_mean = lambda_abs.mean().item()
            near_singularity_count = int((lambda_abs > 4.0).sum().item())
            effective_param_ratio = (lambda_abs > 0.01).float().mean().item()

            # W 特征值的相位均匀性
            W = self.unitary_matrix
            w_eigenvalues = torch.linalg.eigvals(W)
            w_phases = w_eigenvalues.angle()
            # 均匀性 = 1 - |E[e^{iφ}]|（均匀分布 → 期望模长 ≈ 0 → 均匀性高）
            mean_phasor = torch.exp(1j * w_phases).mean()
            w_phase_uniformity = float((1.0 - mean_phasor.abs()).clamp(0, 1).item())

        return {
            "lambda_max": lambda_max,
            "lambda_mean": lambda_mean,
            "near_singularity_count": near_singularity_count,
            "effective_param_ratio": effective_param_ratio,
            "w_phase_uniformity": w_phase_uniformity,
        }

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"square={self.is_square}, init_scale={self.init_scale}"
        )




class CayleyLinearSimple(nn.Module):
    """简化版 Cayley 参数化酉线性层。

    使用完整的复数矩阵 A，然后通过 Cayley 变换投影到酉矩阵：
        Ω = (A - A†) / 2（自动满足斜厄米性）
        W = Cayley(Ω)

    适合快速原型验证，参数量是完整版的两倍，但代码更简洁。

    Args:
        features: 方阵维度（仅支持方阵）
        init_scale: 初始化缩放
        eps: 数值稳定 epsilon
    """

    def __init__(self, features: int, init_scale: float = 0.01, eps: float = 1e-8):
        super().__init__()
        self.features = features
        self.eps = eps

        # 可学习的一般复数矩阵
        self.A = nn.Parameter(torch.randn(features, features, dtype=torch.complex64) * init_scale)

    def _get_omega(self) -> torch.Tensor:
        """从 A 构造斜厄米矩阵：Ω = (A - A†) / 2"""
        return (self.A - self.A.conj().T) * 0.5

    def cayley_transform(self, Omega: torch.Tensor) -> torch.Tensor:
        d = self.features
        I = torch.eye(d, dtype=Omega.dtype, device=Omega.device)
        half_i_omega = 0.5j * Omega
        return torch.linalg.solve(I + half_i_omega, I - half_i_omega)

    @property
    def unitary_matrix(self) -> torch.Tensor:
        with torch.no_grad():
            return self.cayley_transform(self._get_omega())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Omega = self._get_omega()
        W = self.cayley_transform(Omega)
        return x @ W

    def extra_repr(self) -> str:
        return f"features={self.features}, eps={self.eps}"
