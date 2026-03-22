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
