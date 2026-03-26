"""
FFN_Q 量子前馈网络 (Quantum Feed-Forward Network)

核心设计（修正版）：
    FFN_Q(|ψ⟩) = U_out · σ_C(W_up |ψ⟩ + |b⟩) + |ψ⟩

其中：
- W_up: 上投影复数权重（非方阵复数投影，适当初始化）
- σ_C: 复数激活函数（ModReLU — 相位保持，模长调节）
- U_out: 输出投影（方阵时 Cayley 酉投影，非方阵时带正则化约束的复数投影）

门控变体 (Gated-FFN_Q)：
    Gate_Q(|ψ⟩) = |ψ⟩ ⊙ σ_C(W_gate |ψ⟩)
    FFN_Q_gated(|ψ⟩) = Gate_Q(σ_C(W_up |ψ⟩ + |b⟩)) · W_down

相比 Transformer FFN：
- 上/下投影使用复数权重，保持复数量子态信息
- 方阵下投影使用 Cayley 参数化，严格保证酉性
- 门控机制使用复数 sigmoid（实部+虚部分别门控）
- ModReLU 保持相位信息，不丢失复数编码
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict

from .activations import ModReLU
from .unitary import CayleyLinear
from .complex_ops import complex_dropout, check_unitarity


class ComplexBias(nn.Module):
    """可学习的复数偏置。"""

    def __init__(self, features: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(features, dtype=torch.complex64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.bias


class ComplexSigmoid(nn.Module):
    """复数 Sigmoid：分别对实部和虚部施加 Sigmoid。

    σ_C(z) = σ(Re(z)) + i · σ(Im(z))

    输出实部和虚部均在 (0, 1) 范围内，适合作为门控信号。

    与实数 Sigmoid 的区别：
    - 实数 Sigmoid: 输出 ∈ (0, 1) ⊂ ℝ
    - 复数 Sigmoid: 输出 ∈ (0,1) + i(0,1) ⊂ ℂ
    - 映射复平面第一象限内
    """

    def __init__(self):
        super().__init__()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.complex(torch.sigmoid(z.real), torch.sigmoid(z.imag))

    def extra_repr(self) -> str:
        return "ComplexSigmoid(σ_C)"


class ComplexLinear(nn.Module):
    """复数线性层（非酉投影）。

    用于 FFN 中需要维度变换的场景（如 dim → 4*dim）。
    相比 nn.Linear，提供更好的复数初始化。

    当 in_features == out_features 时，可选择使用 Cayley 酉投影。

    Args:
        in_features: 输入维度
        out_features: 输出维度
        use_cayley: 是否使用 Cayley 酉投影（仅方阵时可用）
        init_std: 初始化标准差（使用复数 Kaiming 初始化）
        bias: 是否使用偏置
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_cayley: bool = False,
        init_std: Optional[float] = None,
        bias: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.is_square = in_features == out_features

        if use_cayley and self.is_square:
            # 方阵：使用 Cayley 酉投影
            self.linear = CayleyLinear(in_features, out_features)
            self._is_cayley = True
        else:
            # 非方阵或非 Cayley：使用复数线性投影
            self.linear = nn.Linear(in_features, out_features, bias=bias).to(torch.complex64)
            self._is_cayley = False

            # 复数 Kaiming 初始化
            if init_std is None:
                # 类似 nn.Linear 的默认初始化，但考虑复数方差
                init_std = 1.0 / math.sqrt(in_features)
            nn.init.normal_(self.linear.weight.data.real, std=init_std / math.sqrt(2))
            nn.init.normal_(self.linear.weight.data.imag, std=init_std / math.sqrt(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    @property
    def weight(self):
        if self._is_cayley:
            return None  # Cayley 没有 .weight 属性
        return self.linear.weight

    @property
    def is_cayley(self) -> bool:
        return self._is_cayley

    def get_unitarity_violation(self) -> float:
        """返回酉性违背度。Cayley 时严格为 0（或极小），否则可能很大。"""
        if self._is_cayley:
            return self.linear.get_unitarity_violation().item()
        return float("inf")

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, " f"cayley={self._is_cayley}"


class QuantumGate(nn.Module):
    """量子门控层 (Quantum Gate)。

    实现设计文档中的：
        Gate_Q(|ψ⟩) = |ψ⟩ ⊙ σ_C(W_g |ψ⟩)

    其中 σ_C 是复数 Sigmoid，⊙ 是复数哈达玛积。

    门控信号在复平面第一象限，模长 < √2，相位 ∈ (0, π/2)。
    这意味着门控同时调节模长和相位，而非简单缩放。

    Args:
        dim: 特征维度
        init_std: 门控权重初始化标准差
    """

    def __init__(self, dim: int, init_std: Optional[float] = None):
        super().__init__()
        self.dim = dim
        if init_std is None:
            init_std = 1.0 / math.sqrt(dim)
        self.W_gate = nn.Parameter(
            torch.randn(dim, dim, dtype=torch.complex64) * init_std / math.sqrt(2)
        )
        self.sigmoid = ComplexSigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 复数输入 (..., dim)
        Returns:
            门控后的复数输出 (..., dim)
        """
        gate = self.sigmoid(x @ self.W_gate)
        return x * gate

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


class GatedQuantumFFN(nn.Module):
    """门控量子前馈网络 (Gated-FFN_Q)。

    FFN_Q_gated(|ψ⟩) = W_down · Gate_Q(σ_C(W_up |ψ⟩ + b))

    结构：
        |ψ⟩ → W_up → +b → ModReLU → Gate_Q → W_down

    其中 Gate_Q 使用独立门控投影 W_gate。

    门控机制的作用：
    - 动态选择激活特征（类似 SwiGLU 的效果）
    - 复数门控同时调节模长和相位
    - 门控信号 ∈ (0,1)+i(0,1)，温和的调节

    Args:
        dim: 输入/输出特征维度
        ffn_dim: 中间层维度（默认 4*dim）
        dropout: dropout 概率
        activation: 激活函数类型 ('modrelu', 'gelu')
        gate_proj_dim: 门控投影维度（默认 dim，保持原始维度）
    """

    def __init__(
        self,
        dim: int,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.0,
        activation: str = "modrelu",
        gate_proj_dim: Optional[int] = None,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim or (dim * 4)
        self.dropout_p = dropout
        self._actual_ffn_out = self.ffn_dim

        # 上投影（非方阵复数投影，扩展维度）
        self.W_up = ComplexLinear(
            dim,
            self.ffn_dim,
            use_cayley=False,
            init_std=1.0 / math.sqrt(dim),
        )

        # 复数偏置
        self.bias_up = ComplexBias(self.ffn_dim)

        # 复数激活函数
        if activation == "modrelu":
            self.activation = ModReLU(self.ffn_dim)
        elif activation == "gelu":
            from .activations import ComplexGELU

            self.activation = ComplexGELU()
        elif activation == "swiglu":
            # ComplexSwiGLU 集成门控和激活于一体，内部通过 gate_proj 产生门控信号
            # 注意：SwiGLU 自带门控，无需再叠加 QuantumGate，否则双重门控会过度抑制
            from .activations import ComplexSwiGLU

            self.activation = ComplexSwiGLU(self.ffn_dim, split_input=True)
        else:
            raise ValueError(f"Unknown activation: {activation}, 支持: modrelu, gelu, swiglu")

        # 量子门控层（SwiGLU 自带门控，跳过额外 QuantumGate 以避免双重门控）
        if activation != "swiglu":
            self.gate = QuantumGate(
                dim=self.ffn_dim,
                init_std=1.0 / math.sqrt(self.ffn_dim),
            )
        else:
            self.gate = None  # SwiGLU 内置门控，不需要额外 QuantumGate

        # 下投影（方阵时使用 Cayley 酉投影）
        self.W_down = ComplexLinear(
            self.ffn_dim,
            dim,
            use_cayley=True,  # 自动判断：方阵用 Cayley，非方阵用普通投影
            init_std=1.0 / math.sqrt(self.ffn_dim),
        )

        # LayerNorm
        from .normalization import ComplexLayerNorm

        self.norm = ComplexLayerNorm(dim)

    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        Args:
            x: 复数输入 (batch, seq_len, dim)
            training: 是否训练模式
        Returns:
            复数输出 (batch, seq_len, dim)
        """
        residual = x

        # 上投影
        up = self.W_up(x)  # (B, N, ffn_dim)
        up = self.bias_up(up)

        # 复数激活
        up = self.activation(up)

        # 量子门控（复数哈达玛积）；SwiGLU 自带门控，跳过额外 QuantumGate
        if self.gate is not None:
            up = self.gate(up)

        # Dropout
        if training and self.dropout_p > 0:
            up = complex_dropout(up, p=self.dropout_p, training=True)

        # 下投影
        down = self.W_down(up)  # (B, N, dim)

        # 归一化
        down = self.norm(down)

        # 残差连接
        return residual + down

    @torch.no_grad()
    def get_gate_statistics(self, x: torch.Tensor) -> dict:
        """计算门控信号的统计信息，用于诊断和监控门控健康度。

        门控信号 g = ComplexSigmoid(x @ W_gate) ∈ (0,1)+i(0,1)，
        其模长 |g| ∈ (0, √2)。

        常见异常模式：
        - **死门（Dead Gate）**：gate_mag_mean < 0.1，门控几乎不激活，
          说明 W_gate 梯度消失或初始化过大导致大量输入落在 Sigmoid 饱和区。
        - **开放门（Open Gate）**：gate_mag_mean > 1.3（接近 √2），
          门控几乎全通，丧失选择性。
        - **相位失调（Phase Drift）**：gate_phase_std 接近 π，
          说明门控相位完全随机，没有学到有意义的相位路由。

        仅在有 QuantumGate（即非 SwiGLU 模式）时计算门控统计。
        SwiGLU 模式下返回空字典（门控统计内置于激活函数中）。

        Args:
            x: 复数输入 (batch, seq_len, ffn_dim)，即 W_up 激活后的张量
        Returns:
            dict 包含:
                - ``gate_mag_mean``: 门控信号模长的均值（健康范围 0.3~1.0）
                - ``gate_mag_std``: 门控信号模长的标准差
                - ``gate_mag_min``: 门控信号模长的最小值
                - ``gate_mag_max``: 门控信号模长的最大值
                - ``gate_phase_mean``: 门控信号相位的均值 [弧度]
                - ``gate_phase_std``: 门控信号相位的标准差 [弧度]
                - ``gate_real_mean``: 门控实部均值（实数路由分量）
                - ``gate_imag_mean``: 门控虚部均值（虚数路由分量）
                - ``dead_gate_ratio``: 模长 < 0.1 的门控比例（死门诊断）
        """
        if self.gate is None:
            # SwiGLU 模式：无独立 QuantumGate
            return {}

        # 计算门控信号（复数）
        gate_signal = self.gate.sigmoid(x @ self.gate.W_gate)  # (..., ffn_dim)

        gate_mag = gate_signal.abs()
        gate_phase = gate_signal.angle()

        return {
            "gate_mag_mean": gate_mag.mean().item(),
            "gate_mag_std": gate_mag.std().item(),
            "gate_mag_min": gate_mag.min().item(),
            "gate_mag_max": gate_mag.max().item(),
            "gate_phase_mean": gate_phase.mean().item(),
            "gate_phase_std": gate_phase.std().item(),
            "gate_real_mean": gate_signal.real.mean().item(),
            "gate_imag_mean": gate_signal.imag.mean().item(),
            "dead_gate_ratio": (gate_mag < 0.1).float().mean().item(),
        }

    def extra_repr(self) -> str:
        return f"dim={self.dim}, ffn_dim={self.ffn_dim}, " f"dropout={self.dropout_p}"


class QuantumFFN(nn.Module):
    """量子前馈网络 (FFN_Q)。

    FFN_Q(|ψ⟩) = U_out · σ_C(W_up |ψ⟩ + b) + |ψ⟩

    Args:
        dim: 输入/输出特征维度
        ffn_dim: 中间层维度（通常 4x dim）
        dropout: dropout 概率
        use_glu: 是否使用量子门控变体（旧参数名，兼容保留）
        use_gating: 是否使用量子门控变体（推荐使用此参数名）
        activation: 激活函数类型 ('modrelu', 'gelu')
    """

    def __init__(
        self,
        dim: int,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.0,
        use_glu: bool = False,
        use_gating: bool = False,
        activation: str = "modrelu",
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim or (dim * 4)
        self.use_glu = use_glu or use_gating  # 兼容两种参数名
        self.dropout_p = dropout
        self._actual_ffn_out = self.ffn_dim

        if self.use_glu:
            # 门控模式：使用 GatedQuantumFFN 作为内部实现
            self._gated_ffn = GatedQuantumFFN(
                dim=dim,
                ffn_dim=self.ffn_dim,
                dropout=dropout,
                activation=activation,
            )
            # 暴露内部组件的接口
            self.W_up = self._gated_ffn.W_up
            self.W_down = self._gated_ffn.W_down
            self.gate = self._gated_ffn.gate
            self.activation = self._gated_ffn.activation
            self.bias_up = self._gated_ffn.bias_up
            self.norm = self._gated_ffn.norm
        else:
            # 标准模式
            self._gated_ffn = None

            # 上投影（非方阵复数投影，扩展维度）
            # 使用 ComplexLinear 而非裸 nn.Linear，获得更好的复数初始化
            self.W_up = ComplexLinear(
                dim,
                self.ffn_dim,
                use_cayley=False,
                init_std=1.0 / math.sqrt(dim),
            )

            # 复数偏置
            self.bias_up = ComplexBias(self.ffn_dim)

            # 复数激活函数
            if activation == "modrelu":
                self.activation = ModReLU(self.ffn_dim)
            elif activation == "gelu":
                from .activations import ComplexGELU

                self.activation = ComplexGELU()
            elif activation == "swiglu":
                from .activations import ComplexSwiGLU

                self.activation = ComplexSwiGLU(self.ffn_dim, split_input=True)
            else:
                raise ValueError(f"Unknown activation: {activation}")

            # 下投影（方阵时使用 Cayley 酉投影，自动判断）
            self.W_down = ComplexLinear(
                self.ffn_dim,
                dim,
                use_cayley=True,
                init_std=1.0 / math.sqrt(self.ffn_dim),
            )

            # LayerNorm
            from .normalization import ComplexLayerNorm

            self.norm = ComplexLayerNorm(dim)

    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        Args:
            x: 复数输入 (batch, seq_len, dim)
            training: 是否训练模式
        Returns:
            复数输出 (batch, seq_len, dim)
        """
        if self._gated_ffn is not None:
            return self._gated_ffn(x, training=training)

        residual = x

        # 上投影
        up = self.W_up(x)  # (B, N, ffn_dim)
        up = self.bias_up(up)

        # 复数激活
        up = self.activation(up)

        # Dropout
        if training and self.dropout_p > 0:
            up = complex_dropout(up, p=self.dropout_p, training=True)

        # 下投影
        down = self.W_down(up)  # (B, N, dim)

        # 归一化
        down = self.norm(down)

        # 残差连接
        return residual + down

    def get_unitarity_violation(self) -> Dict[str, float]:
        """返回各酉性约束的违背度。"""
        violations = {}
        violations["W_down"] = self.W_down.get_unitarity_violation()
        if self.use_glu and hasattr(self, "gate"):
            # 门控矩阵不是酉矩阵，但可以检查条件数
            gate_weight = self.gate.W_gate
            cond = torch.linalg.cond(gate_weight).item()
            violations["gate_cond"] = cond
        return violations

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, ffn_dim={self.ffn_dim}, "
            f"gated={self.use_glu}, dropout={self.dropout_p}"
        )
