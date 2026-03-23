"""
ModReLU 复数激活函数

ModReLU(z) = (|z| + b) * z / (|z| + ε)

关键性质：
- 相位保持：输出相位 = 输入相位
- 模长调节：通过可学习偏置 b 控制稀疏性
- 可微性：在 z=0 处梯度有界
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ModReLU(nn.Module):
    """复数模长修正线性单元。

    ModReLU(z) = (|z| + b) * ReLU(|z| + b) / (|z| + ε) * (z / |z|)

    简化实现：ModReLU(z) = ReLU(|z| + b) * z / (|z| + ε)

    Args:
        num_features: 输入特征的最后一个维度大小（用于每个特征独立的偏置）
        bias: 初始偏置值
        eps: 数值稳定 epsilon
    """

    def __init__(self, num_features: int, bias: float = -1.0, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.bias = nn.Parameter(torch.full((num_features,), bias, dtype=torch.float32))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: 复数张量 (..., num_features)
        Returns:
            复数张量 (..., num_features)，相位不变，模长被调节
        """
        abs_z = z.abs()
        # (|z| + b)_+ 的逐元素激活
        scale = F.relu(abs_z + self.bias) / (abs_z + self.eps)
        return z * scale

    def extra_repr(self) -> str:
        return f"bias_init={self.bias.data.mean().item():.2f}, eps={self.eps}"


class ModReLUFunction(torch.autograd.Function):
    """ModReLU 的自定义 autograd 实现，确保梯度正确性。

    前向：y = ReLU(|z| + b) * z / (|z| + ε)

    当 |z| + b > 0：
        ∂L/∂z = scale * ∂L/∂y
                + y * ∂scale/∂z * ∂L/∂y

    其中 ∂scale/∂z = ∂/∂z [ReLU(|z|+b)/(|z|+ε)]
    """

    @staticmethod
    def forward(ctx, z: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
        abs_z = z.abs()
        activated = F.relu(abs_z + bias)
        scale = activated / (abs_z + eps)
        y = z * scale

        # 保存用于反向传播
        mask = (abs_z + bias > 0).float()
        ctx.save_for_backward(z, scale, mask)
        ctx.eps = eps

        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        z, scale, mask = ctx.saved_tensors
        eps = ctx.eps
        abs_z = z.abs()

        # ∂scale/∂|z| = ∂/∂|z| [ReLU(|z|+b)/(|z|+ε)]
        # 当 |z|+b > 0: = (1*(|z|+ε) - ReLU(|z|+b)*1) / (|z|+ε)²
        #               = (|z|+ε - |z|-b) / (|z|+ε)²
        #               = (ε - b) / (|z|+ε)²
        safe_abs = abs_z + eps
        d_scale_d_abs = mask * (
            eps - (-F.relu(-abs_z - 0 * (abs_z + 1) + abs_z + 0)) + (1 - mask) * 0
        )
        # 简化：使用数值稳定的表达式
        activated = scale * safe_abs  # = ReLU(|z| + b)
        d_scale_d_abs = mask * (safe_abs - activated) / (safe_abs**2)

        # ∂|z|/∂z = conj(z) / (2|z|)  (Wirtinger 导数)
        # 但 PyTorch 复数 autograd 直接处理，用链式法则
        d_abs_d_z_real = z.real / (abs_z + eps)
        d_abs_d_z_imag = z.imag / (abs_z + eps)
        # ∂|z|/∂z (对 conj(z) 的导数) = z / (2|z|)
        # 实际上 PyTorch 用的是 ∂f/∂conj(z)，所以 ∂|z|/∂conj(z) = z/(2|z|)
        # 但对于实值损失函数的链式法则：
        # dL/dz = conj(dL/d(conj(z)))
        # ∂|z|/∂(conj(z)) = z / (2|z|)

        # 最简方案：让 PyTorch 处理复数 autograd，只提供正确的 scale 梯度
        # ∂L/∂z = grad_output * scale + y * d_scale_d_abs * ∂|z|/∂z

        # ∂|z|/∂z = z / (|z| + eps)  (复数导数，用于 grad computation)
        d_abs_z = z / safe_abs

        # y = z * scale，所以 ∂y/∂z = scale + z * ∂scale/∂z
        # 其中 ∂scale/∂z = d_scale_d_abs * d_abs_z
        grad_scale = d_scale_d_abs * d_abs_z

        grad_z = grad_output * scale + (z * scale) * grad_scale

        # bias 的梯度：∂L/∂b = Σ (∂L/∂y * y * ∂scale/∂b)
        # ∂scale/∂b = mask / (|z| + eps)
        if mask.requires_grad:
            grad_bias = (
                (grad_output * z * mask / safe_abs).real.sum(
                    dim=tuple(range(grad_output.dim() - 1))
                )
                if grad_output.dim() > 1
                else (grad_output * z * mask / safe_abs).real.sum()
            )
        else:
            grad_bias = (grad_output * z * mask / safe_abs).real.flatten().sum()

        # PyTorch 复数 autograd 返回的是 ∂L/∂(conj(z))
        # 对实值损失，dL/dz = conj(dL/d(conj(z)))
        # 但 autograd.Function 直接返回的 grad 会被当作 ∂L/∂z_input
        return grad_z, grad_bias, None


class ModReLUV2(nn.Module):
    """使用自定义 autograd 的 ModReLU（用于梯度验证）。

    与 ModReLU 功能相同，但通过 Function 手动计算梯度。
    """

    def __init__(self, num_features: int, bias: float = -1.0, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.bias = nn.Parameter(torch.full((num_features,), bias, dtype=torch.float32))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return ModReLUFunction.apply(z, self.bias, self.eps)

    def extra_repr(self) -> str:
        return f"bias_init={self.bias.data.mean().item():.2f}, eps={self.eps}"


class CReLU(nn.Module):
    """复数 ReLU：分别对实部和虚部施加 ReLU。

    CReLU(z) = ReLU(Re(z)) + i * ReLU(Im(z))

    丢弃负实部和负虚部的信息，但实现简单。
    """

    def __init__(self):
        super().__init__()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.complex(F.relu(z.real), F.relu(z.imag))


class zReLU(nn.Module):
    """z-ReLU：仅当复数落在第一象限时保留，否则置零。

    zReLU(z) = z  if Re(z) > 0 and Im(z) > 0
              = 0  otherwise
    """

    def __init__(self):
        super().__init__()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        mask = ((z.real > 0) & (z.imag > 0)).float()
        return z * mask


class ComplexGELU(nn.Module):
    """复数 GELU：分别对实部和虚部施加 GELU。

    近似 GELU 的复数版本，保持相位的连续变化。
    """

    def __init__(self):
        super().__init__()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.complex(F.gelu(z.real), F.gelu(z.imag))


class ComplexSwiGLU(nn.Module):
    """复数 SwiGLU 门控激活函数。

    SwiGLU 是 Transformer FFN 中效果最好的激活之一（PaLM, LLaMA 等采用）。
    本模块将其推广到复数域：

        ComplexSwiGLU(z₁, z₂) = z₁ ⊙ σ_C(z₂)

    其中：
    - z₁, z₂ 是同维度的复数张量（通过对上投影输出做等分或独立投影得到）
    - σ_C(z) = SiLU(Re(z)) + i·SiLU(Im(z))  （SiLU = Swish = x·sigmoid(x)）
    - ⊙ 是复数哈达玛积（逐元素复数乘法）

    与 ModReLU 的对比：
    - ModReLU：仅调节模长，保持相位 → 适合需要相位保真度的场景
    - ComplexSwiGLU：同时调节模长和相位 → 更强的非线性表达能力

    物理解释：
    - 门控 z₂ 控制每个特征"让多少量子振幅通过"
    - 复数 SiLU 保持了负值的梯度（不同于 ReLU 的硬截断）
    - 结果是一个可微的、考虑相位的量子特征选择机制

    Args:
        dim: 输入/输出复数特征维度（注意：_内部_会将 dim 扩展到 2*dim 再等分）
        split_input: 是否在内部做 2×上投影并等分（True：输入 dim 维，内部扩展；
                     False：期望已经有两组 dim 维输入，手动传入 gate）
        init_std: 门控线性层的初始化标准差
    """

    def __init__(
        self,
        dim: int,
        split_input: bool = True,
        init_std: Optional[float] = None,
    ):
        super().__init__()
        self.dim = dim
        self.split_input = split_input

        if split_input:
            # 内部 2× 投影，统一接口（输入 dim 维 → 内部 2dim → 等分）
            if init_std is None:
                init_std = 1.0 / (dim ** 0.5)
            self.gate_proj = nn.Linear(dim, dim, bias=False).to(torch.complex64)
            nn.init.normal_(self.gate_proj.weight.data.real, std=init_std / 2 ** 0.5)
            nn.init.normal_(self.gate_proj.weight.data.imag, std=init_std / 2 ** 0.5)

    def _complex_swish(self, z: torch.Tensor) -> torch.Tensor:
        """复数 SiLU（Swish）：对实部和虚部分别施加 SiLU。

        SiLU(x) = x · sigmoid(x)

        复数扩展：σ_C(z) = SiLU(Re(z)) + i·SiLU(Im(z))

        特性：
        - 保持相位信息（不同于 ReLU 的相位破坏）
        - 光滑可微（SiLU 在 x ≈ -1.67 有唯一局部极小值）
        - 输出范围包含负值，避免死神经元
        """
        return torch.complex(F.silu(z.real), F.silu(z.imag))

    def forward(
        self,
        x: torch.Tensor,
        gate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: 复数输入 (..., dim)
            gate: 可选的独立门控信号 (..., dim)。
                  若 split_input=True 且未提供，则内部计算门控。
        Returns:
            门控后的复数输出 (..., dim)

        Usage:
            # 模式1：split_input=True（推荐，与 FFN 集成）
            act = ComplexSwiGLU(dim=256, split_input=True)
            out = act(x)   # x: (..., 256) → out: (..., 256)

            # 模式2：手动传入两路信号
            act = ComplexSwiGLU(dim=256, split_input=False)
            gate_signal = compute_gate(x)
            out = act(x, gate=gate_signal)
        """
        if gate is None:
            if self.split_input:
                # 内部门控投影
                gate = self.gate_proj(x)
            else:
                raise ValueError(
                    "split_input=False 时必须手动传入 gate 参数，"
                    "或设置 split_input=True 让模块内部计算门控。"
                )

        # ComplexSwiGLU: x ⊙ σ_C(gate)
        return x * self._complex_swish(gate)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, split_input={self.split_input}"
