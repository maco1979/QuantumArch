"""
量子态初始化工具模块
QuantumArch State Initialization Utilities

提供多种量子态初始化策略，包括：
- 均匀叠加初始化
- 随机量子态生成
- 相干态初始化
- 纠缠态初始化
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List


def uniform_superposition_init(
    size: Tuple[int, ...],
    dtype: torch.dtype = torch.cfloat,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    均匀叠加态初始化
    
    生成所有基态等权叠加的量子态，类似Hadamard门作用后的状态。
    |ψ⟩ = (1/√N) Σ|k⟩
    
    Args:
        size: 张量形状
        dtype: 数据类型，默认 cfloat (复数)
        device: 目标设备
    
    Returns:
        复数张量，模长均为 1/√N
    """
    n = size[-1]
    real = torch.ones(size, dtype=torch.float32, device=device) / np.sqrt(n)
    imag = torch.zeros(size, dtype=torch.float32, device=device)
    return torch.complex(real, imag).to(dtype)


def random_pure_state_init(
    size: Tuple[int, ...],
    dtype: torch.dtype = torch.cfloat,
    device: Optional[torch.device] = None,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    随机纯量子态初始化
    
    生成归一化的随机复数量子态，满足 ⟨ψ|ψ⟩ = 1。
    
    Args:
        size: 张量形状
        dtype: 数据类型
        device: 目标设备
        seed: 随机种子（可选）
    
    Returns:
        归一化复数张量
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    real = torch.randn(size, dtype=torch.float32, device=device)
    imag = torch.randn(size, dtype=torch.float32, device=device)
    state = torch.complex(real, imag)
    
    # 归一化
    norm = torch.norm(state, dim=-1, keepdim=True).clamp(min=1e-8)
    return (state / norm).to(dtype)


def coherent_state_init(
    size: Tuple[int, ...],
    alpha: complex = 1.0 + 0.0j,
    dtype: torch.dtype = torch.cfloat,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    相干态初始化（类Glauber相干态）
    
    用于模拟振荡子模式的相干量子态，振幅为alpha。
    ψ_n = exp(-|α|²/2) · α^n / √(n!)
    
    Args:
        size: 张量形状，最后维度为截断的Fock空间维数
        alpha: 相干态振幅（复数）
        dtype: 数据类型
        device: 目标设备
    
    Returns:
        相干态张量
    """
    n_fock = size[-1]
    batch_shape = size[:-1]
    
    # 计算Fock空间系数
    ns = torch.arange(n_fock, dtype=torch.float32, device=device)
    log_coeffs = (
        -0.5 * abs(alpha) ** 2
        + ns * np.log(abs(alpha) + 1e-12)
        - 0.5 * torch.lgamma(ns + 1)
    )
    
    # 添加相位
    phases = ns * np.angle(alpha)
    real_part = torch.exp(log_coeffs) * torch.cos(phases)
    imag_part = torch.exp(log_coeffs) * torch.sin(phases)
    
    coeffs = torch.complex(real_part, imag_part)
    
    # 广播到 batch 形状
    coeffs = coeffs.expand(*batch_shape, n_fock)
    return coeffs.to(dtype)


def bell_state_init(
    batch_size: int,
    state_type: str = "phi_plus",
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Bell态（最大纠缠态）初始化
    
    四种Bell态：
    - phi_plus:  |Φ+⟩ = (|00⟩ + |11⟩) / √2
    - phi_minus: |Φ-⟩ = (|00⟩ - |11⟩) / √2
    - psi_plus:  |Ψ+⟩ = (|01⟩ + |10⟩) / √2
    - psi_minus: |Ψ-⟩ = (|01⟩ - |10⟩) / √2
    
    Args:
        batch_size: 批次大小
        state_type: Bell态类型
        device: 目标设备
    
    Returns:
        形状为 (batch_size, 4) 的复数张量，表示二量子比特态
    """
    state_map = {
        "phi_plus":  [1, 0, 0, 1],
        "phi_minus": [1, 0, 0, -1],
        "psi_plus":  [0, 1, 1, 0],
        "psi_minus": [0, 1, -1, 0],
    }
    if state_type not in state_map:
        raise ValueError(f"未知Bell态类型: {state_type}，可选: {list(state_map.keys())}")
    
    coeffs = torch.tensor(state_map[state_type], dtype=torch.float32, device=device)
    coeffs = coeffs / coeffs.norm()
    state = torch.complex(coeffs, torch.zeros_like(coeffs))
    return state.unsqueeze(0).expand(batch_size, -1)


class QuantumStateInitializer(nn.Module):
    """
    可学习的量子态初始化器
    
    通过可训练参数生成初始量子态，支持：
    - 参数化初始态
    - 混合态初始化
    - 领域特定初始化
    """
    
    def __init__(
        self,
        dim: int,
        init_strategy: str = "uniform",
        learnable: bool = True,
    ):
        """
        Args:
            dim: 量子态维度
            init_strategy: 初始化策略，可选 'uniform', 'random', 'zero'
            learnable: 是否为可学习参数
        """
        super().__init__()
        self.dim = dim
        self.init_strategy = init_strategy
        
        if init_strategy == "uniform":
            init_real = torch.ones(dim) / np.sqrt(dim)
            init_imag = torch.zeros(dim)
        elif init_strategy == "random":
            init_real = torch.randn(dim)
            init_imag = torch.randn(dim)
            norm = torch.sqrt(init_real**2 + init_imag**2).sum().sqrt()
            init_real = init_real / (norm + 1e-8)
            init_imag = init_imag / (norm + 1e-8)
        elif init_strategy == "zero":
            init_real = torch.zeros(dim)
            init_imag = torch.zeros(dim)
            init_real[0] = 1.0  # |0⟩ 基态
        else:
            raise ValueError(f"未知初始化策略: {init_strategy}")
        
        if learnable:
            self.state_real = nn.Parameter(init_real)
            self.state_imag = nn.Parameter(init_imag)
        else:
            self.register_buffer('state_real', init_real)
            self.register_buffer('state_imag', init_imag)
        
        self.learnable = learnable
    
    def forward(self, batch_size: int) -> torch.Tensor:
        """
        生成批次量子态
        
        Args:
            batch_size: 批次大小
        
        Returns:
            形状为 (batch_size, dim) 的复数张量，已归一化
        """
        state = torch.complex(self.state_real, self.state_imag)
        # 归一化确保合法量子态
        norm = state.norm().clamp(min=1e-8)
        state = state / norm
        return state.unsqueeze(0).expand(batch_size, -1)
    
    def entropy(self) -> torch.Tensor:
        """计算初始态的von Neumann熵（纯态熵为0，但可用于监控）"""
        state = torch.complex(self.state_real, self.state_imag)
        probs = (state.abs() ** 2).clamp(min=1e-10)
        probs = probs / probs.sum()
        return -(probs * probs.log()).sum()
    
    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, "
            f"init_strategy='{self.init_strategy}', "
            f"learnable={self.learnable}"
        )
