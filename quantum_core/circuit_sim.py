"""
量子电路模拟基础层
QuantumArch Circuit Simulation Layer

在经典硬件上模拟量子门操作，作为量子架构的基础组件。
支持单量子比特门、双量子比特门及参数化量子门。
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Tuple
import math


# ─── 基础量子门矩阵 ───────────────────────────────────────────────────────────

def hadamard_gate(device=None) -> torch.Tensor:
    """Hadamard门：H = (1/√2)[[1,1],[1,-1]]"""
    H = torch.tensor([[1, 1], [1, -1]], dtype=torch.cfloat) / math.sqrt(2)
    return H.to(device) if device else H


def pauli_x_gate(device=None) -> torch.Tensor:
    """Pauli-X门（量子NOT）：σ_x = [[0,1],[1,0]]"""
    return torch.tensor([[0, 1], [1, 0]], dtype=torch.cfloat, device=device)


def pauli_y_gate(device=None) -> torch.Tensor:
    """Pauli-Y门：σ_y = [[0,-i],[i,0]]"""
    return torch.tensor([[0, -1j], [1j, 0]], dtype=torch.cfloat, device=device)


def pauli_z_gate(device=None) -> torch.Tensor:
    """Pauli-Z门：σ_z = [[1,0],[0,-1]]"""
    return torch.tensor([[1, 0], [0, -1]], dtype=torch.cfloat, device=device)


def phase_gate(phi: float, device=None) -> torch.Tensor:
    """相位门：P(φ) = [[1,0],[0,e^{iφ}]]"""
    return torch.tensor(
        [[1, 0], [0, complex(math.cos(phi), math.sin(phi))]],
        dtype=torch.cfloat, device=device
    )


def rotation_x(theta: float, device=None) -> torch.Tensor:
    """绕X轴旋转门：Rx(θ) = exp(-iθσ_x/2)"""
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return torch.tensor(
        [[c, -1j * s], [-1j * s, c]],
        dtype=torch.cfloat, device=device
    )


def rotation_y(theta: float, device=None) -> torch.Tensor:
    """绕Y轴旋转门：Ry(θ) = exp(-iθσ_y/2)"""
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return torch.tensor(
        [[c, -s], [s, c]],
        dtype=torch.cfloat, device=device
    )


def rotation_z(theta: float, device=None) -> torch.Tensor:
    """绕Z轴旋转门：Rz(θ) = exp(-iθσ_z/2)"""
    e_plus = complex(math.cos(theta / 2), -math.sin(theta / 2))
    e_minus = complex(math.cos(theta / 2), math.sin(theta / 2))
    return torch.tensor(
        [[e_plus, 0], [0, e_minus]],
        dtype=torch.cfloat, device=device
    )


def cnot_gate(device=None) -> torch.Tensor:
    """
    CNOT门（受控NOT）：4x4酉矩阵
    控制比特在低位，目标比特在高位
    """
    return torch.tensor(
        [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 0, 1],
         [0, 0, 1, 0]],
        dtype=torch.cfloat, device=device
    )


# ─── 参数化量子门层 ────────────────────────────────────────────────────────────

class ParametricRotationLayer(nn.Module):
    """
    参数化单量子比特旋转层
    
    学习Bloch球上的旋转角度 (θ_x, θ_y, θ_z)，
    对应 U(θ) = Rz(θ_z) Ry(θ_y) Rx(θ_x)
    """
    
    def __init__(self, n_qubits: int):
        """
        Args:
            n_qubits: 量子比特数量（每个独立学习旋转角度）
        """
        super().__init__()
        self.n_qubits = n_qubits
        self.theta_x = nn.Parameter(torch.zeros(n_qubits))
        self.theta_y = nn.Parameter(torch.zeros(n_qubits))
        self.theta_z = nn.Parameter(torch.zeros(n_qubits))
    
    def get_unitary(self, qubit_idx: int) -> torch.Tensor:
        """获取第 qubit_idx 个量子比特的酉矩阵"""
        device = self.theta_x.device
        Rx = rotation_x(self.theta_x[qubit_idx].item(), device)
        Ry = rotation_y(self.theta_y[qubit_idx].item(), device)
        Rz = rotation_z(self.theta_z[qubit_idx].item(), device)
        return Rz @ Ry @ Rx
    
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        对批次量子态应用参数化旋转
        
        Args:
            states: 形状 (batch, n_qubits, 2) 的复数张量
        
        Returns:
            旋转后的量子态，形状相同
        """
        B, N, _ = states.shape
        assert N == self.n_qubits, f"量子比特数不匹配: {N} vs {self.n_qubits}"
        
        out = torch.zeros_like(states)
        for i in range(self.n_qubits):
            U = self.get_unitary(i)  # (2, 2)
            # states[:, i, :] → (B, 2), U → (2, 2)
            out[:, i, :] = (states[:, i, :].unsqueeze(-1) * U).sum(-2)
        return out
    
    def extra_repr(self) -> str:
        return f"n_qubits={self.n_qubits}"


class EntanglingLayer(nn.Module):
    """
    纠缠层：对相邻量子比特应用参数化受控旋转
    
    实现 ZZ-coupling 和 XX-coupling 纠缠操作：
    U_ent = exp(-i·J·σ_z⊗σ_z·t)
    """
    
    def __init__(self, n_qubits: int, coupling_type: str = "zz"):
        """
        Args:
            n_qubits: 量子比特数
            coupling_type: 耦合类型，'zz' 或 'xx'
        """
        super().__init__()
        self.n_qubits = n_qubits
        self.coupling_type = coupling_type
        # 每对相邻量子比特的耦合强度
        n_pairs = n_qubits - 1
        self.coupling_strength = nn.Parameter(torch.randn(n_pairs) * 0.1)
    
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        应用纠缠耦合
        
        通过相位调制实现近似纠缠操作（轻量化实现）
        
        Args:
            states: 形状 (batch, n_qubits, 2) 的复数张量
        
        Returns:
            纠缠后的量子态
        """
        B, N, D = states.shape
        out = states.clone()
        
        for i in range(N - 1):
            J = self.coupling_strength[i]
            # ZZ相互作用：ψ_i ⊗ ψ_{i+1} → exp(-iJt) 相位
            phase = torch.exp(torch.tensor(0 + 1j) * J.to(torch.cfloat))
            # 相位作用于|11⟩分量（简化实现）
            out[:, i, 1] = out[:, i, 1] * phase
            out[:, i + 1, 1] = out[:, i + 1, 1] * phase.conj()
        
        return out
    
    def extra_repr(self) -> str:
        return f"n_qubits={self.n_qubits}, coupling_type='{self.coupling_type}'"


class QuantumCircuitLayer(nn.Module):
    """
    完整量子电路层
    
    结构：
    [ParametricRotationLayer] → [EntanglingLayer] → [ParametricRotationLayer]
    
    适合嵌入到量子架构的更大模型中。
    """
    
    def __init__(
        self,
        n_qubits: int,
        n_layers: int = 2,
        coupling_type: str = "zz",
    ):
        """
        Args:
            n_qubits: 量子比特数
            n_layers: 旋转+纠缠堆叠层数
            coupling_type: 纠缠耦合类型
        """
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        rot_layers = []
        ent_layers = []
        for _ in range(n_layers):
            rot_layers.append(ParametricRotationLayer(n_qubits))
            ent_layers.append(EntanglingLayer(n_qubits, coupling_type))
        
        self.rot_layers = nn.ModuleList(rot_layers)
        self.ent_layers = nn.ModuleList(ent_layers)
        self.final_rot = ParametricRotationLayer(n_qubits)
    
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        执行量子电路
        
        Args:
            states: 形状 (batch, n_qubits, 2) 的复数张量
        
        Returns:
            经电路演化的量子态
        """
        x = states
        for rot, ent in zip(self.rot_layers, self.ent_layers):
            x = rot(x)
            x = ent(x)
        x = self.final_rot(x)
        return x
    
    def count_parameters(self) -> int:
        """统计可训练参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def extra_repr(self) -> str:
        return (
            f"n_qubits={self.n_qubits}, "
            f"n_layers={self.n_layers}, "
            f"params={self.count_parameters()}"
        )
