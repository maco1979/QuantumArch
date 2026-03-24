"""
量子纠错编码辅助函数
QuantumArch Error Correction Utilities

在经典模拟的量子架构中，实现量子纠错思路的数值稳定化：
- 投影噪声抑制
- 奇偶校验软化
- 量子通道噪声模型
- 纠错码辅助的特征编码
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


# ─── 量子噪声信道 ──────────────────────────────────────────────────────────────

def depolarizing_channel(
    rho: torch.Tensor,
    error_rate: float,
) -> torch.Tensor:
    """
    去极化信道：以概率 p 随机施加 I/X/Y/Z 门误差
    
    ε(ρ) = (1 - p)ρ + (p/3)(XρX† + YρY† + ZρZ†)
    
    Args:
        rho: 量子态密度矩阵，形状 (..., d, d)
        error_rate: 错误概率 p ∈ [0, 1]
    
    Returns:
        加噪后的密度矩阵
    """
    if error_rate <= 0:
        return rho
    
    d = rho.shape[-1]
    device = rho.device
    
    # Pauli 矩阵（2×2）
    X = torch.tensor([[0, 1], [1, 0]], dtype=torch.cfloat, device=device)
    Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.cfloat, device=device)
    Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.cfloat, device=device)
    
    if d == 2:
        paulis = [X, Y, Z]
        noisy = (1 - error_rate) * rho
        for P in paulis:
            noisy = noisy + (error_rate / 3) * (P @ rho @ P.conj().T)
        return noisy
    
    # 高维：仅施加随机相位噪声（近似去极化）
    phase_noise = torch.exp(
        1j * error_rate * torch.randn(*rho.shape[:-2], d, dtype=torch.float32, device=device)
    ).to(torch.cfloat)
    rho_noisy = rho * phase_noise.unsqueeze(-1) * phase_noise.conj().unsqueeze(-2)
    return (1 - error_rate) * rho + error_rate * rho_noisy


def amplitude_damping_channel(
    state: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """
    振幅阻尼信道（模拟能量损耗）
    
    E_0 = [[1, 0], [0, √(1-γ)]]
    E_1 = [[0, √γ], [0, 0]]
    
    Args:
        state: 量子态向量，形状 (..., 2)
        gamma: 阻尼参数 γ ∈ [0, 1]
    
    Returns:
        阻尼后的量子态（归一化）
    """
    if gamma <= 0:
        return state
    
    alpha = state[..., 0]  # |0⟩ 振幅
    beta = state[..., 1]   # |1⟩ 振幅
    
    # E_0: |0⟩→|0⟩, |1⟩→√(1-γ)|1⟩
    alpha_new = alpha
    beta_new = beta * np.sqrt(1 - gamma)
    
    # E_1: |1⟩→√γ|0⟩ (跃迁到基态)
    alpha_decay = beta * np.sqrt(gamma)
    alpha_new = alpha_new + alpha_decay
    
    state_new = torch.stack([alpha_new, beta_new], dim=-1)
    norm = state_new.abs().pow(2).sum(-1, keepdim=True).sqrt().clamp(min=1e-8)
    return state_new / norm


# ─── 投影误差抑制 ──────────────────────────────────────────────────────────────

def project_to_codespace(
    features: torch.Tensor,
    codebook: torch.Tensor,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将特征软投影到码空间（量子纠错码思想的特征稳定化）
    
    通过与码字的相似度加权，将嘈杂特征拉回有效子空间。
    
    Args:
        features: 输入特征，形状 (batch, dim)
        codebook: 码字矩阵，形状 (n_codes, dim)
        temperature: 软投影温度（越低越接近硬投影）
    
    Returns:
        (投影后特征, 码字权重) 元组
    """
    # 计算与每个码字的相似度
    if features.is_complex():
        # 复数内积
        scores = (features.unsqueeze(1) * codebook.unsqueeze(0).conj()).real.sum(-1)
    else:
        scores = torch.matmul(features, codebook.T)  # (batch, n_codes)
    
    # 软投影权重
    weights = F.softmax(scores / temperature, dim=-1)  # (batch, n_codes)
    
    # 加权重构
    projected = torch.matmul(weights, codebook.to(features.dtype))  # (batch, dim)
    
    return projected, weights


class QuantumErrorMitigator(nn.Module):
    """
    量子误差缓解模块
    
    结合三种策略：
    1. 码空间投影：维持特征在有效子空间内
    2. 对称化：强制强制量子态满足物理约束
    3. 后选择：过滤低置信度的量子测量结果
    """
    
    def __init__(
        self,
        dim: int,
        n_codes: int = 64,
        projection_temperature: float = 0.5,
        symmetrize: bool = True,
    ):
        """
        Args:
            dim: 特征维度
            n_codes: 码字数量
            projection_temperature: 投影温度
            symmetrize: 是否启用对称化
        """
        super().__init__()
        self.dim = dim
        self.n_codes = n_codes
        self.projection_temperature = projection_temperature
        self.symmetrize = symmetrize
        
        # 可学习码字
        self.codebook = nn.Parameter(
            F.normalize(torch.randn(n_codes, dim), dim=-1)
        )
        
        # 后处理投影
        self.output_proj = nn.Linear(dim, dim)
    
    def forward(
        self,
        x: torch.Tensor,
        error_rate: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        应用误差缓解
        
        Args:
            x: 输入特征，形状 (batch, seq, dim) 或 (batch, dim)
            error_rate: 模拟噪声率（训练时可用于数据增强）
        
        Returns:
            (缓解后特征, 码字权重)
        """
        batch_shape = x.shape[:-1]
        x_flat = x.view(-1, self.dim)
        
        # 模拟量子噪声（如需要）
        if error_rate > 0 and self.training:
            noise = torch.randn_like(x_flat) * error_rate
            x_flat = x_flat + noise
        
        # 对称化：取实部（若复数输入）
        if x_flat.is_complex() and self.symmetrize:
            x_real = x_flat.real
        else:
            x_real = x_flat.float()
        
        # 码空间投影
        projected, weights = project_to_codespace(
            x_real,
            self.codebook,
            self.projection_temperature,
        )
        
        # 残差连接 + 输出映射
        out = self.output_proj(projected + x_real)
        out = out.view(*batch_shape, self.dim)
        
        return out, weights.view(*batch_shape, self.n_codes)
    
    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, "
            f"n_codes={self.n_codes}, "
            f"temperature={self.projection_temperature}"
        )


class ParityCheck(nn.Module):
    """
    奇偶校验层（量子稳定子码思想）
    
    检测特征中的"逻辑误差"（不满足某些软约束）并软修正。
    """
    
    def __init__(self, dim: int, n_checks: int = 16):
        """
        Args:
            dim: 特征维度
            n_checks: 奇偶校验约束数量
        """
        super().__init__()
        self.dim = dim
        self.n_checks = n_checks
        
        # 校验矩阵 H: (n_checks, dim)
        self.H = nn.Parameter(torch.randn(n_checks, dim) * 0.1)
        
        # 误差估计网络
        self.error_estimator = nn.Sequential(
            nn.Linear(n_checks, n_checks * 2),
            nn.GELU(),
            nn.Linear(n_checks * 2, dim),
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算校验子并修正误差
        
        Args:
            x: 输入特征，形状 (..., dim)
        
        Returns:
            (修正后特征, 校验子) 元组
        """
        # 计算校验子 s = Hx mod 2（连续近似）
        syndrome = torch.sigmoid(torch.matmul(x.float(), self.H.T))  # (..., n_checks)
        
        # 估计并修正误差
        error_estimate = self.error_estimator(syndrome - 0.5)  # (..., dim)
        x_corrected = x - error_estimate.to(x.dtype)
        
        return x_corrected, syndrome
    
    def extra_repr(self) -> str:
        return f"dim={self.dim}, n_checks={self.n_checks}"
