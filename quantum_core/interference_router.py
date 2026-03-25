"""
QIR 量子干涉路由器 (Quantum Interference Router)

核心思想：利用量子相长/相消干涉机制对 token 进行动态稀疏路由，
仅激活与当前查询"干涉相长"的 token，抑制"干涉相消"的 token。

理论基础：
    对于两个复数振幅 α, β：
        相长干涉：|α + β|² > |α|² + |β|²  →  2Re(ᾱβ) > 0
        相消干涉：|α + β|² < |α|² + |β|²  →  2Re(ᾱβ) < 0

    干涉强度：
        I(α, β) = 2Re(ᾱβ) = 2(Re(α)Re(β) + Im(α)Im(β))

路由决策：
    - I > τ_constructive：相长干涉，token 被激活（权重 = 1 + I/I_max）
    - I < τ_destructive ：相消干涉，token 被抑制（权重 = 0 或 ε）
    - 其他情况          ：部分激活（权重 = sigmoid(I/T)，T 为温度）

架构位置：
    QIR 作为 QSA 的前置稀疏过滤器：
        Input → QIR（稀疏化 K/V）→ QSA（仅对活跃 token 计算注意力）

与 QSA Top-K 的差异：
    - QSA Top-K：基于 Born 概率选出概率最高的 K 个 token（统计路由）
    - QIR：基于相位干涉决定 token 激活状态（相位感知路由）
    二者互补，可组合使用。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple

from .complex_ops import (
    interference_score,
    complex_inner_product,
    phase_coherence,
    normalize_quantum_state,
    born_normalize,
)
from .unitary import CayleyLinear


# ──────────────────────────────────────────────
# 干涉强度计算
# ──────────────────────────────────────────────


def pairwise_interference(
    Q: torch.Tensor,
    K: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """计算查询-键对之间的成对干涉强度矩阵。

    对于查询 q_i 和键 k_j，干涉强度为：
        I(q_i, k_j) = 2 · Re(q̄_i · k_j · 1/d)
                    = 2 · Re(inner_product(q_i, k_j)) / d

    其中 d 是特征维度（缩放因子，防止高维时数值爆炸）。

    Args:
        Q: 查询张量 (B, H, N, d) 复数
        K: 键张量 (B, H, M, d) 复数
        normalize: 是否按维度缩放（类似 Attention 的 1/√d 缩放）
    Returns:
        干涉矩阵 (B, H, N, M) 实数，范围约 (-1, 1)（归一化后）
    """
    # ⟨q_i | k_j⟩ = Σ_d conj(q_id) * k_jd
    inner = torch.einsum("bhnd,bhmd->bhnm", Q.conj(), K)  # (B, H, N, M) 复数

    # 干涉强度 = 2 * Re(inner)
    inter = 2.0 * inner.real  # (B, H, N, M) 实数

    if normalize:
        d = Q.shape[-1]
        inter = inter / d  # 防止维度爆炸

    return inter


# ──────────────────────────────────────────────
# 稀疏干涉路由
# ──────────────────────────────────────────────


class SparseInterferenceGate(nn.Module):
    """稀疏干涉门：根据干涉强度决定 token 是否激活。

    每个查询 q_i 根据其与所有键 k_j 的干涉强度，生成一个
    稀疏的布尔路由掩码（或软权重），决定哪些 token 参与计算。

    激活策略（三段式）：
    ┌──────────────────────────────────────────┐
    │  I < τ_d（相消域）：权重 → 0，token 被抑制 │
    │  τ_d ≤ I ≤ τ_c（中间域）：软激活          │
    │  I > τ_c（相长域）：权重 → 1，token 被增强 │
    └──────────────────────────────────────────┘

    软激活使用 sigmoid((I - τ_d) / T)，保证在训练时有梯度流。

    Args:
        tau_constructive: 相长干涉阈值（高于此值强激活）
        tau_destructive: 相消干涉阈值（低于此值被抑制）
        temperature: 软激活温度（低温 → 更稀疏，高温 → 更稠密）
        learnable_tau: 是否将阈值设为可学习参数
    """

    def __init__(
        self,
        tau_constructive: float = 0.1,
        tau_destructive: float = -0.1,
        temperature: float = 1.0,
        learnable_tau: bool = True,
    ):
        super().__init__()
        self.temperature = temperature

        if learnable_tau:
            # 使用 softplus 保证阈值有序：τ_d < τ_c
            # 参数化为 τ_d = -softplus(w_d), τ_c = softplus(w_c)
            init_d = math.log(math.expm1(abs(tau_destructive)))  # softplus 逆
            init_c = math.log(math.expm1(tau_constructive))
            self.w_destructive = nn.Parameter(torch.tensor(init_d, dtype=torch.float32))
            self.w_constructive = nn.Parameter(torch.tensor(init_c, dtype=torch.float32))
        else:
            self.register_buffer(
                "w_destructive", torch.tensor(tau_destructive, dtype=torch.float32)
            )
            self.register_buffer(
                "w_constructive", torch.tensor(tau_constructive, dtype=torch.float32)
            )
        self.learnable_tau = learnable_tau

    @property
    def tau_destructive(self) -> torch.Tensor:
        """相消干涉阈值（始终为负或零）。"""
        if self.learnable_tau:
            return -F.softplus(self.w_destructive)
        return self.w_destructive

    @property
    def tau_constructive(self) -> torch.Tensor:
        """相长干涉阈值（始终为正或零）。"""
        if self.learnable_tau:
            return F.softplus(self.w_constructive)
        return self.w_constructive

    def forward(
        self,
        interference: torch.Tensor,
        hard: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """根据干涉强度生成路由权重。

        Args:
            interference: 干涉强度矩阵 (B, H, N, M) 实数
            hard: 是否使用硬阈值（推理时 True，训练时 False 保留梯度）
        Returns:
            (routing_weights, metrics)
            routing_weights: 路由权重 (B, H, N, M) 实数，范围 [0, 1]
            metrics: 稀疏度等统计信息
        """
        tau_d = self.tau_destructive
        tau_c = self.tau_constructive

        if hard:
            # 硬路由：二值掩码
            routing_weights = (interference > tau_d).float()
        else:
            # 软路由：sigmoid 插值
            # 相消域 (I < τ_d)：sigmoid 输出 ≈ 0
            # 相长域 (I > τ_c)：sigmoid 输出 ≈ 1
            # 中间域：平滑过渡
            routing_weights = torch.sigmoid((interference - tau_d) / (self.temperature + 1e-6))

        # 计算稀疏度（权重 > 0.5 的比例）
        active_ratio = (routing_weights > 0.5).float().mean().item()
        constructive_ratio = (interference > tau_c).float().mean().item()
        destructive_ratio = (interference < tau_d).float().mean().item()

        metrics = {
            "qir_active_ratio": active_ratio,
            "qir_constructive_ratio": constructive_ratio,
            "qir_destructive_ratio": destructive_ratio,
            "qir_tau_constructive": tau_c.item() if hasattr(tau_c, "item") else float(tau_c),
            "qir_tau_destructive": tau_d.item() if hasattr(tau_d, "item") else float(tau_d),
        }

        return routing_weights, metrics


# ──────────────────────────────────────────────
# 量子干涉路由器主模块
# ──────────────────────────────────────────────


class QuantumInterferenceRouter(nn.Module):
    """量子干涉路由器 (QIR)。

    作为 QSA 的前置过滤器，通过相位干涉将注意力矩阵稀疏化：
    1. 将输入投影为路由查询/键（独立于 QSA 的 Q/K/V）
    2. 计算成对干涉强度矩阵
    3. 生成路由掩码，过滤相消干涉的 token
    4. 将路由权重返回给 QSA，与 Born 概率相乘得到最终注意力权重

    与 QSA 的集成：
        attn_final = Born_prob * routing_weight
        attn_final /= attn_final.sum(dim=-1, keepdim=True)

    Args:
        dim: 输入特征维度
        num_heads: 路由头数（通常与 QSA 一致）
        router_dim: 路由投影维度（None 则等于 dim // num_heads）
        tau_constructive: 相长干涉阈值
        tau_destructive: 相消干涉阈值
        temperature: 软路由温度
        learnable_tau: 是否学习阈值参数
        use_phase_routing: 是否额外使用相位相干性作为路由辅助信号
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        router_dim: Optional[int] = None,
        tau_constructive: float = 0.1,
        tau_destructive: float = -0.1,
        temperature: float = 1.0,
        learnable_tau: bool = True,
        use_phase_routing: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.router_dim = router_dim or (dim // num_heads)
        self.use_phase_routing = use_phase_routing

        # 路由投影（使用轻量 CayleyLinear 保证酉性，维持量子态性质）
        self.router_Q = CayleyLinear(dim, self.router_dim * num_heads, init_scale=0.01)
        self.router_K = CayleyLinear(dim, self.router_dim * num_heads, init_scale=0.01)

        # 干涉稀疏门
        self.gate = SparseInterferenceGate(
            tau_constructive=tau_constructive,
            tau_destructive=tau_destructive,
            temperature=temperature,
            learnable_tau=learnable_tau,
        )

        # 可选的相位路由增强：使用局部相位相干性调制路由权重
        if use_phase_routing:
            self.phase_router = nn.Linear(1, 1, bias=True)  # 标量映射
            nn.init.constant_(self.phase_router.weight, 1.0)
            nn.init.constant_(self.phase_router.bias, 0.0)

    def forward(
        self,
        x: torch.Tensor,
        training: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算量子干涉路由掩码。

        Args:
            x: 复数输入 (batch, seq_len, dim)
            training: 是否训练模式（控制软/硬路由）
        Returns:
            (routing_weights, metrics)
            routing_weights: 路由权重 (B, H, N, N)，作为注意力掩码使用
            metrics: 路由统计信息
        """
        B, N, D = x.shape
        H = self.num_heads
        d_r = self.router_dim

        # 路由 Q/K 投影（独立于 QSA 的 Q/K/V）
        rQ = self.router_Q(x).view(B, N, H, d_r).transpose(1, 2)  # (B, H, N, d_r)
        rK = self.router_K(x).view(B, N, H, d_r).transpose(1, 2)  # (B, H, N, d_r)

        # 归一化路由向量（确保干涉强度有界）
        rQ = normalize_quantum_state(rQ, dim=-1)
        rK = normalize_quantum_state(rK, dim=-1)

        # 计算干涉强度矩阵
        inter = pairwise_interference(rQ, rK, normalize=True)  # (B, H, N, N) 实数

        # 可选：相位相干性调制
        if self.use_phase_routing:
            # 计算每个位置的局部相位相干性
            coherence = phase_coherence(x, dim=-1)  # (B, N) 实数 [0, 1]
            # 相干性高的 token 允许更多路由（相干性低的 token 被更多抑制）
            coherence_weight = self.phase_router(
                coherence.unsqueeze(-1).float()
            ).squeeze(-1)  # (B, N)
            coherence_weight = torch.sigmoid(coherence_weight)  # (B, N) → [0, 1]
            # 广播到 (B, H, N, N)，以 key 的相干性权重为主
            inter = inter * coherence_weight.unsqueeze(1).unsqueeze(2)

        # 生成路由权重（训练时软路由，推理时硬路由）
        hard = not training
        routing_weights, metrics = self.gate(inter, hard=hard)

        return routing_weights, metrics

    def apply_to_attention(
        self,
        attn_probs: torch.Tensor,
        routing_weights: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """将路由权重应用到 QSA 的注意力概率上。

        融合策略（乘法融合）：
            attn_final[b, h, i, j] = attn_probs[b, h, i, j] * routing_weights[b, h, i, j]

        重新归一化确保概率和为 1。

        Args:
            attn_probs: QSA 计算的 Born 概率 (B, H, N, N) 实数
            routing_weights: QIR 路由权重 (B, H, N, N) 实数 [0, 1]
            eps: 归一化 epsilon
        Returns:
            融合后的注意力概率 (B, H, N, N)，重新归一化
        """
        # 乘法融合
        fused = attn_probs * routing_weights

        # 重新归一化（避免所有权重为 0 的退化情况）
        fused_sum = fused.sum(dim=-1, keepdim=True).clamp(min=eps)
        return fused / fused_sum

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, num_heads={self.num_heads}, "
            f"router_dim={self.router_dim}, "
            f"phase_routing={self.use_phase_routing}"
        )


# ──────────────────────────────────────────────
# 轻量级干涉路由（无投影，直接使用输入）
# ──────────────────────────────────────────────


class LightweightInterferenceRouter(nn.Module):
    """轻量级量子干涉路由器（无额外参数投影）。

    直接使用 QSA 传入的 Q/K 计算干涉，无需额外投影层。
    适合参数敏感场景或需要与 QSA Q/K 共享干涉信号的情况。

    计算量：O(B·H·N²)，与 QSA full attention 相同量级。
    但通过稀疏化后续的 V 聚合，总体减少约 (1 - active_ratio) 的计算量。

    Args:
        tau_constructive: 相长干涉阈值
        tau_destructive: 相消干涉阈值
        temperature: 软路由温度
    """

    def __init__(
        self,
        tau_constructive: float = 0.05,
        tau_destructive: float = -0.05,
        temperature: float = 2.0,
    ):
        super().__init__()
        self.gate = SparseInterferenceGate(
            tau_constructive=tau_constructive,
            tau_destructive=tau_destructive,
            temperature=temperature,
            learnable_tau=True,
        )

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        training: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            Q: QSA 的查询张量 (B, H, N, d_h) 复数
            K: QSA 的键张量 (B, H, N, d_h) 复数
            training: 是否训练模式
        Returns:
            (routing_weights, metrics)
        """
        inter = pairwise_interference(Q, K, normalize=True)  # (B, H, N, N)
        hard = not training
        return self.gate(inter, hard=hard)
