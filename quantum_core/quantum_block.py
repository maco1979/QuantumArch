"""
QuantumBlock 量子块

量子架构的基本构建单元，组装 QSA + QEL + FFN_Q + QCI：

    Input → PreNorm → QSA → QEL → PreNorm → FFN_Q → [可选 QCI] → Output
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from .attention import QuantumSuperpositionAttention
from .entanglement import QuantumEntanglementLayer
from .collapse import QuantumCollapseInference
from .ffn import QuantumFFN
from .normalization import ComplexLayerNorm


class QuantumBlock(nn.Module):
    """量子块 (QB) —— 量子架构的核心构建单元。

    结构：
        x → norm1 → QSA(+QIR) → QEL → residual
          → norm2 → FFN_Q → residual
          → [可选 QCI 坍缩]

    Args:
        dim: 特征维度
        num_heads: 注意力头数
        ffn_dim: FFN 中间维度
        topk_ratio: QSA Top-K 筛选比例
        collapse_enabled: 是否启用 QCI 坍缩推理
        tau_low: QCI 低阈值
        tau_high: QCI 高阈值
        dropout: dropout 概率
        qsa_mode: QSA 模式 ('topk' 或 'full')
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        ffn_dim: Optional[int] = None,
        topk_ratio: float = 0.1,
        collapse_enabled: bool = True,
        tau_low: float = 0.5,
        tau_high: float = 1.5,
        dropout: float = 0.0,
        qsa_mode: str = "topk",
    ):
        super().__init__()
        self.dim = dim
        self.collapse_enabled = collapse_enabled

        # Pre-Norm
        self.norm1 = ComplexLayerNorm(dim)
        self.norm2 = ComplexLayerNorm(dim)

        # QSA 量子叠加注意力
        self.qsa = QuantumSuperpositionAttention(
            dim=dim,
            num_heads=num_heads,
            topk_ratio=topk_ratio,
            dropout=dropout,
            mode=qsa_mode,
        )

        # QEL 量子纠缠层
        self.qel = QuantumEntanglementLayer(dim=dim)

        # FFN_Q 量子前馈网络
        self.ffn_q = QuantumFFN(
            dim=dim,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )

        # QCI 量子坍缩推理（可选）
        self.qci: Optional[QuantumCollapseInference] = None
        if collapse_enabled:
            self.qci = QuantumCollapseInference(
                dim=dim,
                tau_low=tau_low,
                tau_high=tau_high,
            )

    def forward(
        self,
        x: torch.Tensor,
        training: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            x: 复数输入 (batch, seq_len, dim)
            training: 是否训练模式
        Returns:
            (output, layer_metrics)
        """
        metrics = {}

        # ── SubLayer 1: QSA + QEL ──
        residual = x

        # Pre-Norm
        x_norm = self.norm1(x)

        # QSA 量子叠加注意力
        x_attn, qsa_metrics = self.qsa(x_norm, training=training)
        metrics.update({f"qsa_{k}": v for k, v in qsa_metrics.items()})

        # QEL 量子纠缠层（替代残差连接的非线性融合）
        x_entangled, qel_metrics = self.qel(x_attn, training=training)
        metrics.update({f"qel_{k}": v for k, v in qel_metrics.items()})

        # 残差连接
        x = residual + x_entangled

        # ── SubLayer 2: FFN_Q ──
        x = self.ffn_q(x, training=training)

        # ── 可选: QCI 坍缩推理 ──
        if self.qci is not None and training:
            x, collapse_metrics = self.qci(x, training=training)
            metrics.update({f"qci_{k}": v for k, v in collapse_metrics.items()})

        return x, metrics

    def update_parameters(self, **kwargs):
        """更新可调参数（由优化系统调用）。"""
        if "qsa_topk_ratio" in kwargs:
            self.qsa.topk_ratio = kwargs["qsa_topk_ratio"]
        if "qci_tau_low" in kwargs and self.qci is not None:
            # 获取当前 tau_high（可能在 threshold 子模块或直接属性上）
            if self.qci.threshold is not None:
                current_tau_high = self.qci.threshold.tau_high.item()
            else:
                current_tau_high = self.qci.tau_high.item()
            self.qci.update_thresholds(
                tau_low=kwargs["qci_tau_low"],
                tau_high=kwargs.get("qci_tau_high", current_tau_high),
            )

    def get_complexity_report(self, seq_len: int = 512) -> Dict[str, str]:
        """返回各子模块的计算复杂度报告。

        基于给定序列长度 N 和模型维度 d 估计操作数，
        帮助定位瓶颈并指导优化方向。

        Args:
            seq_len: 假定的序列长度 N（用于估算注意力复杂度）
        Returns:
            dict: 子模块名 → 渐进复杂度描述
        """
        N, D, H = seq_len, self.dim, self.qsa.num_heads
        k = max(1, int(self.qsa.topk_ratio * N))
        ffn_d = self.ffn_q.ffn_dim

        return {
            "QSA（复数投影 Q/K/V）": f"O(N·D²) ≈ {N * D * D // 1000}K ops",
            "QSA（内积矩阵）": (
                f"O(N·k·D) ≈ {N * k * D // 1000}K ops  [topk={k}/{N}]"
                if self.qsa.mode == "topk"
                else f"O(N²·D) ≈ {N * N * D // 1000}K ops  [full]"
            ),
            "QEL（局部纠缠）": f"O(N·D²) ≈ {N * D * D // 1000}K ops",
            "QEL（QFT）": f"O(N·D·log N) ≈ {int(N * D * (N.bit_length())) // 1000}K ops",
            "FFN_Q": f"O(N·D·ffn_d) ≈ {N * D * ffn_d // 1000}K ops  [ffn_dim={ffn_d}]",
            "QCI（POVM）": f"O(N·D·collapse_dim)" if self.collapse_enabled else "disabled",
            "总计（估算）": f"≈ {(N*D*D*4 + N*k*D + N*D*ffn_d) // 1_000_000}M ops / block",
        }

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, collapse={self.collapse_enabled}, "
            f"heads={self.qsa.num_heads}, topk={self.qsa.topk_ratio}"
        )
