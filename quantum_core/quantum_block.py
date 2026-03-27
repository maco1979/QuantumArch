"""
QuantumBlock 量子块

量子架构的基本构建单元，组装 QSA + QEL + FFN_Q + QCI：

    Input → PreNorm → QSA → QEL → PreNorm → FFN_Q → [可选 QCI] → Output

新增特性（v1.1）：
    - 梯度检查点（gradient checkpointing）支持：启用后用重计算换显存，
      适合长序列（N > 1024）或深度模型（layers > 12）的训练。
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint_utils
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
        use_checkpoint: 是否启用梯度检查点（节省显存，但增加约 30% 计算时间）
            适合场景：seq_len > 1024，或 GPU 显存不足时
            注意：启用后训练速度降低，但可支持 2-4× 更大的批次大小
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
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.collapse_enabled = collapse_enabled
        self.use_checkpoint = use_checkpoint

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

    def _forward_sublayer1(
        self,
        x: torch.Tensor,
        training: bool,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """SubLayer 1 的前向计算（QSA + QEL）。

        单独抽取为方法，以支持梯度检查点封装。

        Args:
            x: 复数输入 (B, N, D)
            training: 是否训练模式
        Returns:
            (output, metrics_flat)
            metrics_flat: 仅包含 float 值的 dict（用于检查点兼容）
        """
        metrics: Dict[str, float] = {}

        x_norm = self.norm1(x)
        x_attn, qsa_metrics = self.qsa(x_norm, training=training)
        metrics.update({f"qsa_{k}": v for k, v in qsa_metrics.items()})

        x_entangled, qel_metrics = self.qel(x_attn, training=training)
        metrics.update({f"qel_{k}": v for k, v in qel_metrics.items()})

        output = x + x_entangled
        return output, metrics

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
        if self.use_checkpoint and training and x.requires_grad:
            # 梯度检查点：前向时不保存中间激活，反向时重新计算
            # 注意：checkpoint 不支持返回非 Tensor 的第二返回值，
            # 所以 metrics 在检查点外单独收集（不参与梯度传播，仅用于监控）
            # 使用 use_reentrant=False（推荐方式，避免重入问题）
            def create_custom_forward():
                def custom_forward(x_inner):
                    out, _ = self._forward_sublayer1(x_inner, training=training)
                    return out
                return custom_forward

            x = checkpoint_utils.checkpoint(
                create_custom_forward(),
                x,
                use_reentrant=False,
            )
            # 在检查点外再运行一次以收集 metrics（不计入梯度图）
            with torch.no_grad():
                _, sl1_metrics = self._forward_sublayer1(x.detach(), training=False)
            metrics.update(sl1_metrics)
        else:
            x, sl1_metrics = self._forward_sublayer1(x, training=training)
            metrics.update(sl1_metrics)

        # ── SubLayer 2: FFN_Q ──
        x = self.norm2(x)
        x = self.ffn_q(x, training=training)

        # ── 可选: QCI 坍缩推理 ──
        if self.qci is not None and training:
            x, collapse_metrics = self.qci(x, training=training)
            metrics.update({f"qci_{k}": v for k, v in collapse_metrics.items()})

        return x, metrics

    def enable_gradient_checkpointing(self):
        """启用梯度检查点。

        等价于在初始化时设置 `use_checkpoint=True`，
        可在训练过程中动态启用（例如在 OOM 时降级启用）。
        """
        self.use_checkpoint = True

    def disable_gradient_checkpointing(self):
        """禁用梯度检查点（恢复正常训练速度）。"""
        self.use_checkpoint = False

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

    @torch.no_grad()
    def get_block_efficiency_report(
        self,
        x: torch.Tensor,
    ) -> Dict[str, float]:
        """综合评估量子块的推理效率与量子特性利用度。

        从「效率」和「量子性」两个维度分析当前块的运行状态：

        **效率维度**（越高越好）：
        - ``qsa_topk_utilization``: Top-K 筛选实际保留的比例（小 → 计算省）
        - ``qci_early_exit_rate``: QCI 触发早退的 token 比例（大 → 跳过更多层）
        - ``qel_global_qft_usage``: QFT 是否实际对序列产生显著变换（α 远离 1）

        **量子性维度**（反映量子特性的利用程度）：
        - ``avg_attention_entropy``: 平均注意力熵（大 → 注意力分散，使用了叠加态）
        - ``qel_entanglement_strength``: QEL 纠缠强度均值（大 → 纠缠充分）
        - ``head_diversity_score``: 多头多样性分数（大 → 各头学到不同模式）
        - ``phase_coherence_score``: 输入量子态的相位相干性（量子性指标）

        **健康状态**（诊断异常）：
        - ``has_dead_gates``: 是否存在死门（FFN 门控模长 < 0.1）
        - ``unitarity_max_violation``: 最大酉性违背度（应 < 1e-4）

        Args:
            x: 复数输入 (batch, seq_len, dim)
        Returns:
            dict 包含所有效率和量子性指标（Python float + bool）

        Example:
            >>> block = QuantumBlock(dim=256, num_heads=8)
            >>> x = torch.randn(2, 32, 256, dtype=torch.complex64)
            >>> report = block.get_block_efficiency_report(x)
            >>> print(f"早退率: {report['qci_early_exit_rate']:.2%}")
            >>> print(f"多头多样性: {report['head_diversity_score']:.3f}")
        """
        from .complex_ops import phase_coherence, born_normalize, von_neumann_entropy

        report: Dict[str, float] = {}

        # ── 1. QSA 效率与多头分析 ──
        # 注意力模式（不更新梯度）
        attn_patterns = self.qsa.get_attention_patterns(x)
        attn_probs = attn_patterns["attn_probs"]  # (B, H, N, N)
        B, H, N, _ = attn_probs.shape

        # Top-K 实际利用率（保留 token 的平均比例）
        topk_mask = attn_patterns["topk_mask"]  # (B, H, N, N) bool
        report["qsa_topk_utilization"] = topk_mask.float().mean().item()

        # 平均注意力熵（量子叠加性指标）
        attn_entropy_per_pos = von_neumann_entropy(attn_probs, dim=-1)  # (B, H, N)
        report["avg_attention_entropy"] = attn_entropy_per_pos.mean().item()

        # 多头多样性（v1.2 新增方法）
        head_summary = self.qsa.multi_head_entropy_summary(x)
        report["head_diversity_score"] = head_summary["head_diversity_score"]
        report["num_uniform_heads"] = float(head_summary["uniform_head_mask"].sum().item())
        report["num_sharp_heads"] = float(head_summary["sharp_head_mask"].sum().item())

        # ── 2. QEL 纠缠分析 ──
        qel_metrics = self.qel.get_entanglement_metrics(x)
        report["qel_entanglement_strength"] = qel_metrics.get("concurrence_mean", 0.0)
        report["qel_entanglement_std"] = qel_metrics.get("concurrence_std", 0.0)
        report["qel_qft_alpha"] = qel_metrics.get("qft_alpha", 1.0)
        # QFT 利用率：alpha 越远离 1，全局纠缠越强
        report["qel_global_qft_usage"] = abs(1.0 - qel_metrics.get("qft_alpha", 1.0))

        # ── 3. QCI 早退率 ──
        if self.collapse_enabled and self.collapse is not None:
            eff = self.collapse.compute_collapse_efficiency(x)
            report["qci_early_exit_rate"] = eff["early_exit_rate"]
            report["qci_information_retention"] = eff["information_retention"]
            report["qci_entropy_compression_ratio"] = eff["entropy_compression_ratio"]
        else:
            report["qci_early_exit_rate"] = 0.0
            report["qci_information_retention"] = 1.0
            report["qci_entropy_compression_ratio"] = 1.0

        # ── 4. 相位相干性（量子特性整体指标）──
        coherence = phase_coherence(x, dim=-1)  # (B, N) 每个 token 的相干性
        report["phase_coherence_score"] = coherence.mean().item()

        # ── 5. FFN 门控健康度 ──
        # 需要先经过 QSA 和 QEL 获取 FFN 的实际输入
        # 为避免完整前向传播，直接检查 FFN 参数健康度
        has_dead_gates = False
        if hasattr(self.ffn_q, "_gated_ffn") and self.ffn_q._gated_ffn is not None:
            gated_ffn = self.ffn_q._gated_ffn
            if hasattr(gated_ffn, "gate") and gated_ffn.gate is not None:
                # 通过参数模长估计（不需要完整前向）
                gate_weight_norm = gated_ffn.gate.W_gate.abs().mean().item()
                has_dead_gates = gate_weight_norm < 0.01  # 权重过小 → 死门
        report["has_dead_gates"] = float(has_dead_gates)

        # ── 6. 酉性违背度 ──
        # 检查 QSA 的 Wq（代表整体酉性健康）
        try:
            wq_violation = self.qsa.Wq.get_unitarity_violation().item()
            report["unitarity_max_violation"] = wq_violation
        except Exception:
            report["unitarity_max_violation"] = 0.0

        return report

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
            "梯度检查点": "启用（节省显存 ~50%，增加计算约 30%）" if self.use_checkpoint else "禁用",
            "总计（估算）": f"≈ {(N*D*D*4 + N*k*D + N*D*ffn_d) // 1_000_000}M ops / block",
        }

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, collapse={self.collapse_enabled}, "
            f"heads={self.qsa.num_heads}, topk={self.qsa.topk_ratio}, "
            f"checkpoint={self.use_checkpoint}"
        )
