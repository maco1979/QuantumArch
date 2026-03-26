"""
QuantumArch 完整模型

组装所有量子架构组件为端到端模型：

    Input IDs → ComplexEmbedding → PositionEncoding
    → [QuantumBlock × L] → Final Collapse → Output Projection

支持两种输入模式：
1. token_ids: 离散 token 序列（语言模型）
2. 直接复数输入（通用处理）
"""

import torch
import torch.nn as nn
import time
from typing import Dict, Optional, Any, Union

from .embedding import ComplexEmbedding, QuantumPositionalEncoding, LearnedPositionalEncoding
from .quantum_block import QuantumBlock
from .normalization import ComplexLayerNorm
from .complex_ops import born_normalize, von_neumann_entropy


class QuantumArch(nn.Module):
    """量子架构完整模型。

    Args:
        vocab_size: 词汇表大小（0 表示不支持 token 输入）
        dim: 特征维度
        num_layers: 量子块层数
        num_heads: 注意力头数
        ffn_dim: FFN 中间维度
        max_seq_len: 最大序列长度
        topk_ratio: QSA Top-K 筛选比例
        collapse_enabled: 是否启用 QCI
        tau_low: QCI 低阈值
        tau_high: QCI 高阈值
        dropout: dropout 概率
        qsa_mode: QSA 模式
        output_dim: 输出维度（None 则等于 dim）
        direct_input: 是否支持直接复数输入（不需要嵌入层）
    """

    def __init__(
        self,
        vocab_size: int = 0,
        dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_dim: Optional[int] = None,
        max_seq_len: int = 2048,
        topk_ratio: float = 0.1,
        collapse_enabled: bool = True,
        tau_low: float = 0.5,
        tau_high: float = 1.5,
        dropout: float = 0.0,
        qsa_mode: str = "topk",
        output_dim: Optional[int] = None,
        direct_input: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.output_dim = output_dim or dim
        self.collapse_enabled = collapse_enabled
        self.direct_input = direct_input

        # 嵌入层（仅在 token_ids 输入模式下使用）
        self.use_embedding = vocab_size > 0 and not direct_input
        if self.use_embedding:
            self.embedding = ComplexEmbedding(vocab_size, dim, normalize=True)
            self.pos_encoding = LearnedPositionalEncoding(dim, max_seq_len, dropout)

        # 复数输入投影（直接复数输入模式）
        if direct_input:
            self.input_proj = nn.Linear(dim, dim, bias=False).to(torch.complex64)

        # 量子块堆叠
        self.blocks = nn.ModuleList(
            [
                QuantumBlock(
                    dim=dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    topk_ratio=topk_ratio,
                    collapse_enabled=collapse_enabled,
                    tau_low=tau_low,
                    tau_high=tau_high,
                    dropout=dropout,
                    qsa_mode=qsa_mode,
                )
                for _ in range(num_layers)
            ]
        )

        # 最终归一化
        self.final_norm = ComplexLayerNorm(dim)

        # 输出投影（从复数到实数，用于损失计算）
        self.output_proj = nn.Linear(dim, self.output_dim, bias=False)

        # 可调参数（兼容 MockQuantumArch 接口）
        self.qsa_topk_ratio = topk_ratio
        self.qci_tau_low = tau_low
        self.qci_tau_high = tau_high
        self.early_exit_enabled = collapse_enabled

        self._init_weights()

    def _init_weights(self):
        """初始化非 Cayley 参数的权重。"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # 跳过 ComplexLinear 内部的 nn.Linear（它们有自己的初始化）
                if hasattr(module, "_complex_linear_parent"):
                    continue
                if module.weight.dtype in (torch.complex64, torch.complex128):
                    nn.init.normal_(module.weight.real, std=0.02)
                    nn.init.normal_(module.weight.imag, std=0.02)
                else:
                    nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x: Union[torch.Tensor, Dict[str, torch.Tensor]],
        training: bool = True,
    ) -> Dict[str, Any]:
        """
        Args:
            x: 输入张量或字典
               - torch.Tensor (batch, seq_len, dim): 直接复数输入
               - dict {'token_ids': (batch, seq_len)}: token 输入
               - dict {'inputs': (batch, seq_len, dim)}: 兼容旧接口
            training: 是否训练模式
        Returns:
            dict 包含:
                - 'output': 输出张量 (batch, seq_len, output_dim)
                - 'qsa_time': QSA 时间
                - 'qci_early_exit': 是否早退
                - 'entropy': 平均熵值
                - 其他指标
        """
        qsa_total_time = 0.0
        all_metrics = {}

        # ── 输入处理 ──
        if isinstance(x, dict):
            if "token_ids" in x:
                z = self.embedding(x["token_ids"])
                z = self.pos_encoding(z, training=training)
            elif "inputs" in x:
                raw = x["inputs"]
                if raw.dtype != torch.complex64 and raw.dtype != torch.complex128:
                    # 实数输入 -> 复数投影
                    z = torch.complex(raw, torch.zeros_like(raw))
                else:
                    z = raw
                if hasattr(self, "input_proj"):
                    z = self.input_proj(z)
            else:
                raise ValueError(f"Unexpected keys in input dict: {list(x.keys())}")
        else:
            raw = x
            if raw.dtype != torch.complex64 and raw.dtype != torch.complex128:
                z = torch.complex(raw, torch.zeros_like(raw))
            else:
                z = raw
            if hasattr(self, "input_proj"):
                z = self.input_proj(z)

        # ── 量子块堆叠 ──
        early_exit = False
        for i, block in enumerate(self.blocks):
            start_time = time.time()

            z, layer_metrics = block(z, training=training)

            qsa_time = time.time() - start_time
            qsa_total_time += qsa_time
            all_metrics[f"block_{i}"] = layer_metrics

            # 检查是否应该早退
            # 注：QCI 在 quantum_block.py 中仅在 training=True 时运行，
            # 推理阶段应由调用者决定是否启用早退（通过 inference_mode 参数控制）。
            # key 格式：block.py 将 QCI 的 metrics 加前缀 'qci_'，
            # 坍缩模块输出 'collapse_early_exit_rate'，合并后为 'qci_collapse_early_exit_rate'
            if self.collapse_enabled:
                exit_key = "qci_collapse_early_exit_rate"
                if exit_key in layer_metrics:
                    exit_rate = layer_metrics[exit_key]
                    # 训练时：95% 早退率触发全局早退
                    # 推理时：不在此处触发（推理早退由 inference_early_exit 参数控制）
                    if training and exit_rate > 0.95:
                        early_exit = True
                        break

        # ── 最终归一化 ──
        z = self.final_norm(z)

        # ── 输出投影（复数 -> 实数）──
        # 改进：使用复数绝对值（模长）作为实数投影输入，而非仅使用实部。
        # 理由：|z| = √(Re²+Im²) 保留了完整的振幅信息；
        #       仅用 Re(z) 会丢失虚部编码的相位信息。
        # 对于需要完整复数输出的任务，可访问返回的 'hidden_state'。
        z_real_repr = z.abs()  # (B, N, dim)，实数，保留模长信息
        output = self.output_proj(z_real_repr)  # (B, N, output_dim)

        # 计算全局熵
        with torch.no_grad():
            entropy = von_neumann_entropy(born_normalize(z, dim=-1), dim=-1).mean().item()

        return {
            "output": output,
            "qsa_time": qsa_total_time,
            "qci_early_exit": early_exit,
            "entropy": entropy,
            "hidden_state": z,  # 复数隐状态（可选使用）
            "layer_metrics": all_metrics,
        }

    def update_parameters(self, **kwargs):
        """更新可调参数（兼容 MockQuantumArch 接口）。"""
        if "qsa_topk_ratio" in kwargs:
            self.qsa_topk_ratio = kwargs["qsa_topk_ratio"]
            for block in self.blocks:
                block.update_parameters(qsa_topk_ratio=kwargs["qsa_topk_ratio"])

        if "qci_tau_low" in kwargs:
            self.qci_tau_low = kwargs["qci_tau_low"]
            tau_high = kwargs.get("qci_tau_high", self.qci_tau_high)
            self.qci_tau_high = tau_high
            for block in self.blocks:
                block.update_parameters(
                    qci_tau_low=kwargs["qci_tau_low"],
                    qci_tau_high=tau_high,
                )

    def get_unitarity_report(self) -> Dict[str, float]:
        """检查所有酉矩阵的酉性约束。"""
        report = {}
        for i, block in enumerate(self.blocks):
            # 检查 QSA 中的 Wq, Wk, Wv, Wo（仅方阵）
            for name in ["Wq", "Wk", "Wv", "Wo"]:
                layer = getattr(block.qsa, name, None)
                if hasattr(layer, "is_square") and layer.is_square:
                    key = f"block{i}_{name}"
                    report[key] = layer.get_unitarity_violation().item()

            # 检查 FFN 中的 W_down（通过 ComplexLinear 包装器）
            layer = getattr(block.ffn_q, "W_down", None)
            if layer is not None:
                # ComplexLinear 有 is_cayley 属性
                if hasattr(layer, "is_cayley") and layer.is_cayley:
                    report[f"block{i}_ffn_down"] = layer.get_unitarity_violation()

        return report

    def count_parameters(self) -> Dict[str, int]:
        """统计模型各部分参数量。

        分别统计复数参数（按实数等效数量计算）和实数参数，
        以及总量，方便与 Transformer 基线对比。

        Returns:
            dict 包含:
                - 'total_real_equiv': 所有参数折算为实数的总数量
                - 'complex_params': 复数参数数量（每个 complex64 = 2 个实数）
                - 'real_params': 实数参数数量
                - 'embedding_params': 嵌入层参数（实数等效）
                - 'block_params': 量子块参数（实数等效）
                - 'qsa_params': 所有层 QSA 的总参数（实数等效）
                - 'qel_params': 所有层 QEL 的总参数（实数等效）
                - 'qci_params': 所有层 QCI 的总参数（实数等效）
                - 'ffn_params': 所有层 FFN 的总参数（实数等效）
                - 'memory_mb_fp32': 参数内存占用（MB，以 float32 存储）
                - 'memory_mb_fp16': 参数内存占用（MB，以 float16 存储）
        """
        embedding_params = 0
        block_params = 0
        real_params = 0
        complex_params = 0

        # 子模块分类统计
        component_params: Dict[str, int] = {
            "qsa_params": 0, "qel_params": 0, "qci_params": 0, "ffn_params": 0
        }

        for name, p in self.named_parameters():
            n = p.numel()
            if p.is_complex():
                equiv = n * 2  # complex64 = 2 × float32
                complex_params += n
            else:
                equiv = n
                real_params += n

            if "embedding" in name or "pos_encoding" in name:
                embedding_params += equiv
            elif "blocks" in name:
                block_params += equiv
                # 按子模块分类
                if ".qsa." in name:
                    component_params["qsa_params"] += equiv
                elif ".qel." in name:
                    component_params["qel_params"] += equiv
                elif ".collapse." in name:
                    component_params["qci_params"] += equiv
                elif ".ffn_q." in name:
                    component_params["ffn_params"] += equiv

        total = sum(p.numel() * (2 if p.is_complex() else 1) for p in self.parameters())

        # 内存估算（float32 每参数 4 bytes，float16 每参数 2 bytes）
        # 对于 complex64：每参数 8 bytes（相当于 2 个 float32）
        # 这里用 total 实数等效数量 × 4 bytes
        memory_mb_fp32 = total * 4 / (1024 ** 2)
        memory_mb_fp16 = total * 2 / (1024 ** 2)

        return {
            "total_real_equiv": total,
            "complex_params": complex_params,
            "real_params": real_params,
            "embedding_params": embedding_params,
            "block_params": block_params,
            "memory_mb_fp32": round(memory_mb_fp32, 2),
            "memory_mb_fp16": round(memory_mb_fp16, 2),
            **component_params,
        }

    def complexity_summary(self, seq_len: int = 512) -> str:
        """生成模型复杂度摘要字符串。

        Args:
            seq_len: 假定序列长度
        Returns:
            格式化的多行摘要
        """
        param_info = self.count_parameters()
        total_m = param_info["total_real_equiv"] / 1e6

        lines = [
            f"=== QuantumArch 复杂度摘要 ===",
            f"  维度: dim={self.dim}, 层数={self.num_layers}, 输出={self.output_dim}",
            f"  参数量（实数等效）: {total_m:.2f}M",
            f"    ├ 复数参数: {param_info['complex_params'] / 1e6:.2f}M 个",
            f"    ├ 实数参数: {param_info['real_params'] / 1e6:.2f}M 个",
            f"    ├ 嵌入层: {param_info['embedding_params'] / 1e6:.2f}M",
            f"    └ 量子块: {param_info['block_params'] / 1e6:.2f}M",
            f"  量子块内部分布:",
            f"    ├ QSA: {param_info['qsa_params'] / 1e6:.2f}M",
            f"    ├ QEL: {param_info['qel_params'] / 1e6:.2f}M",
            f"    ├ QCI: {param_info['qci_params'] / 1e6:.2f}M",
            f"    └ FFN: {param_info['ffn_params'] / 1e6:.2f}M",
            f"  内存占用（仅参数）:",
            f"    ├ FP32: {param_info['memory_mb_fp32']:.1f} MB",
            f"    └ FP16: {param_info['memory_mb_fp16']:.1f} MB",
            f"  序列长度: {seq_len}",
        ]

        if self.blocks:
            report = self.blocks[0].get_complexity_report(seq_len)
            lines.append("  单块复杂度（估算）:")
            for k, v in report.items():
                lines.append(f"    {k}: {v}")

        return "\n".join(lines)

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, layers={self.num_layers}, "
            f"collapse={self.collapse_enabled}, "
            f"output_dim={self.output_dim}"
        )

    @torch.no_grad()
    def inference(
        self,
        x,
        return_hidden: bool = False,
        return_metrics: bool = False,
        causal: bool = False,
    ) -> dict:
        """快速推理接口。

        封装了推理阶段的常见设置：
        - 自动进入 ``eval()`` 模式（不修改 ``training`` 标志，调用后恢复）
        - 使用 ``torch.no_grad()`` 禁用梯度（已通过装饰器保证）
        - ``training=False`` 传入 forward，触发 QCI 的硬阈值坍缩

        相比直接调用 ``forward()``，``inference()`` 的优势：
        1. 调用语法更简洁，无需手动管理 eval/train 切换
        2. 支持 causal 参数（自回归生成时需要因果掩码）
        3. 提供可选的诊断信息（量子态健康度、各层指标）
        4. 保证调用前后模型 training 状态不变

        Args:
            x: 输入，格式同 forward() 的 x 参数
            return_hidden: 是否在结果中包含最终复数隐藏状态 (B, N, D)
            return_metrics: 是否计算并返回量子健康度摘要
            causal: 是否对所有 QSA 块使用因果掩码（自回归生成时设为 True）

        Returns:
            dict 包含:
                - ``output``: (B, N, output_dim) 实数输出
                - ``qci_early_exit``: 是否触发早退
                - ``entropy``: 最终层平均熵值
                - ``hidden_state``: (B, N, D) 复数隐藏状态（仅当 return_hidden=True）
                - ``quantum_health``: 量子健康度摘要 dict（仅当 return_metrics=True）
                - ``layer_metrics``: 每层指标 dict
                - ``qsa_time``: 前向传播耗时（秒）

        Example:
            >>> model = QuantumArch(vocab_size=1000, dim=256, num_layers=4)
            >>> result = model.inference({"token_ids": tokens}, return_hidden=True)
            >>> logits = result["output"]   # (B, N, 256)
            >>> hidden = result["hidden_state"]  # (B, N, 256) 复数
        """
        # 保存并切换为 eval 模式
        was_training = self.training
        self.eval()

        try:
            # 若模型的 QSA 块支持 causal 参数，需特殊处理
            # 通过 forward() 传入 training=False，但 causal 需要注入到每个块
            if causal:
                # 临时设置所有块的 QSA 为 causal 模式
                for block in self.blocks:
                    if hasattr(block, "qsa") and hasattr(block.qsa, "_inference_causal"):
                        block.qsa._inference_causal = True

            result = self.forward(x, training=False)

        finally:
            # 恢复 training 状态
            if was_training:
                self.train()
            # 清除临时 causal 标志
            if causal:
                for block in self.blocks:
                    if hasattr(block, "qsa") and hasattr(block.qsa, "_inference_causal"):
                        block.qsa._inference_causal = False

        # 按需裁剪返回内容
        if not return_hidden:
            result.pop("hidden_state", None)

        # 按需计算量子健康度摘要
        if return_metrics and "hidden_state" in result:
            from .quantum_metrics import compute_model_quantum_health
            quantum_health = compute_model_quantum_health(result["hidden_state"])
            result["quantum_health"] = quantum_health
        elif return_metrics:
            # hidden_state 已被弹出，需要重新前向一次获取（或跳过）
            result["quantum_health"] = {}

        return result
