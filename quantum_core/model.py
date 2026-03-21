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
        qsa_mode: str = 'topk',
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
        self.use_embedding = (vocab_size > 0 and not direct_input)
        if self.use_embedding:
            self.embedding = ComplexEmbedding(vocab_size, dim, normalize=True)
            self.pos_encoding = LearnedPositionalEncoding(dim, max_seq_len, dropout)

        # 复数输入投影（直接复数输入模式）
        if direct_input:
            self.input_proj = nn.Linear(dim, dim, bias=False).to(torch.complex64)

        # 量子块堆叠
        self.blocks = nn.ModuleList([
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
        ])

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
            if 'token_ids' in x:
                z = self.embedding(x['token_ids'])
                z = self.pos_encoding(z, training=training)
            elif 'inputs' in x:
                raw = x['inputs']
                if raw.dtype != torch.complex64 and raw.dtype != torch.complex128:
                    # 实数输入 -> 复数投影
                    z = torch.complex(raw, torch.zeros_like(raw))
                else:
                    z = raw
                if hasattr(self, 'input_proj'):
                    z = self.input_proj(z)
            else:
                raise ValueError(f"Unexpected keys in input dict: {list(x.keys())}")
        else:
            raw = x
            if raw.dtype != torch.complex64 and raw.dtype != torch.complex128:
                z = torch.complex(raw, torch.zeros_like(raw))
            else:
                z = raw
            if hasattr(self, 'input_proj'):
                z = self.input_proj(z)

        # ── 量子块堆叠 ──
        early_exit = False
        for i, block in enumerate(self.blocks):
            start_time = time.time()

            z, layer_metrics = block(z, training=training)

            qsa_time = time.time() - start_time
            qsa_total_time += qsa_time
            all_metrics[f'block_{i}'] = layer_metrics

            # 检查是否应该早退
            if self.collapse_enabled and training:
                collapse_key = f'qci_collapse_early_exit_rate'
                if collapse_key in layer_metrics:
                    if layer_metrics[collapse_key] > 0.95:
                        early_exit = True
                        break

        # ── 最终归一化 ──
        z = self.final_norm(z)

        # ── 输出投影（复数 -> 实数）──
        # 取模长作为最终实数表示
        output = self.output_proj(z.real)  # 使用实部投影，避免复数损失

        # 计算全局熵
        with torch.no_grad():
            entropy = von_neumann_entropy(
                born_normalize(z, dim=-1), dim=-1
            ).mean().item()

        return {
            'output': output,
            'qsa_time': qsa_total_time,
            'qci_early_exit': early_exit,
            'entropy': entropy,
            'hidden_state': z,  # 复数隐状态（可选使用）
            'layer_metrics': all_metrics,
        }

    def update_parameters(self, **kwargs):
        """更新可调参数（兼容 MockQuantumArch 接口）。"""
        if 'qsa_topk_ratio' in kwargs:
            self.qsa_topk_ratio = kwargs['qsa_topk_ratio']
            for block in self.blocks:
                block.update_parameters(qsa_topk_ratio=kwargs['qsa_topk_ratio'])

        if 'qci_tau_low' in kwargs:
            self.qci_tau_low = kwargs['qci_tau_low']
            tau_high = kwargs.get('qci_tau_high', self.qci_tau_high)
            self.qci_tau_high = tau_high
            for block in self.blocks:
                block.update_parameters(
                    qci_tau_low=kwargs['qci_tau_low'],
                    qci_tau_high=tau_high,
                )

    def get_unitarity_report(self) -> Dict[str, float]:
        """检查所有酉矩阵的酉性约束。"""
        report = {}
        for i, block in enumerate(self.blocks):
            # 检查 QSA 中的 Wq, Wk, Wv, Wo（仅方阵）
            for name in ['Wq', 'Wk', 'Wv', 'Wo']:
                layer = getattr(block.qsa, name, None)
                if hasattr(layer, 'is_square') and layer.is_square:
                    key = f'block{i}_{name}'
                    report[key] = layer.get_unitarity_violation().item()

            # 检查 FFN 中的 W_down（仅方阵）
            layer = getattr(block.ffn_q, 'W_down', None)
            if hasattr(layer, 'is_square') and layer.is_square:
                report[f'block{i}_ffn_down'] = layer.get_unitarity_violation().item()

        return report

    def extra_repr(self) -> str:
        return (
            f'dim={self.dim}, layers={self.num_layers}, '
            f'collapse={self.collapse_enabled}, '
            f'output_dim={self.output_dim}'
        )
