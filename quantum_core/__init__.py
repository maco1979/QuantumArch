"""
量子架构核心模块 (QuantumArch Core)

将量子力学的数学结构融入 PyTorch 神经网络，实现六大核心机制。

## 核心组件

- **complex_ops**: 复数基础运算工具（Born概率、冯诺依曼熵、复数归一化）
- **activations**: ModReLU 复数激活函数
- **normalization**: ComplexLayerNorm / ComplexBatchNorm
- **unitary**: Cayley 参数化酉矩阵层
- **embedding**: 复数嵌入 + 量子位置编码
- **attention**: QSA 量子叠加注意力（含 QIR 干涉路由）
- **entanglement**: QEL 量子纠缠层（局部门 + QFT 全局）
- **collapse**: QCI 量子坍缩推理（POVM + 自适应早退）
- **ffn**: FFN_Q 量子前馈网络
- **quantum_block**: QuantumBlock 量子块
- **model**: QuantumArch 完整模型
- **optimizer**: QGD 量子梯度下降优化器
"""

from .complex_ops import (
    complex_to_polar,
    polar_to_complex,
    normalize_quantum_state,
    born_probability,
    born_normalize,
    von_neumann_entropy,
    entropy_from_state,
    complex_inner_product,
    complex_softmax,
    complex_dropout,
    check_unitarity,
)
from .activations import ModReLU, ModReLUV2, CReLU, ComplexGELU
from .normalization import ComplexLayerNorm, ComplexBatchNorm
from .unitary import CayleyLinear, CayleyLinearSimple
from .embedding import ComplexEmbedding, QuantumPositionalEncoding, LearnedPositionalEncoding
from .attention import QuantumSuperpositionAttention, PhaseModulation
from .entanglement import (
    QuantumEntanglementLayer, EntanglementGate, AdaptiveEntanglementGate,
    SchmidtEntanglementGate, SchmidtEntanglementGateV2,
    QuantumFourierTransform, UnitaryCoupling,
    concurrence, entanglement_entropy,
)
from .collapse import QuantumCollapseInference, POVMProjector, AdaptiveThreshold
from .ffn import QuantumFFN, GatedQuantumFFN, QuantumGate, ComplexSigmoid, ComplexLinear, ComplexBias
from .quantum_block import QuantumBlock
from .model import QuantumArch
from .optimizer import QGD

__all__ = [
    # 基础运算
    'complex_to_polar', 'polar_to_complex',
    'normalize_quantum_state', 'born_probability', 'born_normalize',
    'von_neumann_entropy', 'entropy_from_state',
    'complex_inner_product', 'complex_softmax', 'complex_dropout',
    'check_unitarity',
    # 激活函数
    'ModReLU', 'ModReLUV2', 'CReLU', 'ComplexGELU',
    # 归一化
    'ComplexLayerNorm', 'ComplexBatchNorm',
    # 酉矩阵
    'CayleyLinear', 'CayleyLinearSimple',
    # 嵌入
    'ComplexEmbedding', 'QuantumPositionalEncoding', 'LearnedPositionalEncoding',
    # 核心组件
    'QuantumSuperpositionAttention', 'PhaseModulation',
    'QuantumEntanglementLayer', 'EntanglementGate', 'AdaptiveEntanglementGate',
    'SchmidtEntanglementGate', 'SchmidtEntanglementGateV2',
    'QuantumFourierTransform', 'UnitaryCoupling',
    'concurrence', 'entanglement_entropy',
    'QuantumCollapseInference', 'POVMProjector', 'AdaptiveThreshold',
    'QuantumFFN', 'GatedQuantumFFN', 'QuantumGate', 'ComplexSigmoid', 'ComplexLinear', 'ComplexBias', 'QuantumBlock',
    # 完整模型和优化器
    'QuantumArch', 'QGD',
]
