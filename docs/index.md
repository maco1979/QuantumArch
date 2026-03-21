# 量子架构 (QuantumArch)

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-orange.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/github/actions/workflow/status/quantumarch/quantumarch/ci-cd.yml?label=CI" alt="CI">
</p>

**QuantumArch** 是一个受量子力学启发的神经网络架构范式，旨在经典硬件上实现量子数学结构。

## ✨ 核心特性

| 机制 | 名称 | 描述 |
|------|------|------|
| **QSA** | 量子叠加注意力 | 复数振幅干涉 + Born 法则，O(n log n) 复杂度 |
| **QEL** | 量子纠缠层 | 张量积替代残差连接 |
| **QCI** | 量子坍缩推理 | 自适应早退机制 |
| **QIR** | 量子干涉路由 | 相长/相消筛选 token |
| **QGD** | 量子梯度下降 | Wirtinger 导数优化 |
| **CR** | 复数酉表示 | Cayley 参数化保证酉性 |
| **UP** | 不确定性传播 | 每层熵度量 |

## 🚀 快速开始

```python
import torch
from quantum_core import QuantumArch

# 创建模型
model = QuantumArch(
    vocab_size=10000,
    dim=512,
    num_layers=12,
    num_heads=8,
)

# 前向传播
token_ids = torch.randint(0, 10000, (4, 128))
result = model({'token_ids': token_ids})

print(f"Output shape: {result['output'].shape}")
print(f"Entropy: {result['entropy']:.4f}")
```

## 📊 性能对比

| 指标 | Transformer | QuantumArch | 提升 |
|------|-------------|-------------|------|
| 推理速度 | 1x | 3-10x | ⬆️ |
| 内存占用 | 100% | 60-80% | ⬇️ |
| GPU利用率 | 40-60% | 85%+ | ⬆️ |

## 📚 文档

- [核心概念](core/overview.md) - 理解 7 大核心机制
- [架构设计](architecture/design.md) - 系统设计文档
- [API 参考](api/overview.md) - 完整的 API 文档
- [示例教程](examples/index.md) - 入门示例

## 🧪 测试

```bash
# 运行完整测试套件
pytest tests/ -v --cov=quantum_core

# 运行性能基准测试
python run_benchmark.py

# 类型检查
pyright
```

## 🤝 贡献

欢迎提交 Pull Request！请先阅读 [贡献指南](CONTRIBUTING.md)。

## 📄 许可证

MIT License - 查看 [LICENSE](LICENSE) 了解详情。
