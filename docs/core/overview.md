# 核心概念概述

QuantumArch 实现了 7 大核心机制，共同构成受量子力学启发的神经网络架构。

## 1. QSA - 量子叠加注意力

**Quantum Superposition Attention**

核心思想：将查询和键视为量子态，通过复数内积计算注意力权重。

```python
# 核心公式
α_ij = ⟨q_i | k_j⟩  # 复数内积
β_ij = α_ij · exp(i · f_φ(|α_ij|))  # 干涉调制
p_ij = |β_ij|²  # Born 概率
```

**复杂度**：O(n·d·log n) vs Transformer O(n²·d)

## 2. QEL - 量子纠缠层

**Quantum Entanglement Layer**

用张量积替代传统残差连接，实现跨 token 的全局信息交互。

```python
# 局部纠缠 + 全局纠缠
z_out = U_local(z) ⊗ U_global(z)
```

## 3. QCI - 量子坍缩推理

**Quantum Collapse Inference**

模拟量子测量坍缩，实现自适应早退。

```python
# 熵阈值判断
if entropy < tau_low:  # 低熵 → 信息集中 → 早退
    early_exit()
elif entropy > tau_high:  # 高熵 → 继续计算
    continue
else:  # 中等熵 → 继续
    pass
```

## 4. QIR - 量子干涉路由

**Quantum Interference Routing**

通过相长/相消干涉筛选重要 token。

## 5. QGD - 量子梯度下降

**Quantum Gradient Descent**

使用 Wirtinger 导数，同时优化模长和相位。

```python
# 分离优化
grad_mod = ∂L/∂|z|  # 模长梯度
grad_phase = ∂L/∂θ  # 相位梯度
```

## 6. CR - 复数酉表示

**Complex Unitary Representation**

使用 Cayley 参数化保证酉矩阵性质。

```python
# Cayley 变换
W = (I + iΩ/2)⁻¹(I - iΩ/2)
```

## 7. UP - 不确定性传播

**Uncertainty Propagation**

每层输出携带熵度量，用于动态阈值调整。

## 机制协作流程

```
Input → Embedding → QSA → QEL → QCI → QIR → FFN → UP → Output
                      ↓           ↓
                  干涉路由     早退判断
```

各机制相互配合：
- QSA 提供高效注意力
- QEL 实现全局信息交互
- QCI 自适应计算资源
- CR 保证数值稳定性
- UP 量化不确定性
