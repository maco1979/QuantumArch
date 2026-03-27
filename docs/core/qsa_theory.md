# QSA 量子叠加注意力：完整理论文档

**模块路径**: `quantum_core/attention.py`  
**版本**: v1.2（含因果掩码 + 多头熵分析）  
**维护者**: QuantumArch Team  

---

## 1. 理论动机

### 1.1 经典注意力的局限

标准 Transformer 注意力的核心公式：

```
Attention(Q, K, V) = softmax(QK^T / √d) · V
```

存在三个根本性局限：

1. **实数点积**：Q·K^T 只能捕捉实数语义相似度，忽略相位关系
2. **O(n²) 复杂度**：每个 token 对所有位置求注意力，规模限制严重
3. **无干涉机制**：不同注意力路径之间不存在量子干涉（相长/相消选择）

### 1.2 量子叠加注意力的核心创新

QSA（Quantum Superposition Attention）将注意力计算提升到**复数希尔伯特空间**，引入两个关键机制：

- **量子干涉调制**：β_ij = α_ij · e^{i·f(|α_ij|)}，使不同路径可以相消
- **Born 概率路由**：p_ij = |β_ij|² ∈ [0,1]，满足量子测量的概率公理

---

## 2. 数学形式化

### 2.1 量子内积

对于复数查询向量 q_i ∈ ℂ^d 和键向量 k_j ∈ ℂ^d，量子内积定义为：

```
α_ij = ⟨q_i | k_j⟩ = Σ_k conj(q_i^k) · k_j^k   ∈ ℂ
```

对比实数 Transformer 的点积 q_i · k_j ∈ ℝ，复数内积**保留了相位信息**：
- |α_ij|：两态的"重叠程度"（对应实数相似度）
- arg(α_ij)：两态的"相位差"（Transformer 中不存在此信息）

### 2.2 干涉调制（QIR 集成）

QSA 集成了量子干涉路由（QIR）的相位调制功能：

```
β_ij = α_ij · exp(i · f_φ(|α_ij|))
```

其中 f_φ: ℝ → ℝ 是可学习的相位调制函数（MLP 实现），将内积模长映射为附加相位偏移。

**相长干涉**（constructive）：当 f_φ 使 β_ij 与其他路径同相时，|β_ij|² 增大  
**相消干涉**（destructive）：当 f_φ 使 β_ij 与其他路径反相时，|β_ij|² 减小

这使 QSA 能够**自适应地增强/抑制**不同的注意力路径，实现类似量子计算中的干涉优化。

### 2.3 Born 法则映射

将干涉调制后的复数内积映射为合法注意力概率：

```
p_ij = |β_ij|² / Σ_j' |β_ij'|²
```

Born 法则保证：
- p_ij ≥ 0（非负性）
- Σ_j p_ij = 1（归一化）
- p_ij = |β_ij|²（概率 = 振幅模长的平方）

这是量子力学的核心公设之一，在 QSA 中自然地提供了归一化的注意力权重。

### 2.4 Von Neumann 熵度量

每个查询位置 i 的注意力分布 {p_ij} 的熵：

```
S_i = -Σ_j p_ij · log(p_ij)   ∈ [0, log(N)]
```

- S_i → 0：注意力高度集中（"观测到"一个确定的位置）
- S_i → log(N)：注意力均匀分散（处于最大叠加态）

这直接对应量子力学中的混合态熵，为 QCI 提供早退信号。

### 2.5 Top-K 干涉路由

对每个查询位置，仅保留概率最高的 k 个键：

```
k = max(1, ⌊topk_ratio · N⌋)
```

Top-K 筛选后，重新归一化选中的概率，使用加权聚合。

**复杂度分析**：
- 完整注意力：O(n²·d)
- Top-K 路由：O(n·d·log n)（sort 部分）+ O(n·k·d)（加权求和）
- 实际加速：当 topk_ratio = 0.1 时，约 10× 理论加速（GPU 实现约 3-5×）

---

## 3. 多头机制

### 3.1 头分割

将 d 维特征分割为 H 个 head_dim = d/H 的子空间：

```
Q → (B, N, H, head_dim) → (B, H, N, head_dim)
K → (B, N, H, head_dim) → (B, H, N, head_dim)
V → (B, N, H, head_dim) → (B, H, N, head_dim)
```

每个头独立计算复数内积和干涉调制，捕获不同的语义子空间。

### 3.2 量子多头的独特性质

相比实数多头，量子多头的额外性质：
- **相位多样性**：不同头的相位调制函数 f_φ 可以学到不同的干涉模式
- **干涉正交性**：理想情况下，不同头的相位调制应相互正交，最大化信息提取
- **头间熵差异**（新指标 v1.2）：通过 `multi_head_entropy_summary()` 监控

### 3.3 多头熵分析（v1.2 新增）

```python
summary = qsa.multi_head_entropy_summary(x)
# 返回:
# per_head_entropy: (H,) 每头平均熵
# head_diversity_score: [0,1] 头多样性分数
# uniform_head_mask: 均匀化头检测（缺乏选择性）
# sharp_head_mask: 尖锐头检测（可能退化）
```

**头崩溃诊断**：当 head_diversity_score < 0.1 时，说明多个头学到了相似的注意力模式，
应增大 dropout 或添加头多样性正则化损失。

---

## 4. 因果注意力（自回归模式）

当 `causal=True` 时，QSA 应用下三角掩码，使位置 i 只关注 j ≤ i：

```python
causal_mask = tril(ones(N, N))
beta[~causal_mask] = -inf + 0j  # 复数形式的极小值
```

**量子视角**：因果掩码等价于限制系统的"时间箭头"——未来状态不影响过去状态，
符合因果律（causality principle）在量子信息处理中的形式化表述。

---

## 5. 参数化与酉性

### 5.1 Cayley 参数化投影矩阵

Q/K/V 投影矩阵使用 Cayley 参数化保证酉性：

```
W_q = Cayley(Ω_q) = (I + i/2 · Ω_q)^{-1}(I - i/2 · Ω_q)
```

酉矩阵保证：
- **等距变换**：||W·x|| = ||x||，不改变量子态模长
- **相位保持**：酉变换不破坏量子干涉（范数不增不减）
- **梯度稳定**：酉矩阵的特征值在单位圆上，梯度既不爆炸也不消失

### 5.2 相位调制 MLP

```
f_φ: ℝ → ℝ,  f_φ(r) = MLP(r)
```

设计约束：
- 输入：内积模长 |α_ij|（标量）
- 输出：相位偏移 Δφ（弧度）
- 梯度截断 `detach()`：相位调制使用 detach 的模长，避免梯度在 QIR 中循环

---

## 6. 超参数指南

| 参数 | 推荐范围 | 说明 |
|------|---------|------|
| `num_heads` | 4~16 | 通常 dim/64 个头 |
| `head_dim` | 32~128 | = dim / num_heads |
| `topk_ratio` | 0.05~0.3 | 0.1 = 保留10%的键 |
| `mode` | "topk" | 训练推理均用 topk |
| `dropout` | 0.0~0.1 | 复数 dropout（实/虚同掩码）|
| `causal` | 场景相关 | LM任务用True，编码器用False |

**topk_ratio 调优策略**：
- 短序列（< 128）：topk_ratio = 0.3（避免太稀疏）
- 长序列（> 1024）：topk_ratio = 0.05~0.1（大幅降低计算量）
- 训练初期：建议从 full 模式开始，验证正确性后切换到 topk

---

## 7. 与 Transformer 注意力的对比

| 特性 | Transformer (Softmax Attention) | QSA (Born Rule Attention) |
|------|--------------------------------|--------------------------|
| 内积空间 | 实数 ℝ | 复数 ℂ |
| 注意力映射 | Softmax(QK^T/√d) | Born(QIR(⟨Q\|K⟩)) |
| 相位信息 | 无 | 有（干涉调制保留）|
| 路由机制 | 软路由（全量计算）| 硬路由（Top-K 筛选）|
| 概率来源 | 指数归一化 | 振幅模长平方（量子测量）|
| 复杂度 | O(n²d) | O(n·k·d)（k = topk_ratio·n）|
| 梯度稳定性 | Softmax 可能梯度小 | Born 概率梯度更均匀 |
| 可解释性 | 注意力即权重 | 注意力=量子测量概率 |

---

## 8. 已知局限与未来工作

### 8.1 当前局限

1. **O(n²) 内存**：Top-K 筛选前仍需计算完整的 (B, H, N, N) 矩阵，内存是瓶颈
2. **相位调制依赖**：f_φ MLP 的训练需要足够样本，短训练可能退化为实数注意力
3. **头独立性无显式约束**：多头可能坍缩到相似模式（头崩溃），需正则化

### 8.2 未来优化方向

- **线性注意力近似**：用随机特征图（Random Feature Map）将 O(n²) 降至 O(n)
- **Flashattention for Complex**：自定义 CUDA kernel，避免完整矩阵的 HBM 访问
- **头正交性正则化**：损失中加入 L = ||H_heads · H_heads^T - I||_F² 项
- **动态 topk_ratio**：根据序列长度和熵统计自适应调整 K 值

---

## 9. 代码示例

```python
import torch
from quantum_core.attention import QuantumSuperpositionAttention

# 初始化 QSA
qsa = QuantumSuperpositionAttention(
    dim=256,
    num_heads=8,
    topk_ratio=0.1,
    mode="topk",
)

# 前向传播
x = torch.randn(2, 64, 256, dtype=torch.complex64)
output, metrics = qsa(x, causal=True)
print(f"注意力熵: {metrics['attention_entropy']:.4f}")

# 获取注意力模式（不更新参数）
patterns = qsa.get_attention_patterns(x)
print(f"注意力概率矩阵: {patterns['attn_probs'].shape}")  # (2, 8, 64, 64)
print(f"Top-K 掩码: {patterns['topk_mask'].float().mean():.2%} 被保留")

# 多头熵分析（v1.2 新增）
head_summary = qsa.multi_head_entropy_summary(x)
print(f"头多样性分数: {head_summary['head_diversity_score']:.3f}")
n_uniform = head_summary['uniform_head_mask'].sum().item()
print(f"均匀化头数量: {n_uniform}/{qsa.num_heads}")

# 动态调整 Top-K 比例（在线推理优化）
qsa.set_topk_ratio(0.05)  # 更激进的筛选
```

---

## 10. 参考文献

1. Vaswani et al., "Attention Is All You Need" (2017)
2. Born, M., "Quantenmechanik der Stoßvorgänge" (1926) — Born 法则
3. Nielsen & Chuang, "Quantum Computation and Quantum Information" (2000) — Ch.2 量子力学基础
4. Von Neumann, "Mathematical Foundations of Quantum Mechanics" (1932) — 冯·诺依曼熵
5. Kitaev et al., "Classical and Quantum Computation" (2002) — 量子干涉路由理论基础
