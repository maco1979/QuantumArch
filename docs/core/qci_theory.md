# QCI — 量子坍缩推理理论文档

> **模块位置**：`quantum_core/collapse.py`
> **版本**：v1.2（2026-03-26）

---

## 1. 动机与背景

传统 Transformer 对每个 token 执行相同深度的前向传播，无论该 token 的语义复杂度如何。这导致：
- 简单 token（标点符号、停用词）消耗与复杂 token（实体、动词）相同的计算资源
- 网络无法根据"当前已知了多少"自适应调整计算深度

**量子坍缩推理（QCI，Quantum Collapse Inference）** 受量子测量的"波函数坍缩"启发，将推理过程视为一系列测量决策：

> 当量子态 |ψ⟩ 的**不确定性足够低**时，停止深层处理，直接"坍缩"到最可能的状态。

---

## 2. 量子测量理论基础

### 2.1 POVM 测量

正定算子值测量（POVM，Positive Operator-Valued Measure）是量子测量的最一般框架，
比标准投影测量（von Neumann 测量）更灵活。

**定义**：POVM 由一组正半定算子 {E_m} 组成，满足完整性约束：

$$\sum_m E_m = I$$

其中 E_m = M_m^\dagger M_m ≥ 0（正半定），I 为单位矩阵。

**测量规则**：
- 测量结果为 m 的概率：$P(m) = \langle\psi|E_m|\psi\rangle = \text{Tr}(E_m |\psi\rangle\langle\psi|)$
- 测量后态（坍缩态）：$|\psi_m\rangle = M_m|\psi\rangle / \sqrt{P(m)}$

**完整性保证**：$\sum_m P(m) = \langle\psi|\sum_m E_m|\psi\rangle = 1$

### 2.2 不确定性度量：冯·诺依曼熵

对于纯态 |ψ⟩ ∈ ℂ^d，严格的冯·诺依曼熵 S(|ψ⟩⟨ψ|) = 0（纯态永远是零熵）。

QCI 使用 **Born 概率分布的 Shannon 熵** 作为实用不确定性度量：

$$S_{Born}(\psi) = -\sum_{i=1}^d P_i \log P_i, \quad P_i = \frac{|\alpha_i|^2}{\sum_j |\alpha_j|^2}$$

这衡量的是："在特征维度上，激活分布有多集中（确定）？"

| 熵值 | 物理含义 |
|------|----------|
| S ≈ 0 | 振幅高度集中在少数维度 → 高确定性，适合早退 |
| S ≈ log(d) | 振幅均匀分布 → 最大不确定性，需要继续计算 |

### 2.3 最大熵

对于 d 维空间的量子态：

$$S_{max} = \log d$$

QCI 将所有阈值以 $S_{max}$ 归一化，使不同维度的模型阈值设置具有可比性。

---

## 3. 坍缩决策机制

### 3.1 双阈值策略

QCI 使用低阈值 τ_low 和高阈值 τ_high 将 token 分为三类：

```
            τ_low        τ_high
               |            |
  ─────────────┼────────────┼──────────────
  高确定性区    | 不确定区    |  高不确定区
  (早退坍缩)   | (继续传播)  |  (深层处理)
```

- $S < \tau_{low}$：token 已足够确定，通过 POVM 坍缩后**早退**
- $\tau_{low} \leq S \leq \tau_{high}$：正常传播到下一层
- $S > \tau_{high}$：高度不确定，可触发额外计算（扩展接口，当前版本传播处理）

### 3.2 训练阶段：Gumbel-Softmax 可微坍缩

训练时需要可微的坍缩决策，以使梯度从损失函数流回坍缩决策本身。
直接使用 hard threshold 不可微，因此使用 **Sigmoid 置信度加权混合**：

$$\text{confidence}(x) = \sigma(-(S(x) - \tau_{low}) \cdot T)$$

$$\text{output} = c(x) \cdot x + (1 - c(x)) \cdot |\psi_{\text{collapsed}}\rangle$$

其中温度 T 随训练步数退火（逆退火，从低 → 高）：

$$T(t) = T_{init} + (T_{max} - T_{init}) \cdot (1 - e^{-t/\tau_{decay}})$$

参数设置：
- $T_{init} = 2.0$（训练初期：平滑决策，梯度流畅）
- $T_{max} = 10.0$（训练后期：接近阶跃函数，决策锐利）
- $\tau_{decay} = 2000$（约 2000 步完成主要退火）

**退火曲线示意**：

```
T
10 |                              ╭─────────────
   |                         ╭───╯
   |                    ╭────╯
   |               ╭────╯
 2 |───────────────╯
   └─────────────────────────────────────────> step
   0    500   1000   1500   2000   2500
```

### 3.3 推理阶段：硬阈值坍缩

推理时使用硬阈值决策（无梯度需求）：

```python
should_collapse = entropy < tau_low          # (B, N) bool
output = torch.where(should_collapse, collapsed_state, x)
```

**早退率**（Early Exit Rate）：

$$\text{EER} = \frac{|\{(b,n) : S(x_{bn}) < \tau_{low}\}|}{B \times N}$$

EER 越高，模型在该层完成更多早退，后续层跳过更多 token，**推理效率越高**。

---

## 4. 自适应阈值

### 4.1 设计原则

固定阈值存在问题：
- 训练初期（模型未收敛）：固定 τ_low = 0.5 可能导致过多错误坍缩
- 训练后期（模型已稳定）：固定阈值导致太少早退，浪费推理计算

自适应阈值 `AdaptiveThreshold` 的策略：

**tau_low 指数衰减**（保守 → 激进）：

$$\tau_{low}(t) = \max\left(\tau_{low,init} \cdot \text{decay}^t, \tau_{min}\right)$$

| 参数 | 默认值 | 含义 |
|------|--------|------|
| τ_low,init | 0.5 | 初始低阈值 |
| decay | 0.95 | 每 100 步衰减率 |
| τ_min | 0.1 | 最小低阈值（防止坍缩过早） |

**tau_high 双阶段自适应**：
- **前期**（step ≤ 500）：跟踪历史熵分布，τ_high = min(μ_S × 2.0, S_max × 0.95)
- **后期**（step > 500）：额外施加衰减，τ_high *= 0.98^(late_steps/100)，减少总计算量

### 4.2 两阈值间距维护

始终保证 τ_high - τ_low ≥ 0.3，防止：
- 决策带过窄（大量 token 落在不确定区）
- 阈值倒置（τ_high < τ_low 导致逻辑错误）

---

## 5. POVM 参数化与正则化

### 5.1 测量基参数化

`POVMProjector` 使用 Softplus 权重 + 可学习测量基：

$$E_m = \alpha_m \cdot |\phi_m\rangle\langle\phi_m|, \quad \alpha_m = \text{softplus}(w_m) > 0$$

完整性近似条件：$\sum_m E_m \approx I$

当测量基 {φ_m} 接近正交（$\langle\phi_m|\phi_n\rangle \approx \delta_{mn}$）且 α_m ≈ 1 时，完整性自动满足。

### 5.2 正交正则化损失

为维护测量基的正交性，训练时添加辅助损失：

$$\mathcal{L}_{ortho} = \frac{\|B B^\dagger - I\|_F^2}{M}$$

其中 B = measurement_basis ∈ ℂ^{M×d}，M = out_dim（测量结果数）。

**推荐使用方式**：

```python
# 在总损失中加入正交正则化
collapse_loss = task_loss + 0.01 * qci.povm.orthogonality_regularization_loss()
```

### 5.3 完整性违背监控

训练日志中应监控 `collapse_povm_violation`（完整性违背度 ||ΣE_m - I||_F）：
- 健康范围：< 0.1（归一化测量）
- 异常信号：> 1.0（测量基严重不正交，坍缩概率不可信）

若违背度持续过高，可增大正交正则化系数或调用 `povm.renormalize_basis()`。

---

## 6. 计算效率分析

### 6.1 早退的计算节省

设模型共 L 层，第 l 层的早退率为 EER_l。

理论计算量节省（相对全层传播）：

$$\text{Savings} = \sum_{l=1}^{L-1} \text{EER}_l \cdot \frac{L - l}{L}$$

例：L=6 层，每层 EER = 0.3：

$$\text{Savings} \approx 0.3 \times \frac{5+4+3+2+1}{6} = 0.3 \times 2.5 \approx 0.75$$

即约节省 75% 的计算（相对于全层传播的 75% 参数被跳过）。

### 6.2 POVM 坍缩的复杂度

对每个需要坍缩的 token：
- 内积 ⟨φ_m|ψ⟩：O(M × D)
- 坍缩态重建：O(M × D)
- 总计：O(M × D) per token

M（测量结果数）等于 collapse_dim，默认等于 dim。
实践中可设 M << d（低秩坍缩）以进一步节省计算。

---

## 7. 超参数指南

| 超参数 | 推荐范围 | 说明 |
|--------|----------|------|
| `tau_low` | 0.3 ~ 0.7 | 越小 → 越少早退（保守），越大 → 越多早退（激进） |
| `tau_high` | 1.0 ~ 2.0 | 通常设为 τ_low 的 2~4 倍 |
| `collapse_dim` | dim/2 ~ dim | 降低可节省 POVM 计算，但可能损失信息 |
| `adaptive_tau` | True（默认） | 关闭时使用固定阈值（适合固定数据分布） |
| `decay_rate` | 0.90 ~ 0.99 | 0.90 衰减快（适合短训练），0.99 衰减慢（长训练） |

---

## 8. 调试与诊断

### 8.1 关键指标

训练日志中应关注以下 QCI 相关指标：

| 指标名 | 含义 | 健康范围 |
|--------|------|----------|
| `collapse_entropy` | 平均 Born 熵 | 随训练降低，< τ_high |
| `collapse_early_exit_rate` | 早退率 | 训练后期 > 0.2（层级降序）|
| `collapse_povm_violation` | POVM 完整性违背 | < 0.1 |
| `collapse_tau_low` | 当前低阈值 | 应从 0.5 逐渐降向 0.1 |
| `collapse_tau_high` | 当前高阈值 | 应从 1.5 逐渐降向 0.8 |

### 8.2 常见问题排查

**问题**：早退率始终为 0
- 原因 A：τ_low 过低，所有 token 熵 > τ_low
- 排查：检查 `collapse_entropy` 是否远大于 `collapse_tau_low`
- 修复：提高 τ_low_init，或降低 decay_rate

**问题**：POVM 违背度持续增大
- 原因：测量基向量模长漂移，正交性退化
- 排查：检查 `collapse_povm_violation` 趋势
- 修复：增大正交正则化权重 + 每 500 步调用 `povm.renormalize_basis()`

**问题**：训练损失出现周期性尖峰（每 100 步）
- 原因：自适应阈值更新时，τ_high 跳变导致大量 token 模式改变
- 修复：增大阈值更新间隔（修改 `step_count % 100` 为更大值）

---

## 9. 参考文献

1. **Nielsen & Chuang** (2000). *Quantum Computation and Quantum Information*. Cambridge University Press. Ch.2: 量子测量与 POVM 形式主义。

2. **Teerapittayanon et al.** (2016). BranchyNet: Fast Inference via Early Exiting. *ICPR 2016*. 早退推理的先驱工作。

3. **Schuster et al.** (2022). Confident Adaptive Language Modeling. *NeurIPS 2022*. 语言模型自适应计算深度。

4. **Cem et al.** (2019). Shallow-Deep Networks: Understanding and Mitigating Network Overthinking. *ICML 2019*. 网络"过度思考"问题。

---

*文档维护：automation-3 | 最后更新：2026-03-26*
