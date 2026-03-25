# QEL 量子纠缠层 — 理论基础与设计分析

> **文档版本**: v1.1  
> **对应实现**: `quantum_core/entanglement.py`  
> **依赖理论**: 量子纠缠、Schmidt 分解、量子傅里叶变换、酉耦合

---

## 1. 动机：为什么用纠缠替代残差连接？

### 1.1 Transformer 残差连接的局限

标准 Transformer 的层间连接为：

$$
\mathbf{y} = \mathbf{x} + \text{SubLayer}(\mathbf{x})
$$

这是一个**实数线性叠加**，具有以下局限：

| 局限 | 具体表现 |
|------|----------|
| **线性性** | 叠加权重固定为 1，无法自适应调节 |
| **相位丢失** | 实数加法无法表示相位干涉 |
| **独立性** | token 间信息通过 Attention 交换，残差不建立直接关联 |
| **无量子关联** | 无法捕获非局域的量子纠缠效应 |

### 1.2 量子纠缠层的优势

QEL 将残差连接替换为**量子纠缠操作**：

$$
|\psi_{\text{out}}\rangle = U_{\text{couple}}\left(|\psi_{\text{input}}\rangle \otimes |\psi_{\text{entangled}}\rangle\right) \xrightarrow{\text{投影}} \mathbb{C}^d
$$

优势对比：

| 特性 | 残差连接 | 量子纠缠层 |
|------|----------|------------|
| 相位处理 | ✗ 丢失 | ✓ 保留并利用 |
| token 间耦合 | 仅通过 Attention | 直接酉耦合 |
| 自适应性 | ✗ 固定系数 | ✓ 动态纠缠强度 |
| 梯度稳定性 | ✓（恒等路径） | ✓（酉变换保模长） |
| 信息守恒 | ✗（理论上允许放大） | ✓（酉性保证等距） |

---

## 2. 理论基础

### 2.1 量子纠缠的数学定义

两个量子态 $|a\rangle, |b\rangle \in \mathbb{C}^d$ 的乘积态为：

$$
|a\rangle \otimes |b\rangle \in \mathbb{C}^{d^2}
$$

一个**纠缠态**是乘积态的**叠加**，不能写成单个乘积态的形式：

$$
|\psi_{AB}\rangle = \sum_{i,j} c_{ij} |i\rangle_A \otimes |j\rangle_B \quad \text{（纠缠，若不可分离）}
$$

**纠缠度量**（Concurrence）：

$$
C(|a\rangle, |b\rangle) = 1 - |\langle a|b\rangle|^2
$$

取值 $[0, 1]$，正交态 $C = 1$（最大纠缠），相同态 $C = 0$（无纠缠）。

### 2.2 Schmidt 分解

任意二分纯态 $|\psi_{AB}\rangle \in \mathcal{H}_A \otimes \mathcal{H}_B$ 有唯一的 Schmidt 分解：

$$
|\psi_{AB}\rangle = \sum_{k=1}^{r} \sigma_k |u_k\rangle_A \otimes |v_k\rangle_B
$$

其中：
- $\{|u_k\rangle\}$, $\{|v_k\rangle\}$ 分别是 $\mathcal{H}_A$, $\mathcal{H}_B$ 中的正交归一基
- $\sigma_k > 0$ 为 Schmidt 系数（$\sum_k \sigma_k^2 = 1$）
- $r$ 为 **Schmidt 秩**（纠缠程度的度量，$r=1$ 为可分离态）

**为什么使用 Schmidt 分解**：

完整的 $d^2$ 维张量积空间在 $d$ 较大时不可行（如 $d=512$ 时 $d^2 = 262144$）。
Schmidt 截断秩 $r \ll d$ 提供了精度-效率的权衡：

$$
|\psi_{AB}\rangle \approx \sum_{k=1}^{r} \sigma_k |u_k\rangle_A \otimes |v_k\rangle_B
$$

QEL 实现中取 $r = \min(d, 32)$（`SchmidtEntanglementGate`）。

### 2.3 量子傅里叶变换（QFT）

QFT 是建立**全局**量子纠缠的标准方法。$N$ 维 QFT 定义为：

$$
\text{QFT}_N |j\rangle = \frac{1}{\sqrt{N}} \sum_{k=0}^{N-1} \omega^{jk} |k\rangle, \quad \omega = e^{2\pi i / N}
$$

变换矩阵元素：$[\text{QFT}_N]_{j,k} = \omega^{jk} / \sqrt{N}$

**关键性质**：
- **酉性**：$\text{QFT}^\dagger \text{QFT} = I$（信息无损）
- **全局关联**：变换后每个分量都依赖所有原始分量（感受野 = 全序列）
- **实现方法**：等价于 `torch.fft.fft(..., norm='ortho')`

QEL 在**序列维度**（token 之间，`dim=1`）而非特征维度执行 QFT，这与量子电路中对多量子比特系统的 QFT 操作对应。

---

## 3. QEL 架构设计

### 3.1 整体流程

```
输入: x ∈ ℂ^{B × N × d}
        │
        ▼
┌───────────────────────────────┐
│  1. 局部纠缠（相邻 token 对）  │
│     AdaptiveEntanglementGate   │
│     批量处理 (x_0,x_1), (x_2,x_3), ... │
└─────────────────┬─────────────┘
                  │  entangled ∈ ℂ^{B × N × d}
                  ▼
┌───────────────────────────────┐
│  2. 全局纠缠（序列维度 QFT）   │
│     QFT_N（在 token 轴）      │
│     α·entangled + (1-α)·fft  │
└─────────────────┬─────────────┘
                  │  entangled_global ∈ ℂ^{B × N × d}
                  ▼
┌───────────────────────────────┐
│  3. 酉耦合融合（替代残差）     │
│     U_couple([x; entangled])  │
│     → 输出 d 维               │
└─────────────────┬─────────────┘
                  │
输出: output ∈ ℂ^{B × N × d}
```

### 3.2 局部纠缠：自适应纠缠门

**核心公式**：

$$
\theta_i = \sigma\left(\text{MLP}\left([|a_i|^2; |b_i|^2; \text{Re}(\langle a|b\rangle); \text{Im}(\langle a|b\rangle)]\right)\right) \cdot \theta_{\max}
$$

$$
a_i' = (1 - \theta_i) \cdot a_i + \theta_i \cdot \tilde{a}_i
$$

其中 $\tilde{a}_i$ 是通过 Schmidt 纠缠门处理后的态：

$$
[\tilde{a}; \tilde{b}] = U_{\text{ent}} \cdot [a; b], \quad U_{\text{ent}} \in \mathcal{U}(2d)
$$

**参数复杂度**：
- $U_{\text{ent}}$（CayleyLinear）：$O(d^2)$ 参数
- Strength MLP：$O(d)$ 参数（$4 \to d/4 \to 1$）

### 3.3 全局纠缠：可学习 QFT 混合

$$
\text{result} = \alpha \cdot x + (1-\alpha) \cdot \text{QFT}(x)
$$

其中 $\alpha \in (0, 1)$ 通过 sigmoid 参数化学习：

$$
\alpha = \sigma\left(\log\frac{\alpha_0}{1-\alpha_0}\right), \quad \alpha_0 = 0.5 \text{（初始值）}
$$

- $\alpha \to 1$：保留局部表示，QFT 贡献小
- $\alpha \to 0$：全局频域表示，长程关联强

### 3.4 酉耦合：替代残差连接

**Full 耦合**（精确酉）：

$$
[\mathbf{c}_1; \mathbf{c}_2] = U_{\text{couple}} \cdot [x; \text{entangled}], \quad U_{\text{couple}} \in \mathcal{U}(2d)
$$

$$
\text{output} = \frac{\mathbf{c}_1 + \mathbf{c}_2}{\sqrt{2}}
$$

**Diagonal 耦合**（高效近似）：

$$
\text{output}_k = e^{i\phi_k} \left(\cos\theta_k \cdot x_k + \sin\theta_k \cdot \text{entangled}_k\right)
$$

其中 $\theta_k = \sigma(\text{mix}_k) \cdot \pi/2$，$\phi_k$ 为可学习相位偏移。

| 耦合类型 | 参数量 | 酉性 | 使用场景 |
|----------|--------|------|----------|
| `full` | $O(d^2)$ | 严格 | 高精度实验 |
| `diagonal` | $O(d)$ | 严格（对角酉） | 高效生产 |

---

## 4. 纠缠度量与监控

### 4.1 纠缠度 (Concurrence)

$$
C = 1 - |\langle a_{\text{norm}} | b_{\text{norm}} \rangle|^2
$$

- $C \in [0, 1]$，监控局部纠缠对的平均纠缠强度
- 实现：`entanglement.concurrence(a, b, dim=-1)`

### 4.2 纠缠熵 (Entanglement Entropy)

$$
S(A) = -\sum_k \lambda_k \log \lambda_k
$$

其中 $\lambda_k$ 是约化密度矩阵 $\rho_A$ 的特征值。QEL 使用 Born 概率分布的 Shannon 熵作为近似。

### 4.3 QFT 混合系数 $\alpha$

通过 `qel.qft.alpha` 监控全局/局部纠缠平衡：
- 训练初期：$\alpha$ 通常从 0.5 开始
- 训练后期：根据任务收敛到最优值

### 4.4 导出 API

```python
# 推理阶段诊断
with torch.no_grad():
    metrics = qel.get_entanglement_metrics(x)
    print(f"纠缠度均值: {metrics['concurrence_mean']:.4f}")
    print(f"纠缠熵: {metrics['entanglement_entropy']:.4f}")
    print(f"QFT 混合系数: {metrics['qft_alpha']:.4f}")
```

---

## 5. 与 Transformer 的对比

| 维度 | Transformer 残差 | QEL |
|------|------------------|-----|
| **操作类型** | 实数向量加法 | 复数酉纠缠变换 |
| **token 关联** | 仅通过 Attention | 直接局部+全局纠缠 |
| **相位信息** | 不涉及 | 核心利用 |
| **梯度路径** | 恒等（防消失） | 酉（保模长） |
| **参数量** | 0（无额外参数） | $O(d^2)$（完整）/$O(d)$（对角） |
| **计算复杂度** | $O(N \cdot d)$ | $O(N \cdot d)$（局部）+ $O(N \log N)$（QFT） |
| **感受野** | 受限于 Attention（Top-K） | 全局（通过 QFT） |

---

## 6. 超参数指南

| 超参数 | 推荐值 | 说明 |
|--------|--------|------|
| `use_adaptive` | `True` | 自适应纠缠强度，训练稳定性好 |
| `use_global_qft` | `True` | 全局纠缠，通常有益于长序列 |
| `coupling_type` | `"diagonal"` | 平衡效率与表达力 |
| `qft_steps` | 1 | 单步 QFT 足够，多步提升有限但开销翻倍 |
| `init_alpha` | 0.5 | 初始等权混合，让模型自适应 |
| `theta_max` | 1.0 | 最大纠缠角，通常不需要调整 |

**调参建议**：
1. 先用 `coupling_type="diagonal"` 验证有效性，再切换到 `"full"` 提升精度
2. 监控 `avg_concurrence`，若持续为 0 说明纠缠强度 MLP 未激活（检查学习率）
3. 若训练不稳定，先关闭 `use_global_qft` 排查 QFT 的影响

---

## 7. 参考文献

1. Nielsen, M. & Chuang, I. (2000). *Quantum Computation and Quantum Information*. Cambridge University Press. — 纠缠理论基础
2. Vidal, G. (2003). *Efficient Classical Simulation of Slightly Entangled Quantum Computations*. PRL. — Schmidt 截断
3. Coppersmith, D. (1994). *An Approximate Fourier Transform Useful in Quantum Factoring*. IBM. — QFT 电路
4. He, K. et al. (2016). *Deep Residual Learning for Image Recognition*. CVPR. — 残差连接（对比基线）

---

*本文档由量子架构自动化迭代系统生成，最后更新：2026-03-25*
