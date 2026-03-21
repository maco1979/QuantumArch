# 量子架构核心概念定义

**QuantumArch Core Concepts v1.0**

---

## 1. 量子叠加注意力 (Quantum Superposition Attention, QSA)

### 1.1 理论基础

传统Transformer的注意力机制基于点积：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其本质是一个**归一化的加权求和**，每个token对其他所有token的信息进行确定性聚合。

**量子叠加注意力**将这一机制重新建模为量子态的叠加过程：

$$|\psi_{attn}\rangle = \sum_{i} \alpha_i |e_i\rangle$$

其中 $\alpha_i \in \mathbb{C}$ 为复数振幅，满足 $|\alpha_i|^2$ 为注意力概率分布。

### 1.2 数学定义

给定输入序列 $X = [x_1, x_2, \ldots, x_n]$，每个token映射为**量子态**：

$$|x_i\rangle = \sum_{j=1}^{d} w_{ij} |e_j\rangle, \quad w_{ij} \in \mathbb{C}, \quad \sum_j |w_{ij}|^2 = 1$$

其中 $d$ 为量子态维度（类比隐藏维度），$w_{ij}$ 为复数嵌入权重。

**叠加注意力操作**：

$$\text{QSA}(X) = \sum_{i=1}^{n} \alpha_i U_{qi} |x_i\rangle$$

其中：
- $\alpha_i = \langle\phi_q | x_i \rangle$ 是复数内积振幅
- $U_{qi} \in U(d)$ 是参数化的酉变换矩阵
- $|\phi_q\rangle$ 是可学习的**查询量子态**

### 1.3 计算复杂度

| 操作 | 传统注意力 | QSA |
|------|-----------|-----|
| 注意力矩阵 | O(n²·d) | O(n·d·k) 其中k为选定量子比特数 |
| 加权聚合 | O(n²·d) | O(n·d) |
| 总复杂度 | **O(n²·d)** | **O(n·d·log n)** |

关键优化：通过**量子干涉筛选**（见第4节），无需计算完整的n×n注意力矩阵。

### 1.4 核心特性

- **复数相位编码**：复数振幅的相位信息编码token间的相对关系
- **干涉增强**：相关token的波函数相长相干，无关token相消干涉
- **不确定性保留**：叠加态保持多种可能的解释，延迟到坍缩时决定

---

## 2. 量子纠缠层 (Quantum Entanglement Layer, QEL)

### 2.1 理论基础

量子纠缠是两个或多个量子系统之间的一种非经典关联——测量一个系统的状态会瞬时影响另一个系统的状态，无论它们之间的距离。

在量子架构中，**纠缠层**模拟了这一机制，使token之间建立超越简单加权求和的深层关联。

### 2.2 数学定义

对于两个token的量子态 $|a\rangle$ 和 $|b\rangle$，**纠缠操作**定义为：

$$\text{ENTANGLE}(|a\rangle, |b\rangle) = (U_{ent} \otimes I)(I \otimes U_{ent}) |a\rangle \otimes |b\rangle$$

其中 $U_{ent} \in U(2)$ 是参数化的2-qubit纠缠门（类比CNOT/CRZ门）。

**纠缠层的批量操作**：

给定序列 $[|x_1\rangle, |x_2\rangle, \ldots, |x_n\rangle]$：

1. **局部纠缠**：相邻token对应用纠缠门
   $$|\psi^{(1)}_i\rangle = U_{ent}(|x_i\rangle, |x_{i+1}\rangle)$$

2. **全局纠缠**：通过量子傅里叶变换建立长程关联
   $$|\psi^{(global)}\rangle = \text{QFT} \cdot |\psi^{(1)}\rangle$$

3. **纠缠强度自适应**：纠缠门参数 $\theta$ 根据输入动态调整
   $$U_{ent}(\theta_i) = \begin{pmatrix} \cos\theta_i & i\sin\theta_i \\ i\sin\theta_i & \cos\theta_i \end{pmatrix}$$

### 2.3 纠缠的信息论解释

纠缠度的度量使用**量子互信息**：

$$I(A:B) = S(\rho_A) + S(\rho_B) - S(\rho_{AB})$$

其中 $S(\rho) = -\text{Tr}(\rho \log \rho)$ 是冯·诺依曼熵。

- **高纠缠**：token间存在强关联（如语法依赖）
- **低纠缠**：token相对独立（如并列短语）

### 2.4 替代Transformer的残差连接

Transformer通过残差连接传递信息：
$$y = x + \text{SubLayer}(x)$$

量子架构用纠缠耦合替代：
$$|\psi_{out}\rangle = U_{couple}(\theta)(|\psi_{input}\rangle \otimes |\psi_{sublayer}\rangle)$$

信息不是简单相加，而是通过纠缠实现**非线性融合**，保留了量子态的相位关系。

---

## 3. 量子坍缩推理 (Quantum Collapse Inference, QCI)

### 3.1 理论基础

在量子力学中，观测导致波函数从叠加态坍缩到确定态。在量子架构中，**坍缩推理**利用这一机制实现概率性、自适应的推理路径。

### 3.2 数学定义

给定量子态 $|\psi\rangle = \sum_i \alpha_i |e_i\rangle$：

**确定性坍缩**（标准推理）：
$$P(\text{collapse to } |e_i\rangle) = |\alpha_i|^2$$

**测量算子坍缩**（可训练推理）：
$$P(m) = \langle\psi| M_m^\dagger M_m |\psi\rangle$$

其中 $\{M_m\}$ 是一组**正交测量算子**（POVM）。

**多级坍缩**（层次化推理）：
$$|\psi\rangle \xrightarrow{M^{(1)}} |\psi^{(1)}\rangle \xrightarrow{M^{(2)}} |\psi^{(2)}\rangle \xrightarrow{\cdots} |\psi^{(final)}\rangle$$

每一级坍缩将量子态从高维叠加态逐步投影到低维确定态。

### 3.3 坍缩在推理中的应用

1. **自适应计算**：简单token快速坍缩（少量量子门），复杂token延迟坍缩（更多量子门）
2. **早期退出**：当坍缩概率超过阈值 $\tau$ 时提前退出当前层
3. **不确定性量化**：坍缩概率的熵直接给出模型的不确定性
   $$H = -\sum_i |\alpha_i|^2 \log |\alpha_i|^2$$

### 3.4 与传统推理的对比

| 特性 | Transformer推理 | 量子坍缩推理 |
|------|----------------|-------------|
| 确定性 | 完全确定性 | 概率性，但可控 |
| 计算量 | 固定 | 自适应 |
| 不确定性 | 无内建机制 | 熵度量自然给出 |
| 输出多样性 | 需要temperature | 坍缩概率直接控制 |

---

## 4. 量子干涉路由 (Quantum Interference Routing, QIR)

### 4.1 理论基础

量子干涉是量子计算加速的核心机制——正确路径的概率振幅相长叠加，错误路径相消叠加。

在量子架构中，**干涉路由**用这一原理实现高效的信息筛选和路由。

### 4.2 数学定义

给定多条信息路径的量子态 $\{|\psi_1\rangle, |\psi_2\rangle, \ldots, |\psi_k\rangle\}$：

**干涉操作**：
$$|\psi_{routed}\rangle = \sum_{j=1}^{k} e^{i\phi_j} \beta_j |\psi_j\rangle$$

其中 $\phi_j$ 是可学习的**相位偏移**，$\beta_j$ 是路由振幅。

**相长干涉**（增强相关信号）：
当 $e^{i\phi_j}\beta_j$ 同相时，$|\psi_{routed}\rangle$ 的振幅增大。

**相消干涉**（抑制噪声）：
当 $e^{i\phi_j}\beta_j$ 反相时，振幅减小甚至为零。

### 4.3 在注意力中的应用

传统注意力需要计算O(n²)的注意力矩阵。量子干涉路由通过以下方式将其降低到O(n log n)：

1. **波函数编码**：将所有token的键向量编码到量子态
2. **干涉筛选**：用查询态作为干涉参考，相消无关token
3. **概率采样**：从干涉后的概率分布中采样top-k个token
4. **局部精确注意力**：仅对选出的k个token计算精确注意力

$$\text{QIR-Attention}(q, X) = \text{LocalAttention}(q, \text{QIR-Sample}(q, X, k))$$

---

## 5. 量子梯度下降 (Quantum Gradient Descent, QGD)

### 5.1 理论基础

传统梯度下降在实数空间优化损失函数。**量子梯度下降**在复数希尔伯特空间中优化，利用量子态的几何性质加速收敛。

### 5.2 数学定义

复数参数 $\theta \in \mathbb{C}^n$ 的损失函数 $L(\theta, \bar{\theta})$：

**量子梯度**：
$$\nabla_{\bar{\theta}} L = \frac{\partial L}{\partial \bar{\theta}}$$

**参数更新**：
$$\theta_{t+1} = \theta_t - \eta \nabla_{\bar{\theta}_t} L$$

其中使用 **Wirtinger导数**（复数微积分）而非普通梯度。

### 5.3 干涉优化器 (Interference Optimizer)

利用波函数干涉加速梯度下降：

1. **相位梯度**：在复数空间的每个维度上独立施加相位旋转
   $$\theta_j \leftarrow \theta_j \cdot e^{-i\eta \frac{\partial L}{\partial \phi_j}}$$

2. **振幅调整**：根据损失梯度调整复数振幅的模
   $$|\theta_j| \leftarrow |\theta_j| \cdot (1 - \eta \frac{\partial L}{\partial |\theta_j|})$$

3. **正交约束保持**：酉矩阵通过Cayley变换保持酉性
   $$U_{new} = (I + \frac{i\delta}{2}\Omega)^{-1}(I - \frac{i\delta}{2}\Omega)$$

### 5.4 优势

- **避免鞍点**：复数空间中的几何结构使鞍点稀疏
- **更快的收敛**：相位梯度提供额外的优化自由度
- **酉性约束**：自然保持参数矩阵的正交/酉性，改善梯度流

---

## 6. 复数权重与酉表示 (Complex Weights & Unitary Representation)

### 6.1 参数化

量子架构中的所有权重矩阵都是**复数酉矩阵**：

$$W \in \mathbb{C}^{d \times d}, \quad W^\dagger W = WW^\dagger = I$$

参数化方式（保证可学习且满足酉性约束）：

**Cayley参数化**：
$$W(\Omega) = (I + \frac{i}{2}\Omega)^{-1}(I - \frac{i}{2}\Omega)$$

其中 $\Omega = \Omega^\dagger$ 为任意厄米矩阵。

**QR分解参数化**：
$$W = QR, \quad Q \in U(d), \quad R \in \mathbb{R}^{d\times d}$$

### 6.2 相位作为信息载体

复数权重的关键优势在于**相位编码信息**：

$$W = |W| \cdot e^{i\Phi(W)}$$

- **模** $|W|$：控制信息传递的强度
- **相位** $\Phi(W)$：控制信息传递的**角度/方向**

这比纯实数权重多了一维的信息编码能力，使网络能够表示更丰富的变换模式。

### 6.3 正交梯度流

由于酉矩阵满足 $W^\dagger W = I$，梯度在传播过程中**模长守恒**：

$$\frac{\partial L}{\partial h_l} = \frac{\partial L}{\partial h_{l+1}} W^\dagger$$

$$\left\|\frac{\partial L}{\partial h_l}\right\| = \left\|\frac{\partial L}{\partial h_{l+1}} W^\dagger\right\| \leq \left\|\frac{\partial L}{\partial h_{l+1}}\right\| \cdot \|W^\dagger\| = \left\|\frac{\partial L}{\partial h_{l+1}}\right\|$$

（因为酉矩阵的谱范数 $\|W\| = 1$）

这从根本上**解决了梯度消失/爆炸**问题。

---

## 7. 不确定性传播 (Uncertainty Propagation)

### 7.1 冯·诺依曼熵作为不确定性度量

每个中间层的量子态都可以计算其冯·诺依曼熵：

$$S(\rho) = -\text{Tr}(\rho \log \rho)$$

其中 $\rho = |\psi\rangle\langle\psi|$ 是纯态的密度矩阵。

- **纯态**（$S = 0$）：完全确定，无纠缠
- **混合态**（$S > 0$）：存在不确定性或纠缠

### 7.2 不确定性缓冲区

在深层网络中，每层维护一个**不确定性预算**：

$$U_{l+1} = U_l + \Delta U_{layer} - \Delta U_{collapse}$$

- $\Delta U_{layer}$：该层操作增加的不确定性
- $\Delta U_{collapse}$：坍缩操作减少的不确定性

当不确定性超过阈值时，触发额外的坍缩操作（确定性计算），确保模型不会在深层中积累过多噪声。

---

## 8. 完整前向传播流程

```
输入序列 X = [x_1, ..., x_n]
        │
        ▼
┌─────────────────────────┐
│  1. 量子嵌入层 (QEL)     │  x_i → |x_i⟩ (复数嵌入 + 归一化)
└─────────┬───────────────┘
          ▼
┌─────────────────────────┐
│  2. 位置编码 (QPE)       │  |x_i⟩ → U_{pos}(i)|x_i⟩ (酉旋转)
└─────────┬───────────────┘
          ▼
  ┌───────────────┐
  │ L层量子块 (QB) │ ◄─── 循环L次
  │  ┌─────────┐  │
  │  │ QSA     │  │  量子叠加注意力（干涉路由筛选 + 局部注意力）
  │  │ QEL     │  │  量子纠缠层（局部+全局纠缠）
  │  │ FFN_Q   │  │  量子前馈网络（复数MLP + 酉变换）
  │  │ Collapse│  │  可选坍缩（不确定性阈值触发）
  │  └─────────┘  │
  └───────┬───────┘
          ▼
┌─────────────────────────┐
│  3. 全局坍缩 (Final)    │  量子态 → 确定性输出（POVM测量）
└─────────┬───────────────┘
          ▼
        输出 Y
```

---

*本文档定义了量子架构的所有核心概念。后续文档将基于这些概念展开详细的架构设计和实验方案。*
