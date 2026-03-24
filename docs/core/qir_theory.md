# 量子干涉路由（QIR）理论文档

> **文件**: `docs/core/qir_theory.md`  
> **版本**: v1.0  
> **日期**: 2026-03-24  
> **模块**: `quantum_core/routing.py`

---

## 1. 理论动机

标准 MoE（混合专家）模型中，Token 路由基于 **Softmax + TopK** 的实数评分：

$$r_i = \text{Softmax}(W_g \cdot x_i), \quad \text{选取 top-}k \text{ 专家}$$

这种路由方式存在两个局限：

1. **路由信息丢失**：Softmax 归一化会压平低分区域，丢失路由判别信息
2. **缺乏相干效应**：不同路由路径之间无法产生量子力学意义上的**干涉**

量子干涉路由（QIR，Quantum Interference Routing）引入**复数路由振幅**，使多个路由路径可以发生**相长干涉**（增强优质路由）和**相消干涉**（抑制冗余路由）。

---

## 2. 数学形式

### 2.1 复数路由振幅

对于输入 token $x_i \in \mathbb{R}^d$（或 $\mathbb{C}^d$），第 $m$ 个专家的路由振幅定义为：

$$a_{im} = W_m^{\text{real}} x_i + i \cdot W_m^{\text{imag}} x_i \in \mathbb{C}$$

其中 $W_m^{\text{real}}, W_m^{\text{imag}} \in \mathbb{R}^{d}$ 是可学习的路由投影权重。

### 2.2 Born 法则路由概率

路由概率由 **Born 测量规则** 给出：

$$p_{im} = |a_{im}|^2 = (a_{im}^{\text{real}})^2 + (a_{im}^{\text{imag}})^2$$

归一化后：

$$\tilde{p}_{im} = \frac{p_{im}}{\sum_{m'} p_{im'}}$$

### 2.3 干涉机制

多路由路径之间的干涉通过**交叉项**实现。对于两条路径 $m, m'$：

$$I_{mm'} = 2 \text{Re}(a_{im} \cdot a_{im'}^*) = 2(a_{im}^R a_{im'}^R + a_{im}^I a_{im'}^I)$$

- $I_{mm'} > 0$：**相长干涉**，两路径协同增强
- $I_{mm'} < 0$：**相消干涉**，两路径相互抑制

干涉增强后的有效路由概率：

$$p_{im}^{\text{eff}} = |a_{im}|^2 + \lambda \sum_{m' \neq m} I_{mm'} \cdot \phi(m, m')$$

其中 $\phi(m, m')$ 是领域相关性矩阵，$\lambda$ 是干涉强度超参数。

### 2.4 稀疏化与 Top-K 选择

最终选择每个 token 路由到 $K$ 个专家（稀疏激活）：

$$\mathcal{S}_i = \text{TopK}\{m : \tilde{p}_{im}\}, \quad |\mathcal{S}_i| = K$$

专家加权输出：

$$\text{QIR}(x_i) = \sum_{m \in \mathcal{S}_i} \tilde{p}_{im} \cdot E_m(x_i)$$

---

## 3. 与标准 MoE 的对比

| 特性 | 标准 MoE | QIR |
|------|----------|-----|
| 路由打分 | 实数点积 | 复数振幅 |
| 概率计算 | Softmax | Born 法则 |
| 路径干涉 | ❌ 无 | ✅ 相长/相消干涉 |
| 参数量 | $M \cdot d$ | $2M \cdot d$ |
| 路由可解释性 | 低（黑箱） | 中（相位可视化） |
| 表达能力 | 线性分类 | 二次（Born 概率） |

---

## 4. 实现细节

### 4.1 数值稳定性

直接计算 $|a_{im}|^2$ 在路由振幅接近零时可能出现梯度消失。实际实现中使用：

```python
eps = 1e-8
prob = (a_real ** 2 + a_imag ** 2 + eps)
prob = prob / prob.sum(dim=-1, keepdim=True)
```

### 4.2 负载均衡辅助损失

为防止路由崩塌（所有 token 路由到少数专家），引入辅助均衡损失：

$$\mathcal{L}_{\text{balance}} = \alpha \cdot \text{CV}^2\left(\frac{1}{N}\sum_i p_{im}\right)$$

其中 $\text{CV}$ 是变异系数，$\alpha = 0.01$ 为损失权重。

### 4.3 相位正则化

对路由振幅的相位施加平滑正则，防止相位剧烈震荡：

$$\mathcal{L}_{\text{phase}} = \beta \cdot \sum_m \|\nabla_x \angle a_{im}\|_2^2$$

---

## 5. 理论性质

### 定理 5.1（路由完备性）

> **命题**: 对任意实数路由评分 $s \in \mathbb{R}^M$，存在复数振幅 $a \in \mathbb{C}^M$ 使得 $|a_m|^2 \propto e^{s_m}$，即 QIR 包含了标准 Softmax MoE 作为特例。

**证明思路**: 取 $a_m = \sqrt{e^{s_m/2}}$（实数），则 $|a_m|^2 = e^{s_m}$，归一化后恰好是 Softmax 分布。$\square$

### 定理 5.2（干涉增益上界）

> **命题**: 相长干涉能将有效路由概率最多放大 $M$ 倍，其中 $M$ 是专家数量。

**证明思路**: 对 $M$ 个振幅相位相同的情况，总振幅 $|\sum_m a_m|^2 = M^2 |a|^2$，归一化后单专家概率提升至 $M$ 倍。$\square$

---

## 6. 超参数指南

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `n_experts` | 8 ~ 32 | 专家数量，越多表达力越强 |
| `topk` | 2 ~ 4 | 稀疏激活专家数 |
| `interference_lambda` | 0.01 ~ 0.1 | 干涉强度 |
| `balance_loss_alpha` | 0.01 | 均衡损失权重 |
| `phase_reg_beta` | 0.001 | 相位正则强度 |

---

## 7. 参考文献

1. Shazeer et al. (2017). *Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer.* ICLR 2017.
2. Fedus et al. (2021). *Switch Transformers: Scaling to Trillion Parameter Models.* JMLR 2021.
3. Nielsen & Chuang (2010). *Quantum Computation and Quantum Information.* Cambridge University Press.
4. 量子架构白皮书 v1.1 (2026). *第4章：量子干涉路由机制详述.* 内部文档.

---

*本文档由量子架构团队维护。如有理论问题，请提交 Issue 至 [GitHub](https://github.com/maco1979/QuantumArch)。*
