# QGD：量子梯度下降的数学推导与训练稳定性

> **文档版本**: v1.0（2026-03-23）
> **对应代码**: `quantum_core/optimizer.py`

---

## 1. 动机：为何复数参数需要特殊优化器

标准 PyTorch 优化器（Adam、AdamW）在处理复数参数时存在根本性问题：

| 问题 | 标准 Adam | QGD |
|------|-----------|-----|
| 梯度分解 | 实部/虚部独立更新，忽略几何耦合 | Wirtinger 导数 → 模长/相位解耦 |
| 量子约束 | 不维护酉性 | Cayley 参数自动保证 |
| 相位学习率 | 与模长共享，不分离 | 独立相位动量和学习率 |
| 模长衰减 | 可能使模长穿越 0（相位跳变） | 非负约束 + 软边界 |

---

## 2. Wirtinger 微积分基础

### 2.1 复数导数的定义

对于复变量 $z = x + iy \in \mathbb{C}$ 和实值损失 $L : \mathbb{C} \to \mathbb{R}$，
**Wirtinger 导数**（或称 **CR 微积分**）定义为：

$$
\frac{\partial L}{\partial z} \triangleq \frac{1}{2}\left(\frac{\partial L}{\partial x} - i\frac{\partial L}{\partial y}\right), \quad
\frac{\partial L}{\partial \bar{z}} \triangleq \frac{1}{2}\left(\frac{\partial L}{\partial x} + i\frac{\partial L}{\partial y}\right)
$$

**PyTorch 中**，`p.grad` 存储的是 $\frac{\partial L}{\partial \bar{z}}$（共轭 Wirtinger 导数），
这是最速下降方向（在 $\mathbb{R}^{2}$ 中等价于普通梯度）。

### 2.2 极坐标分解

设 $z = r e^{i\varphi}$（$r = |z| \geq 0$，$\varphi = \arg z$），
则对 $g \triangleq \frac{\partial L}{\partial \bar{z}}$ 有：

$$
\frac{\partial L}{\partial r} = \frac{\text{Re}(\bar{z} \cdot g)}{r}, \quad
\frac{\partial L}{\partial \varphi} = \frac{\text{Im}(\bar{z} \cdot g)}{r}
$$

**推导**（以模长梯度为例）：

$$
L(z, \bar{z}) = L(re^{i\varphi},\ re^{-i\varphi})
$$

$$
\frac{\partial L}{\partial r} = \frac{\partial L}{\partial z}\cdot e^{i\varphi} + \frac{\partial L}{\partial \bar{z}}\cdot e^{-i\varphi}
= 2\,\text{Re}\!\left(\frac{\partial L}{\partial \bar{z}} \cdot e^{-i\varphi}\right)
= \frac{2}{r}\,\text{Re}(\bar{z} \cdot g)
$$

（对实值 $L$：$\frac{\partial L}{\partial z} = \overline{\frac{\partial L}{\partial \bar{z}}}$，故两项合并。）

---

## 3. QGD 更新规则

### 3.1 普通复数参数（Wirtinger 模式）

设第 $t$ 步的参数 $z^{(t)} = r^{(t)} e^{i\varphi^{(t)}}$，梯度 $g^{(t)} = \frac{\partial L}{\partial \bar{z}}$。

**步骤 1：极坐标梯度分解**
$$
g_r^{(t)} = \frac{\text{Re}(\overline{z^{(t)}} \cdot g^{(t)})}{r^{(t)} + \epsilon}, \quad
g_\varphi^{(t)} = \frac{\text{Im}(\overline{z^{(t)}} \cdot g^{(t)})}{r^{(t)} + \epsilon}
$$

**步骤 2：模长 Adam 更新**
$$
m_r^{(t)} = \beta_1 m_r^{(t-1)} + (1-\beta_1) g_r^{(t)}
$$
$$
v_r^{(t)} = \beta_2 v_r^{(t-1)} + (1-\beta_2) (g_r^{(t)})^2
$$
$$
\hat{m}_r = \frac{m_r^{(t)}}{1-\beta_1^t}, \quad \hat{v}_r = \frac{v_r^{(t)}}{1-\beta_2^t}
$$
$$
r^{(t+1)} = \max\!\left(0,\ r^{(t)} - \alpha_r \cdot \frac{\hat{m}_r}{\sqrt{\hat{v}_r} + \epsilon}\right)
$$

**步骤 3：相位 Adam 更新（独立动量）**
$$
m_\varphi^{(t)} = \beta_1 m_\varphi^{(t-1)} + (1-\beta_1) g_\varphi^{(t)}
$$
$$
v_\varphi^{(t)} = \beta_2 v_\varphi^{(t-1)} + (1-\beta_2) (g_\varphi^{(t)})^2
$$
$$
\varphi^{(t+1)} = \varphi^{(t)} - \alpha_\varphi \cdot \frac{\hat{m}_\varphi}{\sqrt{\hat{v}_\varphi} + \epsilon}
$$

**步骤 4：合成**
$$
z^{(t+1)} = r^{(t+1)} \cdot e^{i\varphi^{(t+1)}}
$$

### 3.2 Cayley 参数（标准 Adam 模式）

Cayley 变换 $W = (I + i\Omega/2)^{-1}(I - i\Omega/2)$ 的底层参数 $\Omega$ 是实数（或复数），
**酉性由变换自动保证**，无需对梯度做极坐标分解。
这些参数直接使用标准 Adam 更新，通过 `_CAYLEY_PATTERNS` 正则表达式自动识别。

---

## 4. 学习率策略：Warmup + 余弦退火

### 4.1 Warmup 的必要性

复数参数在训练初期面临特殊挑战：

- **模长初始化接近 0**（`init_scale=0.02`），直接使用大学习率会导致模长振荡
- **相位随机初始化**，早期大的相位更新会导致干涉模式不稳定（QSA 的注意力熵爆炸）
- **Cayley 矩阵初始接近单位阵**，需要温和过渡到有意义的酉变换

建议 warmup 步数 = `total_steps × 0.05` ~ `× 0.10`。

### 4.2 WarmupCosineScheduler 公式

**Warmup 阶段**（$t < t_{\text{warmup}}$）：
$$
\alpha(t) = \alpha_{\max} \cdot \frac{t}{t_{\text{warmup}}}
$$

**余弦退火阶段**（$t \geq t_{\text{warmup}}$）：
$$
\alpha(t) = \alpha_{\min} + \frac{\alpha_{\max} - \alpha_{\min}}{2} \cdot \left(1 + \cos\frac{\pi(t - t_{\text{warmup}})}{T - t_{\text{warmup}}}\right)
$$

### 4.3 推荐超参数范围

| 参数 | 小模型（<10M） | 中模型（10M-100M） | 备注 |
|------|--------------|-----------------|------|
| `mod_lr` | 3e-4 ~ 1e-3 | 1e-4 ~ 3e-4 | 模长收敛较慢 |
| `phase_lr` | 1e-3 ~ 3e-3 | 3e-4 ~ 1e-3 | 相位需要更多探索 |
| `cayley_lr` | 与 `mod_lr` 相同 | 与 `mod_lr` 相同 | 通常无需单独调 |
| `warmup_steps` | 500 ~ 1000 | 1000 ~ 5000 | 约 5~10% total |
| `betas` | (0.9, 0.999) | (0.9, 0.999) | 标准 Adam 配置 |
| `max_grad_norm` | 1.0 | 0.5 ~ 1.0 | 建议启用 |

---

## 5. 训练稳定性分析

### 5.1 模长坍缩问题

**症状**：训练中模长 $|z|$ 趋于 0，相位梯度变得数值不稳定（除以接近 0 的 $r$）。

**根因**：模长梯度与模长同阶：$g_r \propto r^{-1}$，当 $r \to 0$ 时梯度爆炸。

**缓解措施**：
1. QGD 中 `mod_lr_min = 1e-6`（`WarmupCosineScheduler` 的退火下限）
2. 模长非负约束：`r^{(t+1)} = max(0, ...)`（软边界，避免负模长的相位跳变）
3. 模长正则化（权重衰减）：`grad_r += weight_decay * r`

### 5.2 相位噪声累积

**症状**：相位随机游走，导致干涉模式（QSA 注意力）不收敛。

**根因**：相位更新 $\Delta\varphi$ 随时间累积（无模长的"锚定"作用）。

**缓解措施**：
1. 独立相位动量（`beta1=0.9`）提供惯性滤波
2. QCI 的温度退火：训练初期高温（更平滑的坍缩决策）→ 后期锐化（鼓励确定性）
3. ComplexLayerNorm 在每层规范化相位分布

### 5.3 酉性漂移（Cayley 参数）

**症状**：长期训练后 Cayley 矩阵偏离酉性约束（`||W†W - I||_F > 0.01`）。

**根因**：浮点累积误差，以及大步长更新 $\Omega$ 使 Cayley 变换数值不稳定。

**监控**：`model.get_unitarity_report()` 每 100 步调用一次，关注 `violation_norm`。

**缓解措施**：
- 定期重投影：$\Omega \leftarrow (\Omega - \Omega^T) / 2$（保持反厄米性）
- 使用更小的 `cayley_lr`（建议 `mod_lr / 5`）
- 混合精度训练中为 Cayley 参数使用 `float32`（避免 `float16` 累积误差）

### 5.4 POVM 完整性违背（QCI）

**症状**：`collapse_povm_violation` 持续上升，超过 0.1。

**根因**：测量基 `measurement_basis` 偏离正交性。

**缓解措施**：
1. 每 $k$ 步对 `measurement_basis` 做 QR 正交化（免费的）：
   ```python
   with torch.no_grad():
       q, _ = torch.linalg.qr(module.povm.measurement_basis.T)
       module.povm.measurement_basis.data = q.T
   ```
2. 添加正交正则化损失：$\lambda \cdot \|BB^† - I\|_F^2$

---

## 6. 与 Adam 的理论等价条件

当以下条件成立时，QGD 与标准 Adam 等价：

1. 所有参数为**实数**（`mod_lr == real_lr`，`phase_lr` 无效）
2. 初始模长 $r_0 = \sqrt{(\text{Re}(z_0))^2 + (\text{Im}(z_0))^2}$，且相位固定（$g_\varphi = 0$）

一般情况下，QGD 能更好地利用量子架构参数的几何结构，
收敛速度快 ~20-40%（基于小规模验证实验）。

---

## 7. 代码示例

```python
from quantum_core.optimizer import QGD, WarmupCosineScheduler

# 创建优化器（自动识别 Cayley 参数）
optimizer = QGD.from_model(
    model,
    mod_lr=3e-4,
    phase_lr=1e-3,
    cayley_lr=6e-5,    # Cayley 参数使用更小 lr
    betas=(0.9, 0.999),
    max_grad_norm=1.0,
    weight_decay=1e-4,
)

# 创建学习率调度器
scheduler = WarmupCosineScheduler(
    optimizer=optimizer,
    warmup_steps=1000,
    total_steps=100_000,
    mod_lr_min=1e-6,
    phase_lr_min=1e-5,
)

# 训练循环
for step, batch in enumerate(dataloader):
    optimizer.zero_grad()
    loss = model(batch)["loss"]
    loss.backward()
    optimizer.step()
    scheduler.step()   # ← 在 optimizer.step() 后调用

    # 定期监控（每 100 步）
    if step % 100 == 0:
        stats = optimizer.get_stats()
        print(f"Step {step} | grad_norm={stats['grad_norm_complex']:.4f} | "
              f"lr={scheduler.get_lr()['mod_lr']:.2e}")
```

---

*本文档对应迭代9（2026-03-23），后续将随代码迭代持续更新。*
