# automation-3 执行历史

## 2026-03-26 08:00 — 第4次执行

**状态**: 完成（10/10次迭代）

**改动摘要**:
1. **QSA 诊断**: `attention.py` 新增 `get_attention_patterns()` — Born概率/干涉相位/注意力熵/TopK掩码
2. **QFT 逆变换**: `entanglement.py` 新增 `inverse()` IQFT — 支持编解码器架构
3. **QCI 阈值改进**: `collapse.py` AdaptiveThreshold tau_high 后期衰减 + `get_threshold_summary()`
4. **QGD 参数组**: `optimizer.py` 新增 `per_group_lr_state()` — 多参数组LR监控
5. **FFN 门控监控**: `ffn.py` GatedQuantumFFN 新增 `get_gate_statistics()` — 死门检测
6. **量子度量集**: 新建 `quantum_metrics.py`（250行）— 9个度量函数 + compute_model_quantum_health
7. **推理接口**: `model.py` 新增 `inference()` — 自动eval切换+量子健康度诊断
8. **酉性恢复**: `unitary.py` CayleyLinear 新增 `recover_unitarity()` — QR/rescale双方法
9. **QCI 理论文档**: 新建 `docs/core/qci_theory.md`（300行）— POVM形式主义+超参数指南
10. **单元测试**: 新建 `tests/test_march26_iterations.py`（8类46测试）

**报告文件**: `.workbuddy/memory/iteration-report-2026-03-26.md`

---

## 2026-03-25 08:00 — 第3次执行

**状态**: 完成（10/10次迭代）

**改动摘要**:
1. **QSA 因果掩码**: `attention.py` forward() 新增 `causal` 参数，支持自回归语言模型
2. **QEL 度量导出**: `entanglement.py` 新增 `get_entanglement_metrics()` — 5个纠缠质量指标
3. **POVM 正交正则化**: `collapse.py` 新增 `orthogonality_regularization_loss()` + `renormalize_basis()`
4. **QGD EMA 跟踪**: `optimizer.py` 增强 `get_stats()` + 新增 `grad_norm_ema()` 方法
5. **量子信息三件套**: `complex_ops.py` 新增 `quantum_fidelity`/`trace_distance`/`quantum_relative_entropy`
6. **FFN SwiGLU 集成**: `ffn.py` 正式支持 `activation="swiglu"` 选项（双重门控防护）
7. **QIR 独立模块**: 新建 `interference_router.py`（310行）— pairwise_interference + QuantumInterferenceRouter
8. **模型参数统计增强**: `model.py` count_parameters() 新增子模块分类+内存估算字段
9. **QEL 理论文档**: 新建 `docs/core/qel_theory.md`（260行）
10. **单元测试**: 新建 `tests/test_march25_iterations.py`（8类38测试）

**报告文件**: `.workbuddy/memory/iteration-report-2026-03-25.md`

---

## 2026-03-23 08:00 — 第1次执行

**状态**: 完成（10/10次迭代）

**改动摘要**:
1. **QSA Bug修复**: 修复推理阶段 topk 不生效（`and training` 条件错误）
2. **complex_ops 新增**: `phase_coherence()` 和 `phase_gradient()` 两个相位度量工具
3. **QGD 调度器**: 新增 `WarmupCosineScheduler`（Warmup + 余弦退火，独立模长/相位调度）
4. **QCI 温度退火**: 硬编码 `temperature=5.0` 改为基于步数的动态退火（2.0 → 10.0）
5. **QEL 边界处理**: 奇数序列长度时末尾 token 增加额外边界纠缠
6. **主模型改进**: early_exit key 注释优化 + 输出投影从 `z.real` 改为 `z.abs()`
7. **ComplexLayerNorm**: 新增 `scale_by_magnitude` 参数（量子态单位球约束）
8. **ComplexSwiGLU**: 新激活函数（SwiGLU 的复数域推广）
9. **QGD 文档**: 新建 `docs/core/qgd_math_and_stability.md`（Wirtinger 推导、稳定性分析）
10. **梯度检查点**: QuantumBlock 新增 `use_checkpoint` 支持（节省50%显存）

**报告文件**: `.workbuddy/memory/iteration-report-2026-03-23.md`

