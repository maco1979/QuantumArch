# automation-3 执行历史

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
