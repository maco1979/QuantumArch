# 自动化优化系统 - 使用指南

**量子架构项目 - 自主优化架构师**

---

## 📋 目录

- [系统概述](#系统概述)
- [快速开始](#快速开始)
- [核心功能](#核心功能)
- [配置指南](#配置指南)
- [API文档](#api文档)
- [监控仪表板](#监控仪表板)
- [故障排查](#故障排查)
- [最佳实践](#最佳实践)

---

## 🎯 系统概述

自动化优化系统为量子架构提供以下能力：

1. **实时性能监控**：跟踪7大核心机制的运行效率
2. **影子A/B测试**：安全测试新算法变体
3. **成本智能控制**：防止GPU资源耗尽和API调用超支
4. **自动故障恢复**：熔断器机制保护系统稳定性
5. **模型自适应优化**：基于数据特征自动调整超参数

### 预期收益

- 训练成本降低 **40-60%**
- 推理速度提升 **3-10倍**
- GPU利用率提升至 **85%+**
- 故障自动恢复率 **95%+**

---

## 🚀 快速开始

### 1. 安装依赖

```bash
# 基础依赖
pip install torch numpy pyyaml

# 监控依赖
pip install prometheus_client pynvml

# 可选：统计计算
pip install scipy

# 可选：超参数优化
pip install optuna
```

### 2. 配置系统

编辑 `config.yaml` 文件，根据你的环境调整配置：

```yaml
cost_control:
  gpu:
    cost_per_hour: 3.0  # 你的GPU每小时成本
    daily_budget: 100.0  # 每日预算

monitoring:
  prometheus:
    port: 8000  # Prometheus端口
```

### 3. 运行示例

```bash
# 基础演示
python optimization_system/training_pipeline.py

# 只运行优化引擎
python -m optimization_system.optimization_engine

# 启动Prometheus导出器
python -m optimization_system.metrics_exporter
```

### 4. 查看监控

访问 Grafana 仪表板（需要先配置 Grafana）：

```
http://localhost:3000
```

---

## 🔧 核心功能

### 1. 性能监控

```python
from optimization_engine import PerformanceMonitor, PerformanceMetrics

# 创建监控器
monitor = PerformanceMonitor(window_size=1000)

# 记录性能指标
metrics = PerformanceMetrics(
    loss=2.5,
    accuracy=0.85,
    grad_norm=1.2,
    batch_time=0.15,
    gpu_utilization=0.8,
    gpu_memory_used=10.5,
    qsa_topk_ratio=0.1,
    qci_early_exit_rate=0.3,
    qgd_mod_lr=1e-4,
    qgd_phase_lr=1e-3,
    unitarity_violation=1e-8,
)

monitor.record(metrics)

# 获取统计信息
stats = monitor.get_statistics('loss', last_n=100)
print(f"平均损失: {stats['mean']:.4f}")
print(f"标准差: {stats['std']:.4f}")
```

### 2. 成本控制

```python
from optimization_engine import GPUCostMonitor, APICostGuard

# GPU成本监控
cost_monitor = GPUCostMonitor(
    gpu_cost_per_hour=3.0,
    daily_budget=100.0
)

# 检查预算
budget_status = cost_monitor.check_budget(
    estimated_remaining_steps=10000
)

if budget_status['action'] == 'continue':
    print(f"预算正常，已使用: ${budget_status['current_spent']:.2f}")
elif budget_status['action'] == 'downgrade':
    print("预算警告，需要降低资源使用")

# API成本防护
api_guard = APICostGuard(max_cost_per_hour=5.0)

should_call, reason = api_guard.should_call('external_api', time.time())

if should_call:
    # 调用API
    result = call_external_api()
    api_guard.record_call('external_api', success=True)
else:
    print(f"API调用被阻止: {reason}")
```

### 3. 影子测试

```python
from optimization_engine import ShadowTestEngine

# 创建影子测试引擎
shadow_engine = ShadowTestEngine(shadow_ratio=0.05)

# 记录基线结果
shadow_engine.add_baseline_result(loss=2.5, accuracy=0.85)

# 记录变体结果
shadow_engine.add_variant_result('variant_A', loss=2.4, accuracy=0.86)
shadow_engine.add_variant_result('variant_B', loss=2.3, accuracy=0.87)

# 检查是否应该提升
should_promote, reason = shadow_engine.should_promote('variant_A')

if should_promote:
    print(f"✓ variant_A 应该晋升: {reason}")
else:
    print(f"✗ variant_A 暂不晋升: {reason}")
```

### 4. QSA自适应控制

```python
from optimization_engine import AdaptiveQSAController

# 创建QSA控制器
qsa_controller = AdaptiveQSAController(
    base_ratio=0.1,
    bounds=(0.05, 0.3)
)

# 更新筛选比例
new_ratio = qsa_controller.update_ratio(
    accuracy_drop=0.001,  # 准确率下降0.1%
    speedup=3.0  # 速度提升3倍
)

print(f"新的QSA筛选比例: {new_ratio:.3f}")
```

### 5. QCI阈值学习

```python
from optimization_engine import CollapseThresholdLearner

# 创建坍缩阈值学习器
threshold_learner = CollapseThresholdLearner()

# 前向传播获取阈值
batch_stats = {
    'avg_entropy': 1.2,
    'entropy_std': 0.3,
    'early_exit_rate': 0.4,
}

tau_low, tau_high = threshold_learner.forward(batch_stats)
print(f"坍缩阈值: low={tau_low:.3f}, high={tau_high:.3f}")

# 根据奖励更新策略
threshold_learner.update(
    batch_stats=batch_stats,
    speedup=2.5,
    accuracy_loss=0.002
)
```

### 6. 异常检测与恢复

```python
from optimization_engine import AnomalyDetector, AutoRecovery

# 创建异常检测器
detector = AnomalyDetector()

# 记录指标
detector.record('loss', 2.5)
detector.record('grad_norm', 1.2)

# 检测异常
current_metrics = {
    'loss': 10.0,  # 突然增大
    'grad_norm': 1.5,
    'batch_time': 0.15,
    'unitarity_violation': 1e-5,
}

anomalies = detector.check_anomalies(current_metrics)

if anomalies:
    print(f"检测到异常: {anomalies}")

    # 自动恢复
    recovery = AutoRecovery()
    for anomaly in anomalies:
        action = recovery.recover(anomaly)
        print(f"恢复策略: {action}")
```

---

## ⚙️ 配置指南

### 核心配置文件结构

```yaml
# config.yaml

system:
  name: "QuantumArch Optimization System"
  log_level: "INFO"

cost_control:
  gpu:
    cost_per_hour: 3.0
    daily_budget: 100.0

  api:
    max_cost_per_hour: 5.0
    circuit_breaker:
      enabled: true

shadow_testing:
  enabled: true
  shadow_ratio: 0.05

qsa:
  adaptive_topk:
    enabled: true
    base_ratio: 0.1
    bounds: [0.05, 0.3]

qci:
  thresholds:
    tau_low_initial: 0.5
    tau_high_initial: 1.5
    adaptive: true

qgd:
  learning_rates:
    mod_base: 1e-4
    phase_base: 1e-3
    adaptive: true

anomaly_detection:
  enabled: true
  window_size: 100

auto_recovery:
  enabled: true

checkpointing:
  enabled: true
  save_interval: 1000
  directory: "./checkpoints"

alerts:
  enabled: true
```

### 环境变量覆盖

可以通过环境变量覆盖配置：

```bash
export QUANTUM_ARCH_LOG_LEVEL=DEBUG
export QUANTUM_ARCH_GPU_COST_PER_HOUR=5.0
export QUANTUM_ARCH_DAILY_BUDGET=200.0
```

---

## 📊 监控仪表板

### Prometheus配置

创建 `prometheus.yml`:

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'quantum_arch_metrics'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
```

启动Prometheus:

```bash
prometheus --config.file=prometheus.yml
```

### Grafana Dashboard

导入预配置的仪表板（`monitoring/grafana_dashboards/`）：

1. 打开 Grafana: http://localhost:3000
2. 添加 Prometheus 数据源
3. 导入 Dashboard JSON

**关键指标**：
- `quantum_arch_training_loss` - 训练损失
- `quantum_arch_qsa_topk_ratio` - QSA筛选比例
- `quantum_arch_qci_early_exit_rate` - QCI早退率
- `quantum_arch_gpu_utilization` - GPU利用率
- `quantum_arch_daily_cost` - 每日成本

---

## 🐛 故障排查

### 常见问题

#### 1. Prometheus无法连接

**问题**: 无法访问 http://localhost:8000/metrics

**解决方案**:
```python
# 检查端口是否被占用
import socket
s = socket.socket()
try:
    s.bind(('localhost', 8000))
    print("端口可用")
except:
    print("端口被占用，请修改config.yaml中的port")
```

#### 2. GPU内存不足

**问题**: `CUDA out of memory`

**解决方案**:
```yaml
# 在config.yaml中调整
cost_control:
  budget_exceeded:
    downgrade_strategies:
      - "reduce_batch_size"  # 降低batch size
```

#### 3. 成本超支告警

**问题**: 频繁触发预算超支告警

**解决方案**:
```python
# 检查成本估算是否准确
cost_monitor = GPUCostMonitor()
cost_monitor.check_budget(estimated_remaining_steps=actual_remaining)

# 调整预算或成本参数
```

#### 4. 影子测试结果不稳定

**问题**: shadow test结果波动大

**解决方案**:
```python
# 增加样本数量
shadow_engine = ShadowTestEngine(shadow_ratio=0.1)  # 从0.05增加到0.1

# 调整晋升准则
shadow_engine.significance_level = 0.01  # 更严格的显著性
```

---

## 💡 最佳实践

### 1. 分阶段启用优化

```python
# 第1周：只启用监控
config['shadow_testing']['enabled'] = False
config['auto_recovery']['enabled'] = False

# 第2周：启用影子测试
config['shadow_testing']['enabled'] = True

# 第3周：启用自动恢复
config['auto_recovery']['enabled'] = True
```

### 2. 监控关键指标

始终关注以下指标：

1. **成本指标**: `daily_cost`, `hourly_cost`
2. **性能指标**: `training_loss`, `validation_loss`, `accuracy`
3. **效率指标**: `gpu_utilization`, `throughput`
4. **稳定性指标**: `unitarity_violation`, `grad_norm`

### 3. 定期检查告警

```bash
# 设置告警通知
alerts:
  channels:
    - type: "webhook"
      url: "https://hooks.slack.com/services/XXX"
```

### 4. 版本控制配置

```bash
# 将config.yaml加入Git
git add config.yaml
git commit -m "Add optimization config"

# 本地覆盖可以使用环境变量
export QUANTUM_ARCH_GPU_COST_PER_HOUR=5.0
```

### 5. 备份重要checkpoint

```python
# 定期备份最佳模型
if val_loss < best_loss:
    pipeline.save_checkpoint('best_model')
    # 额外备份到云存储
    upload_to_cloud('best_model.pt')
```

---

## 📚 API文档

### OptimizationSystem

主控制器，整合所有优化模块。

**方法**:

- `on_training_step(metrics: PerformanceMetrics)` - 每个训练步调用
- `optimize_every_100_steps()` - 每100步执行的优化
- `optimize_every_1000_steps()` - 每1000步执行的深度优化
- `on_api_call(api_name: str, success: bool)` - API调用前后
- `get_system_status()` - 获取系统状态

### QuantumArchMetrics

Prometheus指标导出器。

**方法**:

- `start()` - 启动HTTP服务器
- `update_metrics(metrics: Dict)` - 更新所有指标
- `record_anomaly(anomaly_type: str)` - 记录异常
- `record_checkpoint()` - 记录checkpoint保存
- `record_shadow_test(variant_name: str, improvement: float)` - 记录影子测试
- `get_gpu_info()` - 获取GPU信息

### PerformanceMonitor

性能监控器。

**方法**:

- `record(metrics: PerformanceMetrics)` - 记录性能指标
- `get_statistics(metric_name: str, last_n: int)` - 获取统计信息
- `detect_trend(metric_name: str, threshold: float)` - 检测趋势

---

## 🤝 贡献指南

欢迎贡献代码、报告问题或提出改进建议！

---

## 📄 许可证

本项目采用 MIT 许可证。

---

*© 2026 量子架构项目组 | 自动化优化系统 v1.0*
