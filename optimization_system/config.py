"""
自动化优化系统 - 配置文件
"""

import yaml
from typing import Dict, Any
import os


# ============================================================================
# 优化系统配置
# ============================================================================


class OptimizationConfig:
    """优化系统配置类"""

    def __init__(self, config_path: str = None):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if self.config_path and os.path.exists(self.config_path):
            with open(self.config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        else:
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """默认配置"""
        return {
            # 性能监控配置
            "monitoring": {
                "window_size": 1000,
                "save_interval": 100,
                "metrics_path": "./monitoring/metrics/",
            },
            # 成本控制配置
            "cost_control": {
                "gpu": {
                    "cost_per_hour": 3.0,
                    "daily_budget": 100.0,
                    "monthly_budget": 2000.0,
                },
                "api": {
                    "max_cost_per_hour": 5.0,
                    "failure_threshold": 5,
                    "recovery_time": 300,
                },
            },
            # 影子测试配置
            "shadow_testing": {
                "shadow_ratio": 0.05,
                "significance_level": 0.05,
                "min_samples": 100,
                "promotion_threshold": 0.03,  # 3%改进
            },
            # QSA自适应配置
            "qsa_adaptive": {
                "base_ratio": 0.1,
                "bounds": [0.05, 0.3],
                "update_frequency": 100,
            },
            # QCI坍缩配置
            "qci_collapse": {
                "initial_tau_low": 0.5,
                "initial_tau_high": 2.0,
                "learning_rate": 0.01,
                "update_frequency": 1000,
            },
            # 学习率配置
            "learning_rate": {
                "base_mod_lr": 1e-4,
                "base_phase_lr": 1e-3,
                "adapt_frequency": 100,
            },
            # 异常检测配置
            "anomaly_detection": {
                "loss_explosion_threshold": 3.0,  # 相对于基线
                "grad_explosion_threshold": 10.0,
                "grad_vanishing_threshold": 1e-7,
                "performance_degradation_threshold": 2.0,  # 相对于基线
                "unitarity_violation_threshold": 1e-4,
            },
            # 自动恢复配置
            "auto_recovery": {
                "enabled": True,
                "max_recovery_attempts": 3,
                "checkpoint_dir": "./checkpoints/",
            },
            # 告警配置
            "alerts": {
                "enabled": True,
                "channels": ["log", "email", "slack"],
                "email_recipients": ["admin@example.com"],
                "slack_webhook": "https://hooks.slack.com/services/...",
            },
            # 日志配置
            "logging": {
                "level": "INFO",
                "file": "./logs/optimization.log",
                "rotation": "1d",
                "retention": "30d",
            },
        }

    def get(self, key_path: str, default=None):
        """获取配置值，支持点分隔的路径"""
        keys = key_path.split(".")
        value = self.config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def set(self, key_path: str, value: Any):
        """设置配置值"""
        keys = key_path.split(".")
        config = self.config

        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        config[keys[-1]] = value

    def save(self, save_path: str = None):
        """保存配置到文件"""
        path = save_path or self.config_path
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(self.config, f, allow_unicode=True, default_flow_style=False)
        else:
            raise ValueError("未指定保存路径")

    def __str__(self) -> str:
        return yaml.dump(self.config, allow_unicode=True, default_flow_style=False)


# ============================================================================
# Prometheus配置
# ============================================================================

PROMETHEUS_CONFIG = """
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'quantum_arch_metrics'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'node_exporter'
    static_configs:
      - targets: ['localhost:9100']
"""


# ============================================================================
# 告警规则配置
# ============================================================================

ALERTS_CONFIG = """
groups:
  - name: quantum_arch_alerts
    interval: 30s
    rules:
      # 损失爆炸告警
      - alert: LossExplosion
        expr: |
          training_loss > 10 * avg_over_time(training_loss[1h])
        for: 5m
        labels:
          severity: critical
          component: training
        annotations:
          summary: "训练损失爆炸"
          description: "当前损失 {{ $value }} 远高于历史平均水平"

      # GPU利用率低告警
      - alert: LowGPUUtilization
        expr: gpu_utilization < 50
        for: 10m
        labels:
          severity: warning
          component: hardware
        annotations:
          summary: "GPU利用率过低"
          description: "GPU利用率仅为 {{ $value }}%，考虑增加batch_size"

      # GPU利用率高告警
      - alert: HighGPUUtilization
        expr: gpu_utilization > 95
        for: 5m
        labels:
          severity: warning
          component: hardware
        annotations:
          summary: "GPU利用率接近饱和"
          description: "GPU利用率达到 {{ $value }}%，可能需要优化"

      # 显存不足告警
      - alert: HighGPUMemory
        expr: |
          (gpu_memory_used / gpu_memory_total) > 0.9
        for: 5m
        labels:
          severity: critical
          component: hardware
        annotations:
          summary: "显存不足"
          description: "显存使用率达到 {{ $value }}%，请检查内存泄漏"

      # 梯度爆炸告警
      - alert: GradientExplosion
        expr: grad_norm > 10
        for: 1m
        labels:
          severity: critical
          component: training
        annotations:
          summary: "梯度爆炸"
          description: "梯度范数达到 {{ $value }}，需要梯度裁剪"

      # 梯度消失告警
      - alert: GradientVanishing
        expr: grad_norm < 1e-7
        for: 5m
        labels:
          severity: warning
          component: training
        annotations:
          summary: "梯度消失"
          description: "梯度范数过低 ({{ $value }})，考虑调整网络结构"

      # 酉约束违背告警
      - alert: UnitarityViolation
        expr: unitarity_violation > 1e-4
        for: 1m
        labels:
          severity: critical
          component: model
        annotations:
          summary: "酉约束严重违背"
          description: "酉约束违背度达到 {{ $value }}，需要重新参数化"

      # 训练速度下降告警
      - alert: TrainingSlowdown
        expr: |
          batch_time > 2 * avg_over_time(batch_time[1h])
        for: 10m
        labels:
          severity: warning
          component: performance
        annotations:
          summary: "训练速度显著下降"
          description: "批处理时间增加到 {{ $value }}秒，历史平均为 {{ $labels.avg_time }}秒"

      # 成本超支警告
      - alert: BudgetExceeded
        expr: daily_cost > daily_budget
        for: 1m
        labels:
          severity: critical
          component: cost
        annotations:
          summary: "预算超支"
          description: "今日花费 {{ $value }}美元已超过预算 {{ $labels.budget }}美元"

      # API调用失败率高
      - alert: HighAPIFailureRate
        expr: |
          rate(api_failures[5m]) / rate(api_calls[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
          component: api
        annotations:
          summary: "API调用失败率过高"
          description: "API调用失败率为 {{ $value | humanizePercentage }}"

      # QSA筛选比例异常
      - alert: AbnormalQSARatio
        expr: |
          qsa_topk_ratio < 0.05 or qsa_topk_ratio > 0.3
        for: 5m
        labels:
          severity: warning
          component: model
        annotations:
          summary: "QSA筛选比例异常"
          description: "当前QSA筛选比例为 {{ $value }}，超出正常范围"

      # 早退率异常
      - alert: AbnormalEarlyExitRate
        expr: |
          qci_early_exit_rate < 0.2 or qci_early_exit_rate > 0.5
        for: 10m
        labels:
          severity: warning
          component: model
        annotations:
          summary: "早退率异常"
          description: "当前早退率为 {{ $value }}，可能需要调整坍缩阈值"
"""


# ============================================================================
# Grafana Dashboard配置
# ============================================================================

GRAFANA_DASHBOARD_CONFIG = {
    "dashboard": {
        "title": "量子架构 - 自动化优化仪表板",
        "uid": "quantum-arch-optimization",
        "description": "实时监控量子架构训练和优化状态",
        "tags": ["quantum", "optimization", "monitoring"],
        "timezone": "browser",
        "refresh": "5s",
        "panels": [
            # 训练指标
            {
                "id": 1,
                "title": "训练损失",
                "type": "timeseries",
                "targets": [
                    {"expr": "training_loss", "legendFormat": "训练损失"},
                    {"expr": "validation_loss", "legendFormat": "验证损失"},
                ],
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
            },
            {
                "id": 2,
                "title": "准确率",
                "type": "timeseries",
                "targets": [
                    {"expr": "training_accuracy", "legendFormat": "训练准确率"},
                    {"expr": "validation_accuracy", "legendFormat": "验证准确率"},
                ],
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
            },
            # 硬件监控
            {
                "id": 3,
                "title": "GPU利用率",
                "type": "gauge",
                "targets": [{"expr": "gpu_utilization", "legendFormat": "利用率"}],
                "fieldConfig": {
                    "defaults": {
                        "unit": "percent",
                        "min": 0,
                        "max": 100,
                        "thresholds": {
                            "steps": [
                                {"value": 0, "color": "green"},
                                {"value": 70, "color": "yellow"},
                                {"value": 95, "color": "red"},
                            ]
                        },
                    }
                },
                "gridPos": {"h": 6, "w": 6, "x": 0, "y": 8},
            },
            {
                "id": 4,
                "title": "显存使用",
                "type": "gauge",
                "targets": [{"expr": "gpu_memory_used", "legendFormat": "已用显存"}],
                "fieldConfig": {
                    "defaults": {
                        "unit": "bytes",
                        "thresholds": {
                            "steps": [
                                {"value": 0, "color": "green"},
                                {"value": 8000000000, "color": "yellow"},
                                {"value": 10000000000, "color": "red"},
                            ]
                        },
                    }
                },
                "gridPos": {"h": 6, "w": 6, "x": 6, "y": 8},
            },
            # QSA/QCI指标
            {
                "id": 5,
                "title": "QSA筛选比例",
                "type": "timeseries",
                "targets": [{"expr": "qsa_topk_ratio", "legendFormat": "筛选比例"}],
                "gridPos": {"h": 6, "w": 6, "x": 12, "y": 8},
            },
            {
                "id": 6,
                "title": "QCI早退率",
                "type": "timeseries",
                "targets": [{"expr": "qci_early_exit_rate", "legendFormat": "早退率"}],
                "gridPos": {"h": 6, "w": 6, "x": 18, "y": 8},
            },
            # 梯度和学习率
            {
                "id": 7,
                "title": "梯度范数",
                "type": "timeseries",
                "targets": [{"expr": "grad_norm", "legendFormat": "梯度范数"}],
                "yaxes": [{"format": "short", "label": "值"}, {"format": "short", "label": "值"}],
                "gridPos": {"h": 6, "w": 12, "x": 0, "y": 14},
            },
            {
                "id": 8,
                "title": "学习率",
                "type": "timeseries",
                "targets": [
                    {"expr": "qgd_mod_lr", "legendFormat": "模长学习率"},
                    {"expr": "qgd_phase_lr", "legendFormat": "相位学习率"},
                ],
                "yaxes": [
                    {"format": "scientific", "label": "学习率"},
                    {"format": "scientific", "label": "学习率"},
                ],
                "gridPos": {"h": 6, "w": 12, "x": 12, "y": 14},
            },
            # 成本监控
            {
                "id": 9,
                "title": "每小时成本",
                "type": "stat",
                "targets": [{"expr": "hourly_cost", "legendFormat": "成本"}],
                "fieldConfig": {"defaults": {"unit": "currencyUSD", "decimals": 2}},
                "gridPos": {"h": 4, "w": 6, "x": 0, "y": 20},
            },
            {
                "id": 10,
                "title": "每日预算使用率",
                "type": "stat",
                "targets": [
                    {"expr": "(daily_cost / daily_budget) * 100", "legendFormat": "使用率"}
                ],
                "fieldConfig": {
                    "defaults": {
                        "unit": "percent",
                        "decimals": 1,
                        "thresholds": {
                            "steps": [
                                {"value": 0, "color": "green"},
                                {"value": 80, "color": "yellow"},
                                {"value": 100, "color": "red"},
                            ]
                        },
                    }
                },
                "gridPos": {"h": 4, "w": 6, "x": 6, "y": 20},
            },
            # 酉约束违背
            {
                "id": 11,
                "title": "酉约束违背度",
                "type": "timeseries",
                "targets": [{"expr": "unitarity_violation", "legendFormat": "违背度"}],
                "yaxes": [
                    {"format": "scientific", "label": "值"},
                    {"format": "scientific", "label": "值"},
                ],
                "gridPos": {"h": 6, "w": 12, "x": 12, "y": 20},
            },
            # 批处理时间
            {
                "id": 12,
                "title": "批处理时间",
                "type": "timeseries",
                "targets": [{"expr": "batch_time", "legendFormat": "时间"}],
                "fieldConfig": {"defaults": {"unit": "s"}},
                "gridPos": {"h": 6, "w": 12, "x": 0, "y": 26},
            },
            # 影子测试结果
            {
                "id": 13,
                "title": "影子测试改进率",
                "type": "timeseries",
                "targets": [{"expr": "shadow_test_improvement", "legendFormat": "{{variant}}"}],
                "fieldConfig": {"defaults": {"unit": "percent"}},
                "gridPos": {"h": 6, "w": 12, "x": 12, "y": 26},
            },
        ],
    }
}


# ============================================================================
# 导出配置到文件
# ============================================================================

if __name__ == "__main__":
    # 创建配置目录
    os.makedirs("./configs", exist_ok=True)

    # 保存默认配置
    config = OptimizationConfig()
    config.save("./configs/optimization.yaml")

    # 保存Prometheus配置
    with open("./configs/prometheus.yml", "w") as f:
        f.write(PROMETHEUS_CONFIG.strip())

    # 保存告警规则
    with open("./configs/alerts.yml", "w") as f:
        f.write(ALERTS_CONFIG.strip())

    # 保存Grafana dashboard
    import json

    with open("./configs/grafana_dashboard.json", "w") as f:
        json.dump(GRAFANA_DASHBOARD_CONFIG, f, indent=2)

    print("配置文件已生成:")
    print("  - ./configs/optimization.yaml")
    print("  - ./configs/prometheus.yml")
    print("  - ./configs/alerts.yml")
    print("  - ./configs/grafana_dashboard.json")
