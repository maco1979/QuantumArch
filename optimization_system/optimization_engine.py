"""
自动化优化系统 - 核心实现框架
QuantumArch Optimization System v1.0
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import time
import logging
from dataclasses import dataclass, field
from collections import defaultdict
import json
from enum import Enum

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# 1. 性能监控模块
# ============================================================================


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""

    loss: float
    accuracy: float
    grad_norm: float
    batch_time: float
    gpu_utilization: float
    gpu_memory_used: float
    qsa_topk_ratio: float
    qci_early_exit_rate: float
    qgd_mod_lr: float
    qgd_phase_lr: float
    unitarity_violation: float
    timestamp: float = field(default_factory=time.time)


class PerformanceMonitor:
    """实时性能监控器"""

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.metrics_history: List[PerformanceMetrics] = []

    def record(self, metrics: PerformanceMetrics):
        """记录性能指标"""
        self.metrics_history.append(metrics)

        # 保持窗口大小
        if len(self.metrics_history) > self.window_size:
            self.metrics_history.pop(0)

    def get_statistics(self, metric_name: str, last_n: int = 100) -> Dict[str, float]:
        """获取指定指标的统计信息"""
        recent = self.metrics_history[-last_n:]
        values = [getattr(m, metric_name) for m in recent if hasattr(m, metric_name)]

        if not values:
            return {}

        return {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "median": np.median(values),
            "p95": np.percentile(values, 95),
            "p99": np.percentile(values, 99),
        }

    def detect_trend(self, metric_name: str, threshold: float = 0.05) -> str:
        """检测指标趋势"""
        if len(self.metrics_history) < 50:
            return "insufficient_data"

        recent = [getattr(m, metric_name) for m in self.metrics_history[-20:]]
        earlier = [getattr(m, metric_name) for m in self.metrics_history[-50:-20]]

        recent_mean = np.mean(recent)
        earlier_mean = np.mean(earlier)
        change_rate = (recent_mean - earlier_mean) / (earlier_mean + 1e-8)

        if change_rate > threshold:
            return "increasing"
        elif change_rate < -threshold:
            return "decreasing"
        else:
            return "stable"


# ============================================================================
# 2. 成本控制模块
# ============================================================================


class CircuitBreakerState(Enum):
    """熔断器状态"""

    CLOSED = "closed"  # 正常工作
    OPEN = "open"  # 熔断（停止调用）
    HALF_OPEN = "half_open"  # 半开（尝试恢复）


class APICostGuard:
    """API调用成本防护熔断器"""

    def __init__(self, max_cost_per_hour: float = 5.0):
        self.max_cost = max_cost_per_hour
        self.hourly_calls = defaultdict(int)
        self.cost_per_call = 0.01  # 假设每次调用成本
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.failure_threshold = 5
        self.recovery_time = 300  # 5分钟后尝试恢复

    def should_call(self, api_name: str, current_time: float) -> Tuple[bool, str]:
        """
        检查是否允许调用API

        返回: (是否允许, 原因)
        """
        # 检查熔断器状态
        if self.state == CircuitBreakerState.OPEN:
            # 检查是否到恢复时间
            if time.time() - self.failure_count * self.recovery_time < self.recovery_time:
                return False, "circuit_breaker_open"
            else:
                self.state = CircuitBreakerState.HALF_OPEN
                logger.info(f"熔断器进入半开状态，尝试恢复 {api_name}")

        # 计算当前小时成本
        hour_key = int(current_time // 3600)
        current_cost = self.hourly_calls[hour_key] * self.cost_per_call

        # 检查预算
        if current_cost > self.max_cost * 0.8:  # 80%阈值
            logger.warning(f"API成本即将超限: {current_cost:.2f}/{self.max_cost:.2f}")
            self.state = CircuitBreakerState.OPEN
            return False, "cost_limit_approaching"

        if current_cost < self.max_cost * 0.5 and self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.CLOSED
            logger.info(f"熔断器恢复正常: {api_name}")

        return True, "ok"

    def record_call(self, api_name: str, success: bool, current_time: float):
        """记录API调用结果"""
        hour_key = int(current_time // 3600)
        self.hourly_calls[hour_key] += 1

        if not success:
            self.failure_count += 1
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitBreakerState.OPEN
                logger.error(f"熔断器触发: {api_name}, 连续失败 {self.failure_count} 次")
        else:
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.failure_count = 0  # 重置


class GPUCostMonitor:
    """GPU资源成本监控"""

    def __init__(self, gpu_cost_per_hour: float = 3.0, daily_budget: float = 100.0):
        self.hourly_cost = gpu_cost_per_hour
        self.daily_budget = daily_budget
        self.daily_spent = 0.0
        self.hourly_spent = defaultdict(float)

    def update_spent(self, hours_used: float, current_time: float):
        """更新已花费的成本"""
        cost = hours_used * self.hourly_cost
        day_key = int(current_time // 86400)
        hour_key = int(current_time // 3600)

        self.daily_spent += cost
        self.hourly_spent[hour_key] += cost

        # 每日重置
        if day_key != int((current_time - 3600) // 86400):
            self.daily_spent = cost

    def check_budget(
        self, estimated_remaining_steps: int, steps_per_hour: int = 1000
    ) -> Dict[str, Any]:
        """
        检查预算是否充足

        返回决策建议
        """
        hours_remaining = estimated_remaining_steps / steps_per_hour
        estimated_cost = hours_remaining * self.hourly_cost

        total_cost = self.daily_spent + estimated_cost
        budget_utilization = total_cost / self.daily_budget

        if total_cost > self.daily_budget:
            logger.warning(f"预算超支预测: {total_cost:.2f}/{self.daily_budget:.2f}")
            return {
                "action": "stop",
                "reason": "budget_exceeded",
                "current_spent": self.daily_spent,
                "estimated_remaining": estimated_cost,
                "budget_utilization": budget_utilization,
            }
        elif budget_utilization > 0.9:
            logger.warning(f"预算即将超支: {budget_utilization:.1%}")
            return {
                "action": "downgrade",
                "reason": "budget_warning",
                "current_spent": self.daily_spent,
                "estimated_remaining": estimated_cost,
                "suggestions": [
                    "降低batch_size",
                    "减少数据采样",
                    "提前早退",
                ],
            }
        else:
            return {
                "action": "continue",
                "reason": "budget_ok",
                "current_spent": self.daily_spent,
                "estimated_remaining": estimated_cost,
                "budget_utilization": budget_utilization,
            }


# ============================================================================
# 3. 影子测试引擎
# ============================================================================


@dataclass
class ShadowTestResult:
    """影子测试结果"""

    variant_name: str
    loss: float
    accuracy: float
    improvement: float
    p_value: float
    sample_count: int
    timestamp: float = field(default_factory=time.time)


class ShadowTestEngine:
    """影子测试引擎 - 安全测试新算法变体"""

    def __init__(self, shadow_ratio: float = 0.05, significance_level: float = 0.05):
        self.shadow_ratio = shadow_ratio
        self.significance_level = significance_level
        self.variant_results: Dict[str, List[ShadowTestResult]] = defaultdict(list)
        self.baseline_results: List[ShadowTestResult] = []

    def add_baseline_result(self, loss: float, accuracy: float):
        """添加基线模型结果"""
        result = ShadowTestResult(
            variant_name="baseline",
            loss=loss,
            accuracy=accuracy,
            improvement=0.0,
            p_value=1.0,
            sample_count=len(self.baseline_results) + 1,
        )
        self.baseline_results.append(result)

    def add_variant_result(self, variant_name: str, loss: float, accuracy: float):
        """添加变体模型结果"""
        # 计算相对于基线的改进
        if self.baseline_results:
            baseline_loss = np.mean([r.loss for r in self.baseline_results[-10:]])
            improvement = (baseline_loss - loss) / baseline_loss
        else:
            improvement = 0.0

        # 简化的统计检验（实际应该用t-test）
        variant_history = self.variant_results[variant_name]
        if len(variant_history) > 10:
            improvement_mean = np.mean([r.improvement for r in variant_history[-10:]])
            improvement_std = np.std([r.improvement for r in variant_history[-10:]])
            if improvement_std > 0:
                from scipy import stats

                z_score = improvement / improvement_std
                p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
            else:
                p_value = 1.0
        else:
            p_value = 1.0

        result = ShadowTestResult(
            variant_name=variant_name,
            loss=loss,
            accuracy=accuracy,
            improvement=improvement,
            p_value=p_value,
            sample_count=len(variant_history) + 1,
        )
        self.variant_results[variant_name].append(result)

    def should_promote(self, variant_name: str) -> Tuple[bool, str]:
        """
        判断是否应该提升shadow模型到生产

        返回: (是否提升, 原因)
        """
        if variant_name not in self.variant_results:
            return False, "no_results"

        results = self.variant_results[variant_name]

        # 检查样本数量
        if len(results) < 100:
            return False, f"insufficient_samples ({len(results)}/100)"

        # 计算平均改进
        improvements = [r.improvement for r in results[-50:]]
        avg_improvement = np.mean(improvements)
        avg_p_value = np.mean([r.p_value for r in results[-50:]])

        # 晋升准则
        if avg_p_value < self.significance_level and avg_improvement > 0.03:
            return True, f"statistically_significant_improvement ({avg_improvement:.2%})"
        elif avg_improvement > 0.05:
            return True, f"large_improvement ({avg_improvement:.2%})"
        else:
            return False, f"insufficient_improvement ({avg_improvement:.2%})"

    def get_variant_stats(self, variant_name: str) -> Dict[str, Any]:
        """获取变体的统计信息"""
        if variant_name not in self.variant_results:
            return {}

        results = self.variant_results[variant_name]

        return {
            "sample_count": len(results),
            "avg_improvement": np.mean([r.improvement for r in results[-50:]]),
            "avg_p_value": np.mean([r.p_value for r in results[-50:]]),
            "avg_loss": np.mean([r.loss for r in results[-50:]]),
            "avg_accuracy": np.mean([r.accuracy for r in results[-50:]]),
        }


# ============================================================================
# 4. QSA自适应控制器
# ============================================================================


class AdaptiveQSAController:
    """量子叠加注意力的自适应筛选比例控制器"""

    def __init__(self, base_ratio: float = 0.1, bounds: Tuple[float, float] = (0.05, 0.3)):
        self.ratio = base_ratio
        self.bounds = bounds
        self.performance_history: List[float] = []
        self.speedup_history: List[float] = []

    def update_ratio(self, accuracy_drop: float, speedup: float) -> float:
        """
        根据性能-成本权衡动态调整筛选比例

        Args:
            accuracy_drop: 准确率下降比例
            speedup: 加速比

        Returns:
            更新后的筛选比例
        """
        # 计算综合评分
        score = speedup * 0.7 - accuracy_drop * 0.3

        # 记录历史
        self.performance_history.append(accuracy_drop)
        self.speedup_history.append(speedup)

        # 自适应调整
        if score > 1.5:  # 表现好，增加比例
            self.ratio = min(self.ratio * 1.05, self.bounds[1])
            logger.info(f"QSA比例增加: {self.ratio:.3f} (评分: {score:.2f})")
        elif score < 0.5:  # 表现差，减少比例
            self.ratio = max(self.ratio * 0.95, self.bounds[0])
            logger.info(f"QSA比例减少: {self.ratio:.3f} (评分: {score:.2f})")

        # 确保比例在边界内
        self.ratio = np.clip(self.ratio, self.bounds[0], self.bounds[1])

        return self.ratio


# ============================================================================
# 5. QCI坍缩阈值学习器
# ============================================================================


class CollapseThresholdLearner:
    """基于强化学习的坍缩阈值控制器"""

    def __init__(self, state_dim: int = 3, learning_rate: float = 0.01):
        self.learning_rate = learning_rate

        # 策略网络：输入状态，输出阈值
        self.policy = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2),  # 输出 tau_low 和 tau_high
        )
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)

        # 经验回放缓冲
        self.replay_buffer: List[Tuple[torch.Tensor, float, float]] = []

    def forward(self, batch_stats: Dict[str, float]) -> Tuple[float, float]:
        """
        输入：批次统计信息
            - avg_entropy: 平均熵
            - entropy_std: 熵标准差
            - early_exit_rate: 早退率

        输出：
            - tau_low: 早退低阈值
            - tau_high: 早退高阈值
        """
        x = torch.tensor(
            [
                batch_stats["avg_entropy"],
                batch_stats["entropy_std"],
                batch_stats["early_exit_rate"],
            ],
            dtype=torch.float32,
        ).unsqueeze(0)

        thresholds = self.policy(x)
        tau_low = torch.sigmoid(thresholds[0, 0]) * 2.0  # 0-2范围
        tau_high = tau_low + torch.sigmoid(thresholds[0, 1]) * 3.0  # 确保tau_high > tau_low

        return tau_low.item(), tau_high.item()

    def update(self, batch_stats: Dict[str, float], speedup: float, accuracy_loss: float):
        """
        根据奖励更新策略

        Args:
            batch_stats: 批次统计
            speedup: 加速比
            accuracy_loss: 准确率损失
        """
        # 计算奖励
        reward = speedup * 0.8 - accuracy_loss * 0.2

        # 构造状态张量
        x = torch.tensor(
            [
                batch_stats["avg_entropy"],
                batch_stats["entropy_std"],
                batch_stats["early_exit_rate"],
            ],
            dtype=torch.float32,
        ).unsqueeze(0)

        # 前向传播
        thresholds = self.policy(x)
        tau_low = torch.sigmoid(thresholds[0, 0]) * 2.0
        tau_high = tau_low + torch.sigmoid(thresholds[0, 1]) * 3.0

        # 损失函数：最大化奖励（最小化负奖励）
        # 这里简化处理，实际应该用策略梯度（REINFORCE）
        loss = -reward

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


# ============================================================================
# 6. 自适应学习率控制器
# ============================================================================


class AdaptiveLearningRate:
    """分别为模长和相位维度设计自适应学习率"""

    def __init__(self, base_mod_lr: float = 1e-4, base_phase_lr: float = 1e-3):
        self.mod_lr = base_mod_lr
        self.phase_lr = base_phase_lr
        self.mod_history: List[float] = []
        self.phase_history: List[float] = []

    def update(self, mod_grad_norm: float, phase_grad_norm: float) -> Tuple[float, float]:
        """
        根据梯度范数动态调整学习率

        原理：
        - 相位空间是紧致的（模长固定），可以使用较大学习率
        - 模长空间是非紧致的，需要较小学习率保持稳定
        """
        # 平滑系数
        alpha = 0.05

        # 根据梯度范数动态调整
        new_mod_lr = 1e-4 * (1.0 / mod_grad_norm) ** 0.5
        new_phase_lr = 1e-3 * (1.0 / phase_grad_norm) ** 0.5

        # 指数移动平均平滑
        self.mod_lr = (1 - alpha) * self.mod_lr + alpha * new_mod_lr
        self.phase_lr = (1 - alpha) * self.phase_lr + alpha * new_phase_lr

        # 记录历史
        self.mod_history.append(self.mod_lr)
        self.phase_history.append(self.phase_lr)

        # 限制学习率范围
        self.mod_lr = np.clip(self.mod_lr, 1e-6, 1e-2)
        self.phase_lr = np.clip(self.phase_lr, 1e-5, 1e-1)

        return self.mod_lr, self.phase_lr


# ============================================================================
# 7. 异常检测器
# ============================================================================


class AnomalyDetector:
    """训练过程异常检测"""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics: Dict[str, List[float]] = defaultdict(list)

    def record(self, metric_name: str, value: float):
        """记录指标"""
        self.metrics[metric_name].append(value)
        if len(self.metrics[metric_name]) > self.window_size:
            self.metrics[metric_name].pop(0)

    def check_anomalies(self, current_metrics: Dict[str, float]) -> List[str]:
        """检测异常情况"""
        anomalies = []

        # 1. 损失爆炸检测
        if "loss" in current_metrics and len(self.metrics["loss"]) > 10:
            baseline = np.mean(self.metrics["loss"][-10:])
            if current_metrics["loss"] > baseline * 3:
                anomalies.append("loss_explosion")

        # 2. 梯度消失/爆炸检测
        if "grad_norm" in current_metrics:
            if current_metrics["grad_norm"] < 1e-7:
                anomalies.append("gradient_vanishing")
            elif current_metrics["grad_norm"] > 10:
                anomalies.append("gradient_explosion")

        # 3. 性能退化检测
        if "batch_time" in current_metrics and len(self.metrics["batch_time"]) > 100:
            recent_time = np.mean(self.metrics["batch_time"][-10:])
            baseline_time = np.mean(self.metrics["batch_time"][:10])
            if recent_time > baseline_time * 2:
                anomalies.append("performance_degradation")

        # 4. 酉约束违背检测
        if "unitarity_violation" in current_metrics:
            if current_metrics["unitarity_violation"] > 1e-4:
                anomalies.append("unitarity_violation")

        return anomalies


# ============================================================================
# 8. 自动恢复策略
# ============================================================================


class AutoRecovery:
    """异常自动恢复策略"""

    def __init__(self):
        self.recovery_strategies = {
            "loss_explosion": self._rollback_and_reduce_lr,
            "gradient_explosion": self._clip_grad_and_reduce_lr,
            "gradient_vanishing": self._reinitialize_output_layer,
            "performance_degradation": self._restart_from_checkpoint,
            "unitarity_violation": self._reproject_to_unitary,
        }

    def recover(self, anomaly_type: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        根据异常类型执行恢复策略

        返回：恢复动作和参数
        """
        if anomaly_type not in self.recovery_strategies:
            logger.warning(f"未知的异常类型: {anomaly_type}")
            return {"action": "ignore"}

        logger.warning(f"检测到异常: {anomaly_type}, 执行自动恢复")
        return self.recovery_strategies[anomaly_type](context or {})

    def _rollback_and_reduce_lr(self, context: Dict) -> Dict[str, Any]:
        """损失爆炸：回滚到最近checkpoint，降低学习率"""
        return {"action": "rollback", "reduce_lr_factor": 0.5, "reason": "loss_explosion_detected"}

    def _clip_grad_and_reduce_lr(self, context: Dict) -> Dict[str, Any]:
        """梯度爆炸：梯度裁剪 + 降低学习率"""
        return {
            "action": "clip_gradient",
            "clip_value": 1.0,
            "reduce_lr_factor": 0.5,
            "reason": "gradient_explosion_detected",
        }

    def _reinitialize_output_layer(self, context: Dict) -> Dict[str, Any]:
        """梯度消失：重新初始化输出层"""
        return {"action": "reinitialize_output", "reason": "gradient_vanishing_detected"}

    def _restart_from_checkpoint(self, context: Dict) -> Dict[str, Any]:
        """性能退化：从checkpoint重启"""
        return {"action": "restart", "reason": "performance_degradation_detected"}

    def _reproject_to_unitary(self, context: Dict) -> Dict[str, Any]:
        """酉约束违背：重新投影到酉矩阵空间"""
        return {"action": "reproject", "reason": "unitarity_violation_detected"}


# ============================================================================
# 9. 优化系统主控制器
# ============================================================================


class OptimizationSystem:
    """自动化优化系统主控制器"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

        # 初始化各个模块
        self.monitor = PerformanceMonitor()
        self.cost_monitor = GPUCostMonitor()
        self.api_guard = APICostGuard()
        self.shadow_engine = ShadowTestEngine()
        self.qsa_controller = AdaptiveQSAController()
        self.qci_learner = CollapseThresholdLearner()
        self.lr_controller = AdaptiveLearningRate()
        self.anomaly_detector = AnomalyDetector()
        self.auto_recovery = AutoRecovery()

        # 系统状态
        self.step_count = 0
        self.last_optimization_step = 0

        logger.info("自动化优化系统初始化完成")

    def on_training_step(self, metrics: PerformanceMetrics):
        """每个训练步调用"""
        self.step_count += 1

        # 1. 记录指标
        self.monitor.record(metrics)
        self.anomaly_detector.record("loss", metrics.loss)
        self.anomaly_detector.record("grad_norm", metrics.grad_norm)
        self.anomaly_detector.record("batch_time", metrics.batch_time)

        # 2. 异常检测
        current_metrics = {
            "loss": metrics.loss,
            "grad_norm": metrics.grad_norm,
            "batch_time": metrics.batch_time,
            "unitarity_violation": metrics.unitarity_violation,
        }
        anomalies = self.anomaly_detector.check_anomalies(current_metrics)

        if anomalies:
            for anomaly in anomalies:
                recovery_action = self.auto_recovery.recover(anomaly)
                logger.info(f"恢复策略: {recovery_action}")
                # 这里应该将recovery_action传递给训练器

        # 3. 每100步优化一次
        if self.step_count % 100 == 0:
            self.optimize_every_100_steps()

        # 4. 每1000步深度优化
        if self.step_count % 1000 == 0:
            self.optimize_every_1000_steps()

    def optimize_every_100_steps(self):
        """每100步执行的优化"""
        logger.info(f"执行第 {self.step_count} 步优化")

        # 1. 检查成本预算
        budget_status = self.cost_monitor.check_budget(estimated_remaining_steps=10000)
        logger.info(f"预算状态: {budget_status}")

        # 2. 更新QSA筛选比例
        loss_stats = self.monitor.get_statistics("loss", last_n=50)
        if loss_stats and "mean" in loss_stats:
            # 模拟：假设准确率下降0.1%，速度提升3倍
            accuracy_drop = 0.001
            speedup = 3.0
            new_ratio = self.qsa_controller.update_ratio(accuracy_drop, speedup)

    def optimize_every_1000_steps(self):
        """每1000步执行的深度优化"""
        logger.info(f"执行第 {self.step_count} 步深度优化")

        # 1. 评估影子测试变体
        for variant_name in ["variant_A", "variant_B"]:
            should_promote, reason = self.shadow_engine.should_promote(variant_name)
            if should_promote:
                logger.info(f"✓ {variant_name} 晋升到生产: {reason}")

        # 2. 调整QCI坍缩阈值
        batch_stats = {
            "avg_entropy": np.mean(
                [m.qci_early_exit_rate for m in self.monitor.metrics_history[-100:]]
            ),
            "entropy_std": np.std(
                [m.qci_early_exit_rate for m in self.monitor.metrics_history[-100:]]
            ),
            "early_exit_rate": np.mean(
                [m.qci_early_exit_rate for m in self.monitor.metrics_history[-100:]]
            ),
        }
        tau_low, tau_high = self.qci_learner.forward(batch_stats)
        logger.info(f"坍缩阈值更新: tau_low={tau_low:.3f}, tau_high={tau_high:.3f}")

    def on_api_call(self, api_name: str, success: bool = True):
        """API调用前后"""
        current_time = time.time()

        # 调用前检查
        should_call, reason = self.api_guard.should_call(api_name, current_time)
        if not should_call:
            logger.warning(f"API调用被阻止: {api_name}, 原因: {reason}")
            return False

        # 记录调用结果
        self.api_guard.record_call(api_name, success, current_time)

        return True

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统当前状态"""
        return {
            "step_count": self.step_count,
            "qsa_ratio": self.qsa_controller.ratio,
            "daily_cost": self.cost_monitor.daily_spent,
            "api_guard_state": self.api_guard.state.value,
            "metrics_count": len(self.monitor.metrics_history),
        }


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    # 创建优化系统
    optimizer = OptimizationSystem()

    # 模拟训练过程
    for step in range(10000):
        # 模拟性能指标
        metrics = PerformanceMetrics(
            loss=2.0 - step / 5000 + np.random.normal(0, 0.1),
            accuracy=step / 10000,
            grad_norm=np.random.uniform(0.5, 2.0),
            batch_time=np.random.uniform(0.1, 0.3),
            gpu_utilization=np.random.uniform(0.6, 0.9),
            gpu_memory_used=np.random.uniform(8, 12),
            qsa_topk_ratio=optimizer.qsa_controller.ratio,
            qci_early_exit_rate=0.3,
            qgd_mod_lr=optimizer.lr_controller.mod_lr,
            qgd_phase_lr=optimizer.lr_controller.phase_lr,
            unitarity_violation=1e-8,
        )

        # 调用优化系统
        optimizer.on_training_step(metrics)

        # 每1000步打印状态
        if step % 1000 == 0:
            status = optimizer.get_system_status()
            logger.info(f"系统状态: {status}")

    logger.info("训练完成")
