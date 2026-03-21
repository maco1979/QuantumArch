"""
Prometheus指标导出器
用于Grafana监控仪表板
"""

from prometheus_client import start_http_server, Counter, Gauge, Histogram, Summary
from prometheus_client.core import CollectorRegistry
import time
import torch
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class QuantumArchMetrics:
    """量子架构Prometheus指标收集器"""

    def __init__(self, port: int = 8000):
        self.port = port

        # 创建自定义registry
        self.registry = CollectorRegistry()

        # 1. 训练指标
        self.training_loss = Gauge(
            'quantum_arch_training_loss',
            '训练损失',
            registry=self.registry
        )
        self.validation_loss = Gauge(
            'quantum_arch_validation_loss',
            '验证损失',
            registry=self.registry
        )
        self.accuracy = Gauge(
            'quantum_arch_accuracy',
            '准确率',
            registry=self.registry
        )

        # 2. QSA指标
        self.qsa_topk_ratio = Gauge(
            'quantum_arch_qsa_topk_ratio',
            'QSA筛选比例',
            registry=self.registry
        )
        self.qsa_computation_time = Histogram(
            'quantum_arch_qsa_computation_time_seconds',
            'QSA计算时间',
            registry=self.registry
        )
        self.qsa_interference_score = Gauge(
            'quantum_arch_qsa_interference_score',
            'QSA干涉评分',
            registry=self.registry
        )

        # 3. QEL指标
        self.qel_entanglement_strength = Gauge(
            'quantum_arch_qel_entanglement_strength',
            'QEL纠缠强度',
            registry=self.registry
        )
        self.qel_qft_time = Histogram(
            'quantum_arch_qel_qft_time_seconds',
            'QEL QFT计算时间',
            registry=self.registry
        )

        # 4. QCI指标
        self.qci_early_exit_rate = Gauge(
            'quantum_arch_qci_early_exit_rate',
            'QCI早退率',
            registry=self.registry
        )
        self.qci_avg_entropy = Gauge(
            'quantum_arch_qci_avg_entropy',
            'QCI平均熵',
            registry=self.registry
        )
        self.qci_tau_low = Gauge(
            'quantum_arch_qci_tau_low',
            'QCI早退低阈值',
            registry=self.registry
        )
        self.qci_tau_high = Gauge(
            'quantum_arch_qci_tau_high',
            'QCI早退高阈值',
            registry=self.registry
        )

        # 5. QGD指标
        self.qgd_mod_lr = Gauge(
            'quantum_arch_qgd_mod_lr',
            'QGD模长学习率',
            registry=self.registry
        )
        self.qgd_phase_lr = Gauge(
            'quantum_arch_qgd_phase_lr',
            'QGD相位学习率',
            registry=self.registry
        )
        self.qgd_mod_grad_norm = Gauge(
            'quantum_arch_qgd_mod_grad_norm',
            'QGD模长梯度范数',
            registry=self.registry
        )
        self.qgd_phase_grad_norm = Gauge(
            'quantum_arch_qgd_phase_grad_norm',
            'QGD相位梯度范数',
            registry=self.registry
        )

        # 6. 酉约束指标
        self.unitarity_violation = Gauge(
            'quantum_arch_unitarity_violation',
            '酉约束违背度',
            registry=self.registry
        )
        self.matrix_condition_number = Gauge(
            'quantum_arch_matrix_condition_number',
            '矩阵条件数',
            registry=self.registry
        )

        # 7. 性能指标
        self.batch_processing_time = Histogram(
            'quantum_arch_batch_processing_time_seconds',
            '批处理时间',
            registry=self.registry
        )
        self.step_time = Gauge(
            'quantum_arch_step_time_seconds',
            '每步时间',
            registry=self.registry
        )
        self.throughput = Gauge(
            'quantum_arch_throughput_samples_per_second',
            '吞吐量（样本/秒）',
            registry=self.registry
        )

        # 8. GPU指标
        self.gpu_utilization = Gauge(
            'quantum_arch_gpu_utilization',
            'GPU利用率',
            registry=self.registry
        )
        self.gpu_memory_used = Gauge(
            'quantum_arch_gpu_memory_used_gb',
            'GPU显存使用（GB）',
            registry=self.registry
        )
        self.gpu_memory_allocated = Gauge(
            'quantum_arch_gpu_memory_allocated_gb',
            'GPU显存分配（GB）',
            registry=self.registry
        )

        # 9. 成本指标
        self.daily_cost = Gauge(
            'quantum_arch_daily_cost_usd',
            '每日成本（美元）',
            registry=self.registry
        )
        self.hourly_cost = Gauge(
            'quantum_arch_hourly_cost_usd',
            '每小时成本（美元）',
            registry=self.registry
        )

        # 10. 系统指标
        self.active_steps = Counter(
            'quantum_arch_active_steps_total',
            '总训练步数',
            registry=self.registry
        )
        self.checkpoints_saved = Counter(
            'quantum_arch_checkpoints_saved_total',
            '保存的checkpoint数量',
            registry=self.registry
        )
        self.anomaly_detected = Counter(
            'quantum_arch_anomaly_detected_total',
            '检测到的异常次数',
            ['anomaly_type'],
            registry=self.registry
        )

        # 11. 影子测试指标
        self.shadow_test_samples = Counter(
            'quantum_arch_shadow_test_samples_total',
            '影子测试样本数',
            ['variant_name'],
            registry=self.registry
        )
        self.shadow_test_improvement = Gauge(
            'quantum_arch_shadow_test_improvement',
            '影子测试改进率',
            ['variant_name'],
            registry=self.registry
        )

        # 12. 自动恢复指标
        self.auto_recovery_triggered = Counter(
            'quantum_arch_auto_recovery_triggered_total',
            '自动恢复触发次数',
            ['recovery_type'],
            registry=self.registry
        )

        logger.info(f"Prometheus指标导出器初始化完成，端口: {self.port}")

    def start(self):
        """启动HTTP服务器"""
        start_http_server(self.port, registry=self.registry)
        logger.info(f"Prometheus指标服务器启动: http://localhost:{self.port}/metrics")

    def update_metrics(self, metrics: Dict[str, Any]):
        """更新所有指标"""
        # 训练指标
        if 'training_loss' in metrics:
            self.training_loss.set(metrics['training_loss'])
        if 'validation_loss' in metrics:
            self.validation_loss.set(metrics['validation_loss'])
        if 'accuracy' in metrics:
            self.accuracy.set(metrics['accuracy'])

        # QSA指标
        if 'qsa_topk_ratio' in metrics:
            self.qsa_topk_ratio.set(metrics['qsa_topk_ratio'])
        if 'qsa_computation_time' in metrics:
            self.qsa_computation_time.observe(metrics['qsa_computation_time'])

        # QEL指标
        if 'qel_entanglement_strength' in metrics:
            self.qel_entanglement_strength.set(metrics['qel_entanglement_strength'])

        # QCI指标
        if 'qci_early_exit_rate' in metrics:
            self.qci_early_exit_rate.set(metrics['qci_early_exit_rate'])
        if 'qci_avg_entropy' in metrics:
            self.qci_avg_entropy.set(metrics['qci_avg_entropy'])
        if 'qci_tau_low' in metrics:
            self.qci_tau_low.set(metrics['qci_tau_low'])
        if 'qci_tau_high' in metrics:
            self.qci_tau_high.set(metrics['qci_tau_high'])

        # QGD指标
        if 'qgd_mod_lr' in metrics:
            self.qgd_mod_lr.set(metrics['qgd_mod_lr'])
        if 'qgd_phase_lr' in metrics:
            self.qgd_phase_lr.set(metrics['qgd_phase_lr'])
        if 'qgd_mod_grad_norm' in metrics:
            self.qgd_mod_grad_norm.set(metrics['qgd_mod_grad_norm'])
        if 'qgd_phase_grad_norm' in metrics:
            self.qgd_phase_grad_norm.set(metrics['qgd_phase_grad_norm'])

        # 酉约束指标
        if 'unitarity_violation' in metrics:
            self.unitarity_violation.set(metrics['unitarity_violation'])

        # 性能指标
        if 'batch_processing_time' in metrics:
            self.batch_processing_time.observe(metrics['batch_processing_time'])
        if 'step_time' in metrics:
            self.step_time.set(metrics['step_time'])
        if 'throughput' in metrics:
            self.throughput.set(metrics['throughput'])

        # GPU指标
        if 'gpu_utilization' in metrics:
            self.gpu_utilization.set(metrics['gpu_utilization'])
        if 'gpu_memory_used' in metrics:
            self.gpu_memory_used.set(metrics['gpu_memory_used'])
        if 'gpu_memory_allocated' in metrics:
            self.gpu_memory_allocated.set(metrics['gpu_memory_allocated'])

        # 成本指标
        if 'daily_cost' in metrics:
            self.daily_cost.set(metrics['daily_cost'])
        if 'hourly_cost' in metrics:
            self.hourly_cost.set(metrics['hourly_cost'])

        # 系统指标
        if 'step_increment' in metrics:
            self.active_steps.inc(metrics['step_increment'])

    def record_anomaly(self, anomaly_type: str):
        """记录异常"""
        self.anomaly_detected.labels(anomaly_type=anomaly_type).inc()
        logger.warning(f"异常记录: {anomaly_type}")

    def record_checkpoint(self):
        """记录checkpoint保存"""
        self.checkpoints_saved.inc()
        logger.info("Checkpoint已保存")

    def record_shadow_test(self, variant_name: str, improvement: float):
        """记录影子测试"""
        self.shadow_test_samples.labels(variant_name=variant_name).inc()
        self.shadow_test_improvement.labels(variant_name=variant_name).set(improvement)

    def record_auto_recovery(self, recovery_type: str):
        """记录自动恢复"""
        self.auto_recovery_triggered.labels(recovery_type=recovery_type).inc()
        logger.warning(f"自动恢复触发: {recovery_type}")

    def get_gpu_info(self) -> Dict[str, float]:
        """获取GPU信息"""
        if not torch.cuda.is_available():
            return {
                'gpu_utilization': 0.0,
                'gpu_memory_used': 0.0,
                'gpu_memory_allocated': 0.0,
            }

        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

            gpu_util = info.gpu / 100.0
            memory_used = mem_info.used / (1024**3)
            memory_allocated = torch.cuda.memory_allocated() / (1024**3)

            return {
                'gpu_utilization': gpu_util,
                'gpu_memory_used': memory_used,
                'gpu_memory_allocated': memory_allocated,
            }
        except ImportError:
            # 如果没有pynvml，使用torch的基本信息
            memory_allocated = torch.cuda.memory_allocated() / (1024**3)
            return {
                'gpu_utilization': 0.0,  # 无法获取
                'gpu_memory_used': memory_allocated,
                'gpu_memory_allocated': memory_allocated,
            }
