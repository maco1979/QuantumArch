"""
集成优化系统的训练管线
使用真实 QuantumArch 核心模型进行训练
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# 将项目根目录加入路径，以导入 quantum_core
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from quantum_core.model import QuantumArch
from quantum_core.optimizer import QGD

from optimization_engine import OptimizationSystem, PerformanceMetrics
from metrics_exporter import QuantumArchMetrics

logger = logging.getLogger(__name__)


class MockQuantumArchLegacy(nn.Module):
    """旧版模拟模型（保留用于回退对比）"""

    def __init__(self, dim: int = 512, num_layers: int = 6):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.qsa = nn.MultiheadAttention(dim, num_heads=8)
        self.qel = nn.Linear(dim, dim)
        self.qci = nn.Sequential(nn.Linear(dim, dim // 2), nn.ReLU(), nn.Linear(dim // 2, dim))
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.ReLU(), nn.Linear(dim * 4, dim))
        self.qsa_topk_ratio = 0.1
        self.qci_tau_low = 0.5
        self.qci_tau_high = 1.5
        self.early_exit_enabled = True

    def forward(self, x):
        batch_size, seq_len, dim = x.shape
        qsa_start = time.time()
        attn_output, _ = self.qsa(x, x, x)
        qsa_time = time.time() - qsa_start
        x = x + self.qel(attn_output)
        qci_early_exit = False
        entropy = 1.0
        if self.early_exit_enabled:
            entropy = torch.randn(batch_size).abs().mean().item()
            if entropy < self.qci_tau_low:
                qci_early_exit = True
            elif entropy > self.qci_tau_high:
                x = x + self.qci(x)
        x = x + self.ffn(x)
        return {
            "output": x,
            "qsa_time": qsa_time,
            "qci_early_exit": qci_early_exit,
            "entropy": entropy,
        }

    def update_parameters(self, **kwargs):
        if "qsa_topk_ratio" in kwargs:
            self.qsa_topk_ratio = kwargs["qsa_topk_ratio"]
        if "qci_tau_low" in kwargs:
            self.qci_tau_low = kwargs["qci_tau_low"]
        if "qci_tau_high" in kwargs:
            self.qci_tau_high = kwargs["qci_tau_high"]


def create_model(config: dict, use_legacy: bool = False):
    """根据配置创建模型。

    Args:
        config: YAML 配置字典
        use_legacy: 是否使用旧版 MockQuantumArch
    Returns:
        模型实例
    """
    if use_legacy:
        return MockQuantumArchLegacy()

    # 从配置中读取超参数
    model_cfg = config.get("model", {})
    dim = model_cfg.get("dim", 256)
    num_layers = model_cfg.get("num_layers", 4)
    num_heads = model_cfg.get("num_heads", 8)
    ffn_dim = model_cfg.get("ffn_dim", None)
    topk_ratio = config.get("qsa", {}).get("adaptive", {}).get("topk_ratio", 0.1)
    collapse_cfg = config.get("qci", {}).get("collapse", {})
    tau_low = collapse_cfg.get("tau_low", 0.5)
    tau_high = collapse_cfg.get("tau_high", 1.5)
    dropout = config.get("training", {}).get("dropout", 0.0)

    return QuantumArch(
        dim=dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        topk_ratio=topk_ratio,
        collapse_enabled=True,
        tau_low=tau_low,
        tau_high=tau_high,
        dropout=dropout,
        qsa_mode="topk",
        direct_input=True,
    )


class OptimizedTrainingPipeline:
    """集成优化系统的训练管线"""

    def __init__(
        self, config_path: str = "./optimization_system/config.yaml", use_legacy: bool = False
    ):
        # 加载配置
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.use_legacy = use_legacy

        # 初始化模型（使用真实 QuantumArch 或旧版 Mock）
        self.model = create_model(self.config, use_legacy=use_legacy)

        # 选择损失函数
        self.criterion = nn.MSELoss()

        # 选择优化器
        if use_legacy:
            self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        else:
            # 使用 QGD 量子梯度下降优化器（复数参数用模长/相位分离更新）
            self.optimizer = QGD(
                self.model.parameters(),
                mod_lr=1e-4,
                phase_lr=1e-3,
            )

        # 初始化优化系统
        self.optimization_system = OptimizationSystem(self.config)

        # 初始化Prometheus指标
        self.metrics_exporter = QuantumArchMetrics(
            port=self.config["monitoring"]["prometheus"]["port"]
        )

        # 训练状态
        self.step_count = 0
        self.best_validation_loss = float("inf")

        logger.info("优化训练管线初始化完成")

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """单步训练"""
        batch_start_time = time.time()

        # 前向传播
        self.optimizer.zero_grad()
        inputs = batch["inputs"]
        targets = batch["targets"]

        # QuantumArch 接受 dict 输入以兼容接口
        model_input = {"inputs": inputs}
        outputs_dict = self.model(model_input, training=self.model.training)
        outputs = outputs_dict["output"]

        # 确保输出与目标形状一致
        if outputs.shape != targets.shape:
            targets = targets[..., : outputs.shape[-1]]

        # 计算损失
        loss = self.criterion(outputs, targets)
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # 参数更新
        self.optimizer.step()

        batch_time = time.time() - batch_start_time

        # 计算性能指标
        with torch.no_grad():
            accuracy = 1.0 / (1.0 + loss.item())

            # 梯度范数
            grad_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.norm().item() ** 2
            grad_norm = grad_norm**0.5

            # GPU信息
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / (1024**3)
                gpu_util = 0.8
            else:
                gpu_memory = 0.0
                gpu_util = 0.0

            # QCI早退率
            qci_early_exit = 1.0 if outputs_dict["qci_early_exit"] else 0.0

            # 酉性违背度量（真实模型）
            unitarity_violation = 0.0
            if not self.use_legacy and hasattr(self.model, "get_unitarity_report"):
                report = self.model.get_unitarity_report()
                if report:
                    unitarity_violation = max(report.values())

        return {
            "loss": loss.item(),
            "accuracy": accuracy,
            "grad_norm": grad_norm,
            "batch_time": batch_time,
            "qsa_time": outputs_dict["qsa_time"],
            "qci_early_exit": qci_early_exit,
            "qci_entropy": outputs_dict["entropy"],
            "gpu_memory": gpu_memory,
            "gpu_util": gpu_util,
            "unitarity_violation": unitarity_violation,
        }

    def train_epoch(self, train_loader: DataLoader, epoch: int):
        """训练一个epoch"""
        self.model.train()
        epoch_start_time = time.time()
        total_loss = 0.0

        logger.info(f"开始Epoch {epoch}")

        for batch_idx, batch in enumerate(train_loader):
            self.step_count += 1

            # 执行训练步
            step_metrics = self.train_step(batch)
            total_loss += step_metrics["loss"]

            # 构造性能指标对象
            performance_metrics = PerformanceMetrics(
                loss=step_metrics["loss"],
                accuracy=step_metrics["accuracy"],
                grad_norm=step_metrics["grad_norm"],
                batch_time=step_metrics["batch_time"],
                gpu_utilization=step_metrics["gpu_util"],
                gpu_memory_used=step_metrics["gpu_memory"],
                qsa_topk_ratio=self.model.qsa_topk_ratio,
                qci_early_exit_rate=step_metrics["qci_early_exit"],
                qgd_mod_lr=1e-4,
                qgd_phase_lr=1e-3,
                unitarity_violation=step_metrics.get("unitarity_violation", 1e-8),
            )

            # 调用优化系统
            self.optimization_system.on_training_step(performance_metrics)

            # 更新Prometheus指标
            self.metrics_exporter.update_metrics(
                {
                    "training_loss": step_metrics["loss"],
                    "accuracy": step_metrics["accuracy"],
                    "qsa_topk_ratio": self.model.qsa_topk_ratio,
                    "qci_early_exit_rate": step_metrics["qci_early_exit"],
                    "qci_avg_entropy": step_metrics["qci_entropy"],
                    "batch_processing_time": step_metrics["batch_time"],
                    "step_time": step_metrics["batch_time"],
                    "gpu_memory_used": step_metrics["gpu_memory"],
                    "gpu_utilization": step_metrics["gpu_util"],
                    "step_increment": 1,
                }
            )

            # 日志输出
            if batch_idx % 10 == 0:
                logger.info(
                    f"Epoch {epoch}, Batch {batch_idx}, "
                    f"Loss: {step_metrics['loss']:.4f}, "
                    f"Acc: {step_metrics['accuracy']:.4f}, "
                    f"QSA Ratio: {self.model.qsa_topk_ratio:.3f}, "
                    f"Time: {step_metrics['batch_time']:.3f}s"
                )

        # Epoch统计
        avg_loss = total_loss / len(train_loader)
        epoch_time = time.time() - epoch_start_time

        logger.info(f"Epoch {epoch} 完成, 平均损失: {avg_loss:.4f}, 耗时: {epoch_time:.2f}s")

        # 更新QCI阈值（来自优化系统）
        batch_stats = {
            "avg_entropy": avg_loss,
            "entropy_std": 0.1,  # 简化
            "early_exit_rate": 0.3,  # 简化
        }
        tau_low, tau_high = self.optimization_system.qci_learner.forward(batch_stats)
        self.model.update_parameters(qci_tau_low=tau_low, qci_tau_high=tau_high)

        logger.info(f"QCI阈值更新: tau_low={tau_low:.3f}, tau_high={tau_high:.3f}")

        return avg_loss

    def validate(self, val_loader: DataLoader) -> float:
        """验证"""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["inputs"]
                targets = batch["targets"]

                model_input = {"inputs": inputs}
                outputs_dict = self.model(model_input, training=False)
                outputs = outputs_dict["output"]

                if outputs.shape != targets.shape:
                    targets = targets[..., : outputs.shape[-1]]

                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

        avg_loss = total_loss / len(val_loader)
        logger.info(f"验证损失: {avg_loss:.4f}")

        # 酉性报告（真实模型）
        if not self.use_legacy and hasattr(self.model, "get_unitarity_report"):
            report = self.model.get_unitarity_report()
            if report:
                max_violation = max(report.values())
                avg_violation = sum(report.values()) / len(report)
                logger.info(
                    f"酉性报告: max_violation={max_violation:.2e}, avg_violation={avg_violation:.2e}"
                )

        # 更新Prometheus
        self.metrics_exporter.update_metrics(
            {
                "validation_loss": avg_loss,
            }
        )

        return avg_loss

    def train(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int):
        """完整训练流程"""
        # 启动Prometheus服务器
        self.metrics_exporter.start()

        logger.info("开始训练流程")

        for epoch in range(1, num_epochs + 1):
            # 训练
            train_loss = self.train_epoch(train_loader, epoch)

            # 验证
            val_loss = self.validate(val_loader)

            # 保存最佳模型
            if val_loss < self.best_validation_loss:
                self.best_validation_loss = val_loss
                self.save_checkpoint(f"best_epoch_{epoch}")
                self.metrics_exporter.record_checkpoint()
                logger.info(f"✓ 新的最佳模型，验证损失: {val_loss:.4f}")

            # 定期保存
            if epoch % 5 == 0:
                self.save_checkpoint(f"epoch_{epoch}")

            # 每个epoch后更新成本监控
            estimated_remaining = (num_epochs - epoch) * len(train_loader)
            budget_status = self.optimization_system.cost_monitor.check_budget(
                estimated_remaining_steps=estimated_remaining
            )

            if budget_status["action"] != "continue":
                logger.warning(f"预算状态: {budget_status}")
                if budget_status["action"] == "stop":
                    logger.error("预算超支，停止训练")
                    break

        logger.info("训练完成")
        self.optimization_system.get_system_status()

    def save_checkpoint(self, name: str):
        """保存checkpoint"""
        checkpoint_path = Path(self.config["checkpointing"]["directory"]) / f"{name}.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "step_count": self.step_count,
                "best_validation_loss": self.best_validation_loss,
                "config": self.config,
            },
            checkpoint_path,
        )

        logger.info(f"Checkpoint保存到: {checkpoint_path}")


def create_mock_dataloader(batch_size: int = 32, num_samples: int = 1000):
    """创建模拟数据加载器"""

    class MockDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples: int, dim: int = 512, seq_len: int = 128):
            self.num_samples = num_samples
            self.dim = dim
            self.seq_len = seq_len

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            inputs = torch.randn(self.seq_len, self.dim)
            targets = torch.randn(self.seq_len, self.dim)
            return {"inputs": inputs, "targets": targets}

    dataset = MockDataset(num_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def main():
    """主函数 - 使用真实 QuantumArch 进行训练"""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    logger.info("=== 量子架构训练管线 ===")

    # 创建训练管线（use_legacy=False 使用真实 QuantumArch）
    pipeline = OptimizedTrainingPipeline(use_legacy=False)

    # 创建模拟数据
    train_loader = create_mock_dataloader(batch_size=8, num_samples=100)
    val_loader = create_mock_dataloader(batch_size=8, num_samples=20)

    # 开始训练
    pipeline.train(train_loader=train_loader, val_loader=val_loader, num_epochs=3)

    logger.info("训练完成")


if __name__ == "__main__":
    main()
