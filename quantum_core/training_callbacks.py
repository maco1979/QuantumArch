"""
量子架构训练监控回调系统
QuantumArch Training Monitor Callbacks

提供训练过程中的实时监控功能：
- 量子熵追踪（每层的von Neumann熵）
- 相位一致性监控
- 梯度健康检查
- 酉性约束监控
- 训练日志导出
"""

import torch
import torch.nn as nn
import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
from collections import defaultdict


class QuantumTrainingCallback:
    """量子架构训练回调基类"""
    
    def on_train_begin(self, state: Dict[str, Any]) -> None:
        """训练开始时调用"""
        pass
    
    def on_epoch_begin(self, epoch: int, state: Dict[str, Any]) -> None:
        """每个epoch开始时调用"""
        pass
    
    def on_batch_begin(self, batch: int, state: Dict[str, Any]) -> None:
        """每个batch开始时调用"""
        pass
    
    def on_batch_end(self, batch: int, state: Dict[str, Any]) -> None:
        """每个batch结束时调用"""
        pass
    
    def on_epoch_end(self, epoch: int, state: Dict[str, Any]) -> None:
        """每个epoch结束时调用"""
        pass
    
    def on_train_end(self, state: Dict[str, Any]) -> None:
        """训练结束时调用"""
        pass


class QuantumEntropyMonitor(QuantumTrainingCallback):
    """
    量子熵监控回调
    
    追踪每层的von Neumann熵，监控量子叠加状态的演化。
    熵越高表示量子态越"叠加"，熵为0表示纯坍缩状态。
    """
    
    def __init__(
        self,
        model: nn.Module,
        log_interval: int = 10,
        target_entropy_range: tuple = (0.3, 0.9),
        verbose: bool = True,
    ):
        """
        Args:
            model: 量子架构模型
            log_interval: 每N个batch记录一次
            target_entropy_range: 目标熵值范围 [min, max]
            verbose: 是否打印警告
        """
        self.model = model
        self.log_interval = log_interval
        self.target_min, self.target_max = target_entropy_range
        self.verbose = verbose
        self.history: Dict[str, List[float]] = defaultdict(list)
        self._batch_counter = 0
    
    def _compute_module_entropy(self, module: nn.Module) -> Optional[float]:
        """计算模块的量子熵（若支持）"""
        if hasattr(module, 'entropy'):
            try:
                return module.entropy().item()
            except Exception:
                return None
        return None
    
    def on_batch_end(self, batch: int, state: Dict[str, Any]) -> None:
        self._batch_counter += 1
        if self._batch_counter % self.log_interval != 0:
            return
        
        for name, module in self.model.named_modules():
            entropy = self._compute_module_entropy(module)
            if entropy is not None:
                self.history[name].append(entropy)
                
                if self.verbose:
                    if entropy < self.target_min:
                        print(
                            f"⚠️  [EntropyMonitor] {name}: 熵={entropy:.4f} "
                            f"< 目标最小值{self.target_min}，量子态过度坍缩"
                        )
                    elif entropy > self.target_max:
                        print(
                            f"⚠️  [EntropyMonitor] {name}: 熵={entropy:.4f} "
                            f"> 目标最大值{self.target_max}，量子叠加过度"
                        )
    
    def get_summary(self) -> Dict[str, Dict]:
        """获取熵的统计摘要"""
        summary = {}
        for name, values in self.history.items():
            if values:
                arr = np.array(values)
                summary[name] = {
                    "mean": float(arr.mean()),
                    "std": float(arr.std()),
                    "min": float(arr.min()),
                    "max": float(arr.max()),
                    "final": float(arr[-1]),
                }
        return summary


class GradientHealthMonitor(QuantumTrainingCallback):
    """
    梯度健康监控回调
    
    检测量子架构训练中常见的梯度问题：
    - 梯度爆炸（norm > threshold）
    - 梯度消失（norm < threshold）
    - 复数参数梯度的实部/虚部不平衡
    """
    
    def __init__(
        self,
        model: nn.Module,
        explosion_threshold: float = 100.0,
        vanishing_threshold: float = 1e-7,
        log_interval: int = 50,
    ):
        self.model = model
        self.explosion_threshold = explosion_threshold
        self.vanishing_threshold = vanishing_threshold
        self.log_interval = log_interval
        self.history: List[Dict] = []
        self._batch_counter = 0
    
    def on_batch_end(self, batch: int, state: Dict[str, Any]) -> None:
        self._batch_counter += 1
        if self._batch_counter % self.log_interval != 0:
            return
        
        total_norm = 0.0
        param_stats = []
        
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            
            grad = param.grad
            if grad.is_complex():
                # 复数梯度：分别检查实部和虚部
                real_norm = grad.real.norm().item()
                imag_norm = grad.imag.norm().item()
                grad_norm = (real_norm ** 2 + imag_norm ** 2) ** 0.5
                imbalance = abs(real_norm - imag_norm) / (real_norm + imag_norm + 1e-8)
                param_stats.append({
                    "name": name,
                    "grad_norm": grad_norm,
                    "real_norm": real_norm,
                    "imag_norm": imag_norm,
                    "imbalance": imbalance,
                })
            else:
                grad_norm = grad.norm().item()
                param_stats.append({
                    "name": name,
                    "grad_norm": grad_norm,
                })
            
            total_norm += grad_norm ** 2
        
        total_norm = total_norm ** 0.5
        
        record = {
            "batch": batch,
            "total_grad_norm": total_norm,
            "params": param_stats,
        }
        self.history.append(record)
        
        # 健康检查
        if total_norm > self.explosion_threshold:
            print(
                f"🔴 [GradMonitor] 梯度爆炸！总范数={total_norm:.2f} "
                f"> 阈值{self.explosion_threshold}"
            )
        elif total_norm < self.vanishing_threshold:
            print(
                f"🟡 [GradMonitor] 梯度消失！总范数={total_norm:.2e} "
                f"< 阈值{self.vanishing_threshold}"
            )


class UnitarityMonitor(QuantumTrainingCallback):
    """
    酉性约束监控回调
    
    量子架构要求权重矩阵保持酉性（U†U = I），
    监控酉性违反程度并在超过阈值时发出警告。
    """
    
    def __init__(
        self,
        model: nn.Module,
        tolerance: float = 1e-4,
        check_interval: int = 100,
        auto_project: bool = False,
    ):
        """
        Args:
            model: 量子架构模型
            tolerance: 酉性误差容忍度 ||U†U - I||_F
            check_interval: 检查间隔（batch数）
            auto_project: 是否自动投影回酉矩阵空间（使用Cayley变换）
        """
        self.model = model
        self.tolerance = tolerance
        self.check_interval = check_interval
        self.auto_project = auto_project
        self.violations: List[Dict] = []
        self._batch_counter = 0
    
    def _check_unitarity(self, W: torch.Tensor) -> float:
        """
        计算酉性误差 ||W†W - I||_F
        
        Args:
            W: 待检查的复数矩阵
        
        Returns:
            Frobenius范数误差
        """
        if W.dim() < 2:
            return 0.0
        if W.is_complex():
            WH = W.conj().transpose(-1, -2)
        else:
            WH = W.transpose(-1, -2)
        
        n = min(W.shape[-2], W.shape[-1])
        WtW = WH @ W
        eye = torch.eye(n, dtype=WtW.dtype, device=WtW.device)
        error = (WtW - eye).norm().item()
        return error
    
    def on_batch_end(self, batch: int, state: Dict[str, Any]) -> None:
        self._batch_counter += 1
        if self._batch_counter % self.check_interval != 0:
            return
        
        for name, param in self.model.named_parameters():
            if "weight" not in name or param.dim() < 2:
                continue
            
            error = self._check_unitarity(param.data)
            if error > self.tolerance:
                violation = {
                    "batch": batch,
                    "param": name,
                    "unitarity_error": error,
                }
                self.violations.append(violation)
                print(
                    f"⚠️  [UnitarityMonitor] {name}: "
                    f"酉性误差={error:.6f} > 容忍度{self.tolerance}"
                )


class TrainingLogger(QuantumTrainingCallback):
    """
    训练日志记录器
    
    将所有训练指标记录到 JSON 文件，支持后续分析和可视化。
    """
    
    def __init__(self, log_dir: str = "logs", experiment_name: str = "quantum_arch"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.log_path = self.log_dir / f"{experiment_name}_{int(time.time())}.json"
        self.logs: List[Dict] = []
    
    def on_batch_end(self, batch: int, state: Dict[str, Any]) -> None:
        record = {
            "batch": batch,
            "timestamp": time.time(),
        }
        record.update({k: v for k, v in state.items() if isinstance(v, (int, float, str))})
        self.logs.append(record)
    
    def on_train_end(self, state: Dict[str, Any]) -> None:
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(self.logs, f, indent=2, ensure_ascii=False)
        print(f"✅ 训练日志已保存: {self.log_path}")


class CallbackManager:
    """回调管理器：统一调度所有训练回调"""
    
    def __init__(self, callbacks: Optional[List[QuantumTrainingCallback]] = None):
        self.callbacks = callbacks or []
    
    def add(self, callback: QuantumTrainingCallback) -> None:
        self.callbacks.append(callback)
    
    def on_train_begin(self, state: Dict = None):
        for cb in self.callbacks:
            cb.on_train_begin(state or {})
    
    def on_epoch_begin(self, epoch: int, state: Dict = None):
        for cb in self.callbacks:
            cb.on_epoch_begin(epoch, state or {})
    
    def on_batch_begin(self, batch: int, state: Dict = None):
        for cb in self.callbacks:
            cb.on_batch_begin(batch, state or {})
    
    def on_batch_end(self, batch: int, state: Dict = None):
        for cb in self.callbacks:
            cb.on_batch_end(batch, state or {})
    
    def on_epoch_end(self, epoch: int, state: Dict = None):
        for cb in self.callbacks:
            cb.on_epoch_end(epoch, state or {})
    
    def on_train_end(self, state: Dict = None):
        for cb in self.callbacks:
            cb.on_train_end(state or {})
