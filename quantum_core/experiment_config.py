"""
量子架构实验配置管理模块
QuantumArch Experiment Configuration Manager

统一管理训练实验的所有超参数，支持：
- YAML/JSON 配置文件读写
- 参数网格搜索
- 实验版本追踪
- 配置差异比对
"""

import json
import yaml
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Iterator
from dataclasses import dataclass, field, asdict
from copy import deepcopy
import itertools


@dataclass
class ModelConfig:
    """量子架构模型超参数"""
    # 基础维度
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    
    # QSA 参数
    qsa_topk: int = 64
    qsa_temperature: float = 1.0
    use_phase_encoding: bool = True
    
    # QEL 参数
    entangle_dim: int = 64
    entangle_global_ratio: float = 0.5
    
    # QCI 参数
    collapse_base_threshold: float = 0.7
    collapse_temperature_init: float = 2.0
    collapse_temperature_max: float = 10.0
    
    # QIR 参数
    router_n_experts: int = 4
    router_topk: int = 2
    
    # QGD 参数（Wirtinger 优化）
    use_wirtinger_grad: bool = True
    phase_lr_scale: float = 0.1
    
    # 复数表示
    use_cayley_init: bool = True
    unitary_loss_weight: float = 0.01
    
    # 量子纠错
    use_error_mitigation: bool = False
    error_mitigation_codes: int = 64
    
    # 位置编码
    pos_encoding_type: str = "bloch_sphere"  # "bloch_sphere" | "sinusoidal" | "learned"


@dataclass
class TrainingConfig:
    """训练超参数"""
    # 基础训练
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    batch_size: int = 32
    max_epochs: int = 100
    grad_clip: float = 1.0
    
    # 学习率调度
    scheduler_type: str = "warmup_cosine"  # "warmup_cosine" | "cosine" | "constant"
    warmup_steps: int = 1000
    min_lr: float = 1e-6
    
    # 混合精度
    use_amp: bool = True
    amp_dtype: str = "bfloat16"  # "float16" | "bfloat16"
    
    # 梯度检查点
    use_gradient_checkpointing: bool = False
    
    # 优化器
    optimizer: str = "adamw"  # "adamw" | "adam" | "sgd"
    beta1: float = 0.9
    beta2: float = 0.999
    
    # 量子专属
    separate_phase_lr: bool = True  # 相位参数使用独立学习率
    
    # 监控
    log_interval: int = 10
    eval_interval: int = 100
    save_interval: int = 1000
    
    # 随机种子
    seed: int = 42


@dataclass
class DataConfig:
    """数据集配置"""
    dataset_name: str = "synthetic"
    data_dir: str = "data/"
    seq_len: int = 512
    vocab_size: int = 50257
    
    # 数据增强
    use_noise_augmentation: bool = False
    noise_rate: float = 0.05
    
    # DataLoader
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class ExperimentConfig:
    """完整实验配置（聚合所有子配置）"""
    # 实验元信息
    experiment_name: str = "quantum_arch_exp"
    experiment_version: str = "v1.0"
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    # 子配置
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    # 输出路径
    output_dir: str = "experiments/outputs"
    checkpoint_dir: str = "experiments/checkpoints"
    log_dir: str = "logs"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    def config_hash(self) -> str:
        """计算配置的哈希值（用于实验去重）"""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def save(self, path: Optional[str] = None) -> Path:
        """保存配置到 YAML 文件"""
        if path is None:
            ts = time.strftime("%Y%m%d_%H%M%S")
            fname = f"{self.experiment_name}_{ts}_{self.config_hash()}.yaml"
            out_path = Path(self.output_dir) / fname
        else:
            out_path = Path(path)
        
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, allow_unicode=True, default_flow_style=False)
        
        return out_path
    
    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        """从 YAML/JSON 文件加载配置"""
        p = Path(path)
        with open(p, "r", encoding="utf-8") as f:
            if p.suffix in (".yaml", ".yml"):
                raw = yaml.safe_load(f)
            else:
                raw = json.load(f)
        
        config = cls(
            experiment_name=raw.get("experiment_name", "exp"),
            experiment_version=raw.get("experiment_version", "v1.0"),
            description=raw.get("description", ""),
            tags=raw.get("tags", []),
            model=ModelConfig(**raw.get("model", {})),
            training=TrainingConfig(**raw.get("training", {})),
            data=DataConfig(**raw.get("data", {})),
            output_dir=raw.get("output_dir", "experiments/outputs"),
            checkpoint_dir=raw.get("checkpoint_dir", "experiments/checkpoints"),
            log_dir=raw.get("log_dir", "logs"),
        )
        return config
    
    def diff(self, other: "ExperimentConfig") -> Dict[str, Any]:
        """与另一个配置的差异"""
        self_dict = self.to_dict()
        other_dict = other.to_dict()
        
        def _diff(d1, d2, path=""):
            diffs = {}
            for key in set(d1) | set(d2):
                full_key = f"{path}.{key}" if path else key
                v1, v2 = d1.get(key), d2.get(key)
                if isinstance(v1, dict) and isinstance(v2, dict):
                    diffs.update(_diff(v1, v2, full_key))
                elif v1 != v2:
                    diffs[full_key] = {"self": v1, "other": v2}
            return diffs
        
        return _diff(self_dict, other_dict)


class GridSearch:
    """
    超参数网格搜索
    
    自动生成所有参数组合，支持量子架构特定的搜索空间。
    """
    
    def __init__(self, base_config: ExperimentConfig):
        self.base_config = base_config
        self.search_space: Dict[str, List[Any]] = {}
    
    def add_param(self, param_path: str, values: List[Any]) -> "GridSearch":
        """
        添加搜索参数
        
        Args:
            param_path: 参数路径，如 "model.d_model" 或 "training.learning_rate"
            values: 候选值列表
        
        Returns:
            self（支持链式调用）
        """
        self.search_space[param_path] = values
        return self
    
    def _set_nested(self, config: ExperimentConfig, path: str, value: Any) -> ExperimentConfig:
        """设置嵌套配置参数"""
        parts = path.split(".")
        obj = config
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)
        return config
    
    def generate_configs(self) -> Iterator[ExperimentConfig]:
        """生成所有参数组合的配置"""
        if not self.search_space:
            yield deepcopy(self.base_config)
            return
        
        keys = list(self.search_space.keys())
        values = list(self.search_space.values())
        
        for combo in itertools.product(*values):
            config = deepcopy(self.base_config)
            name_parts = []
            for key, val in zip(keys, combo):
                self._set_nested(config, key, val)
                short_key = key.split(".")[-1]
                name_parts.append(f"{short_key}={val}")
            
            config.experiment_name = (
                f"{self.base_config.experiment_name}_"
                + "_".join(name_parts)
            )
            yield config
    
    def count(self) -> int:
        """返回总实验数量"""
        total = 1
        for values in self.search_space.values():
            total *= len(values)
        return total
