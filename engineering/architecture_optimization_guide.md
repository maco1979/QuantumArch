# QuantumArch 架构优化指南

**针对 QuantumArch 复杂模块的架构优化建议**

---

## 目录

1. [模块解耦与抽象](#模块解耦与抽象)
2. [配置管理系统](#配置管理系统)
3. [内存优化策略](#内存优化策略)
4. [并行计算优化](#并行计算优化)
5. [可扩展性设计](#可扩展性设计)
6. [架构决策记录](#架构决策记录)

---

## 模块解耦与抽象

### 问题分析

当前 `quantum_core` 模块存在以下耦合问题:

1. **紧密耦合**: `QuantumBlock` 直接依赖所有子模块实现
2. **接口不统一**: 各核心组件缺乏统一抽象基类
3. **配置分散**: 超参数散落在各个模块中
4. **测试困难**: 高耦合导致单元测试复杂

### 优化方案

#### 1. 核心抽象基类

```python
# quantum_core/base.py
"""量子架构核心抽象基类。"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn


class QuantumOperator(ABC):
    """量子算子抽象基类。

    所有核心量子操作(QSA, QEL, QCI 等)都应继承此基类。

    设计原则:
    1. 统一接口: 所有算子使用相同的前向传播接口
    2. 度量标准: 所有算子返回标准化的度量字典
    3. 酉性检查: 酉矩阵算子提供统一的酉性验证接口
    """

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        training: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """前向传播。

        Args:
            x: 输入张量 (batch, seq_len, dim) 或相关形状
            training: 是否训练模式

        Returns:
            (output, metrics):
                - output: 输出张量
                - metrics: 度量字典,应包含算子特定的性能/质量指标
        """
        pass

    @abstractmethod
    def get_unitarity_report(self) -> Dict[str, float]:
        """返回酉性约束报告。

        Returns:
            酉性违背度字典 {参数名: violation}
            非酉矩阵算子应返回空字典
        """
        pass


class UnitaryOperator(QuantumOperator):
    """酉矩阵算子基类。

    特点:
    1. 保证严格的酉性约束
    2. 提供酉性验证方法
    3. 支持酉性监控和警告
    """

    @abstractmethod
    def get_unitarity_violation(self) -> torch.Tensor:
        """返回当前酉性违背度。

        Returns:
            违背度标量张量,值越小越接近完美酉性
        """
        pass

    def check_unitarity(
        self,
        tolerance: float = 1e-3,
        raise_on_violation: bool = False
    ) -> bool:
        """检查酉性约束。

        Args:
            tolerance: 可容忍的违背度阈值
            raise_on_violation: 违背时是否抛出异常

        Returns:
            是否满足酉性约束

        Raises:
            RuntimeError: 当违背度超过阈值且 raise_on_violation=True
        """
        violation = self.get_unitarity_violation().item()

        if violation > tolerance:
            msg = f"酉性违背度过大: {violation:.2e} > {tolerance:.2e}"
            if raise_on_violation:
                raise RuntimeError(msg)
            return False
        return True
```

#### 2. 重构示例

```python
# 重构前: 紧密耦合
class QuantumBlock(nn.Module):
    def __init__(self, dim, num_heads, ...):
        super().__init__()
        self.qsa = QuantumSuperpositionAttention(dim, num_heads, ...)
        self.qel = QuantumEntanglementLayer(dim, ...)
        self.qci = QuantumCollapseInference(dim, ...)
        self.ffn = QuantumFFN(dim, ...)

    def forward(self, x, training=True):
        # 直接调用各模块
        q_out, qsa_metrics = self.qsa(x, training=training)
        e_out, qel_metrics = self.qel(q_out, training=training)
        c_out, qci_metrics = self.qci(e_out, training=training)
        f_out, ffn_metrics = self.ffn(c_out, training=training)

        return f_out, {**qsa_metrics, **qel_metrics, **qci_metrics, **ffn_metrics}

# 重构后: 基于抽象接口
class QuantumBlock(nn.Module):
    def __init__(
        self,
        *args,
        qsa_module: QuantumOperator,
        qel_module: QuantumOperator,
        qci_module: QuantumOperator,
        ffn_module: QuantumOperator,
        **kwargs
    ):
        super().__init__()
        self.qsa = qsa_module
        self.qel = qel_module
        self.qci = qci_module
        self.ffn = ffn_module

        # 自动酉性监控
        self.unitarity_monitor = UnitarityMonitor(
            modules=[self.qsa, self.qel],
            check_interval=100
        )

    def forward(self, x, training=True):
        # 统一接口调用
        out, metrics = self._apply_operator(self.qsa, x, training, metrics={})

        out, metrics = self._apply_operator(self.qel, out, training, metrics)
        out, metrics = self._apply_operator(self.qci, out, training, metrics)
        out, metrics = self._apply_operator(self.ffn, out, training, metrics)

        # 酉性监控
        if training:
            self.unitarity_monitor.check()

        return out, metrics

    def _apply_operator(
        self,
        operator: QuantumOperator,
        x: torch.Tensor,
        training: bool,
        metrics: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """应用量子算子并合并度量。"""
        out, op_metrics = operator(x, training=training)
        # 添加算子名称前缀
        prefixed = {f"{operator.__class__.__name__}_{k}": v for k, v in op_metrics.items()}
        metrics.update(prefixed)
        return out, metrics
```

---

## 配置管理系统

### 问题分析

当前超参数配置问题:
1. 硬编码在各个模块中
2. 难以进行超参数搜索
3. 缺乏配置验证
4. 无法保存/加载实验配置

### 优化方案

#### 1. 分层配置系统

```python
# quantum_core/config.py
"""QuantumArch 配置管理系统。"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
import json
import yaml


@dataclass
class BaseConfig:
    """配置基类。

    提供:
    1. 配置验证
    2. 序列化/反序列化
    3. 配置合并
    """

    def validate(self) -> None:
        """验证配置合法性。"""
        pass

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseConfig':
        """从字典创建配置。"""
        return cls(**data)

    def save(self, path: str) -> None:
        """保存配置到文件。"""
        ext = path.split('.')[-1]
        data = self.to_dict()

        if ext == 'json':
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        elif ext in ['yaml', 'yml']:
            with open(path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        else:
            raise ValueError(f"不支持的配置格式: {ext}")

    @classmethod
    def load(cls, path: str) -> 'BaseConfig':
        """从文件加载配置。"""
        ext = path.split('.')[-1]

        if ext == 'json':
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif ext in ['yaml', 'yml']:
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        else:
            raise ValueError(f"不支持的配置格式: {ext}")

        return cls.from_dict(data)


@dataclass
class QSAConfig(BaseConfig):
    """QSA 量子叠加注意力配置。"""
    mode: str = 'topk'  # 'topk' 或 'full'
    topk_ratio: float = 0.1

    def validate(self) -> None:
        if self.mode not in ['topk', 'full']:
            raise ValueError(f"QSA mode 必须是 'topk' 或 'full', 实际: {self.mode}")

        if not 0 < self.topk_ratio <= 1:
            raise ValueError(f"topk_ratio 必须在 (0, 1] 范围内, 实际: {self.topk_ratio}")


@dataclass
class QELConfig(BaseConfig):
    """QEL 量子纠缠层配置。"""
    use_adaptive: bool = True
    use_global_qft: bool = True
    qft_steps: int = 1
    coupling_type: str = 'full'  # 'full' 或 'diagonal'

    def validate(self) -> None:
        if self.coupling_type not in ['full', 'diagonal']:
            raise ValueError(f"coupling_type 必须是 'full' 或 'diagonal', 实际: {self.coupling_type}")

        if self.qft_steps < 1:
            raise ValueError(f"qft_steps 必须 >= 1, 实际: {self.qft_steps}")


@dataclass
class QCIConfig(BaseConfig):
    """QCI 量子坍缩推理配置。"""
    enabled: bool = True
    tau_low: float = 0.5
    tau_high: float = 1.5
    adaptive_tau: bool = False

    def validate(self) -> None:
        if self.tau_low >= self.tau_high:
            raise ValueError(f"tau_low ({self.tau_low}) 必须小于 tau_high ({self.tau_high})")

        if self.tau_low <= 0:
            raise ValueError(f"tau_low 必须 > 0, 实际: {self.tau_low}")


@dataclass
class QuantumArchConfig(BaseConfig):
    """QuantumArch 完整配置。"""
    # 模型结构
    dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    ffn_dim: Optional[int] = None

    # 核心组件配置
    qsa: QSAConfig = field(default_factory=QSAConfig)
    qel: QELConfig = field(default_factory=QELConfig)
    qci: QCIConfig = field(default_factory=QCIConfig)

    # 训练配置
    dropout: float = 0.0
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2

    # 系统配置
    use_checkpoint: bool = False
    use_mixed_precision: bool = True

    def validate(self) -> None:
        """验证完整配置。"""
        # 验证基础约束
        if self.dim % self.num_heads != 0:
            raise ValueError(f"dim ({self.dim}) 必须能被 num_heads ({self.num_heads}) 整除")

        if self.ffn_dim is not None and self.ffn_dim <= self.dim:
            raise ValueError(f"ffn_dim ({self.ffn_dim}) 必须大于 dim ({self.dim})")

        if self.num_layers < 1:
            raise ValueError(f"num_layers 必须 >= 1, 实际: {self.num_layers}")

        # 验证子配置
        self.qsa.validate()
        self.qel.validate()
        self.qci.validate()

    @property
    def head_dim(self) -> int:
        """每个注意力头的维度。"""
        return self.dim // self.num_heads

    @property
    def effective_ffn_dim(self) -> int:
        """有效的 FFN 中间维度。"""
        return self.ffn_dim or 4 * self.dim
```

#### 2. 配置使用示例

```python
# 示例 1: 默认配置
config = QuantumArchConfig()
config.validate()
model = QuantumArch.from_config(config)

# 示例 2: 自定义配置
config = QuantumArchConfig(
    dim=768,
    num_heads=12,
    qsa=QSAConfig(topk_ratio=0.2),
    qel=QELConfig(use_global_qft=False),
    qci=QCIConfig(tau_low=0.3, tau_high=2.0)
)
config.validate()

# 示例 3: 从文件加载
config = QuantumArchConfig.load('configs/experiment_001.yaml')
model = QuantumArch.from_config(config)

# 示例 4: 保存配置
config.save('configs/experiment_001.yaml')

# 示例 5: 超参数搜索
def hyperparameter_search():
    configs = []

    for dim in [256, 512, 768]:
        for num_heads in [4, 8, 12]:
            if dim % num_heads == 0:
                for topk_ratio in [0.05, 0.1, 0.2]:
                    config = QuantumArchConfig(
                        dim=dim,
                        num_heads=num_heads,
                        qsa=QSAConfig(topk_ratio=topk_ratio)
                    )
                    try:
                        config.validate()
                        configs.append(config)
                    except ValueError as e:
                        print(f"无效配置: {e}")

    print(f"生成 {len(configs)} 个有效配置")
    return configs
```

---

## 内存优化策略

### 问题分析

当前内存使用问题:
1. 复数张量占用 2x 内存
2. QSA O(N²) 注意力矩阵内存
3. 缺乏梯度检查点
4. 没有内存监控

### 优化方案

#### 1. 内存高效的 QSA 实现

```python
# quantum_core/optimized_attention.py
"""内存高效的量子叠加注意力。"""

import torch
import torch.nn as nn
import math
from typing import Tuple, Dict, Optional
from .attention import QuantumSuperpositionAttention
from .base import QuantumOperator


class MemoryEfficientQSA(QuantumSuperpositionAttention):
    """内存高效的 QSA 实现。

    优化策略:
    1. 分块计算注意力
    2. 及时释放中间张量
    3. 梯度检查点
    4. 精确控制内存分配
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        topk_ratio: float = 0.1,
        mode: str = 'topk',
        chunk_size: Optional[int] = None,
        use_checkpoint: bool = False
    ):
        super().__init__(dim, num_heads, topk_ratio, mode)
        self.chunk_size = chunk_size or 1024
        self.use_checkpoint = use_checkpoint

    def forward(
        self,
        x: torch.Tensor,
        training: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """内存高效的前向传播。"""
        if self.use_checkpoint and training:
            return torch.utils.checkpoint.checkpoint(
                self._forward_chunked, x, training
            )
        else:
            return self._forward_chunked(x, training)

    def _forward_chunked(
        self,
        x: torch.Tensor,
        training: bool
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """分块计算注意力。"""
        B, N, D = x.shape

        # 1. 投影 Q, K, V
        q = self.Wq(x)  # (B, N, D)
        k = self.Wk(x)  # (B, N, D)
        v = self.Wv(x)  # (B, N, D)

        # 2. 重塑为多头
        q = q.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. 分块计算注意力
        outputs = []
        metrics = {}

        for i in range(0, N, self.chunk_size):
            end_idx = min(i + self.chunk_size, N)
            chunk_q = q[:, :, i:end_idx, :]  # (B, H, chunk_size, d)

            # 计算注意力分数
            scores = (chunk_q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            # scores: (B, H, chunk_size, N)

            # Top-K 筛选
            if self.mode == 'topk':
                k = int(N * self.topk_ratio)
                topk_scores, topk_indices = scores.topk(k, dim=-1)

                # Softmax
                weights = torch.softmax(topk_scores, dim=-1)

                # 加权聚合
                chunk_v = v.gather(
                    2,
                    topk_indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
                )
                chunk_output = (weights.unsqueeze(-1) * chunk_v).sum(dim=-2)
            else:
                # Full attention
                weights = torch.softmax(scores, dim=-1)
                chunk_output = weights @ v

            outputs.append(chunk_output)

            # 及时释放中间张量
            del chunk_q, scores, weights

        # 4. 合并输出
        output = torch.cat(outputs, dim=2)  # (B, H, N, d)
        output = output.transpose(1, 2).reshape(B, N, D)

        # 5. 输出投影
        output = self.Wo(output)

        # 6. 度量统计
        with torch.no_grad():
            metrics['attention_entropy'] = self._compute_entropy(x, output)
            metrics['interference_phase_std'] = self._compute_interference_std(q, k)

        return output, metrics
```

#### 2. 内存监控系统

```python
# quantum_core/memory_monitor.py
"""内存监控系统。"""

import torch
import gc
from typing import Dict, Optional
from contextlib import contextmanager


class MemoryMonitor:
    """GPU/CPU 内存监控器。"""

    def __init__(self, device: torch.device):
        self.device = device
        self.device_type = device.type
        self.baseline = self._get_memory()

    def _get_memory(self) -> Dict[str, float]:
        """获取当前内存使用情况。"""
        stats = {}

        if self.device_type == 'cuda':
            allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            reserved = torch.cuda.memory_reserved(self.device) / 1024**3
            stats['allocated_gb'] = allocated
            stats['reserved_gb'] = reserved
            stats['max_allocated_gb'] = torch.cuda.max_memory_allocated(self.device) / 1024**3
        else:
            import psutil
            process = psutil.Process()
            mem_info = process.memory_info()
            stats['rss_gb'] = mem_info.rss / 1024**3

        return stats

    def snapshot(self) -> Dict[str, float]:
        """获取当前内存快照。"""
        current = self._get_memory()
        return {
            **current,
            'allocated_delta_gb': current.get('allocated_gb', 0) - self.baseline.get('allocated_gb', 0)
        }

    def reset(self) -> None:
        """重置基线。"""
        self.baseline = self._get_memory()

    @staticmethod
    def clear_cache() -> None:
        """清理缓存。"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @contextmanager
    def profile(self, name: str):
        """上下文管理器: 监控代码块的内存使用。"""
        print(f"[MemoryMonitor] 开始: {name}")
        self.reset()
        before = self.snapshot()

        try:
            yield
        finally:
            after = self.snapshot()
            print(f"[MemoryMonitor] 结束: {name}")
            print(f"  分配增量: {after['allocated_delta_gb']:.2f} GB")
            if 'max_allocated_gb' in after:
                print(f"  最大分配: {after['max_allocated_gb']:.2f} GB")


# 使用示例
def train_with_memory_monitoring():
    """训练时监控内存。"""
    monitor = MemoryMonitor(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    with monitor.profile("模型初始化"):
        model = QuantumArch(dim=512, num_layers=6).cuda()

    with monitor.profile("训练 epoch 1"):
        # ... 训练代码
        pass

    monitor.clear_cache()
```

---

## 并行计算优化

### 问题分析

当前并行计算问题:
1. 未充分利用 PyTorch 并行原语
2. 没有实现 pipeline parallel
3. 缺乏 tensor parallel 优化
4. 没有分布式训练支持

### 优化方案

#### 1. 数据并行优化

```python
# quantum_core/distributed.py
"""分布式训练支持。"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import Optional


class DistributedQuantumArch:
    """支持分布式训练的 QuantumArch。"""

    def __init__(
        self,
        config: QuantumArchConfig,
        rank: int,
        world_size: int,
        backend: str = 'nccl'
    ):
        self.rank = rank
        self.world_size = world_size
        self.config = config

        # 初始化进程组
        if not dist.is_initialized():
            dist.init_process_group(
                backend=backend,
                init_method='env://'
            )

        # 创建模型并包装为 DDP
        self.model = QuantumArch.from_config(config).to(rank)
        self.model = torch.nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=False
        )

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """单步训练。"""
        self.model.zero_grad()

        # 前向传播
        result = self.model({'inputs': batch['x']}, training=True)
        loss = compute_loss(result, batch['target'])

        # 反向传播
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        return {'loss': loss.item()}


def spawn_distributed_training(
    config: QuantumArchConfig,
    world_size: int,
    train_fn: callable
):
    """启动分布式训练。"""
    mp.spawn(
        train_fn,
        args=(config, world_size),
        nprocs=world_size,
        join=True
    )
```

#### 2. Pipeline Parallel 实现

```python
# quantum_core/pipeline.py
"""Pipeline Parallel 实现。"""

from typing import List, Tuple
import torch
import torch.nn as nn


class PipelineQuantumArch(nn.Module):
    """Pipeline Parallel 的 QuantumArch。

    将模型切分为多个阶段,每个阶段在独立的 GPU 上运行。
    """

    def __init__(
        self,
        config: QuantumArchConfig,
        num_stages: int,
        micro_batch_size: int
    ):
        super().__init__()
        self.num_stages = num_stages
        self.micro_batch_size = micro_batch_size

        # 创建模型
        full_model = QuantumArch.from_config(config)

        # 切分模型
        self.stages = self._split_model(full_model, num_stages)

    def _split_model(
        self,
        model: nn.Module,
        num_stages: int
    ) -> List[nn.Module]:
        """将模型切分为多个阶段。"""
        layers_per_stage = model.num_layers // num_stages
        stages = []

        current_blocks = []
        for i, block in enumerate(model.blocks):
            current_blocks.append(block)

            if len(current_blocks) == layers_per_stage or i == len(model.blocks) - 1:
                stage = nn.Sequential(*current_blocks)
                stages.append(stage)
                current_blocks = []

        return stages

    def forward(
        self,
        x: torch.Tensor,
        training: bool = True
    ) -> torch.Tensor:
        """Pipeline parallel 前向传播。"""
        # 将 batch 切分为 micro-batches
        B, N, D = x.shape
        num_micro_batches = B // self.micro_batch_size

        # 简化实现: 串行执行 micro-batches
        outputs = []
        for i in range(num_micro_batches):
            start_idx = i * self.micro_batch_size
            end_idx = (i + 1) * self.micro_batch_size
            micro_x = x[start_idx:end_idx, :, :]

            # 通过所有阶段
            for stage in self.stages:
                micro_x, _ = stage(micro_x, training=training)

            outputs.append(micro_x)

        return torch.cat(outputs, dim=0)
```

---

## 可扩展性设计

### 问题分析

当前可扩展性问题:
1. 新增机制需要修改多处代码
2. 缺乏插件系统
3. 没有配置热更新
4. 难以支持变体实验

### 优化方案

#### 1. 插件系统

```python
# quantum_core/plugins.py
"""插件系统。"""

from typing import Dict, Type, Any
from abc import ABC, abstractmethod
import importlib


class QuantumPlugin(ABC):
    """量子架构插件基类。"""

    @property
    @abstractmethod
    def name(self) -> str:
        """插件名称。"""
        pass

    @abstractmethod
    def register(self) -> None:
        """注册插件。"""
        pass

    @abstractmethod
    def unregister(self) -> None:
        """注销插件。"""
        pass


class PluginManager:
    """插件管理器。"""

    def __init__(self):
        self._plugins: Dict[str, QuantumPlugin] = {}

    def register(self, plugin: QuantumPlugin) -> None:
        """注册插件。"""
        if plugin.name in self._plugins:
            raise ValueError(f"插件 {plugin.name} 已存在")

        plugin.register()
        self._plugins[plugin.name] = plugin
        print(f"插件已注册: {plugin.name}")

    def unregister(self, name: str) -> None:
        """注销插件。"""
        if name not in self._plugins:
            raise ValueError(f"插件 {name} 不存在")

        self._plugins[name].unregister()
        del self._plugins[name]
        print(f"插件已注销: {name}")

    def get_plugin(self, name: str) -> QuantumPlugin:
        """获取插件。"""
        if name not in self._plugins:
            raise ValueError(f"插件 {name} 不存在")
        return self._plugins[name]

    def load_from_module(self, module_path: str) -> None:
        """从模块加载插件。"""
        module = importlib.import_module(module_path)

        for attr_name in dir(module):
            attr = getattr(module, attr_name)

            if (
                isinstance(attr, type) and
                issubclass(attr, QuantumPlugin) and
                attr != QuantumPlugin
            ):
                plugin = attr()
                self.register(plugin)


# 插件示例
class CustomActivationPlugin(QuantumPlugin):
    """自定义激活函数插件。"""

    @property
    def name(self) -> str:
        return "custom_activation"

    def register(self) -> None:
        # 注册新的激活函数到全局注册表
        from quantum_core import activations
        activations.register_activation("custom", self.custom_activation)

    def unregister(self) -> None:
        from quantum_core import activations
        activations.unregister_activation("custom")

    @staticmethod
    def custom_activation(x: torch.Tensor) -> torch.Tensor:
        """自定义激活函数。"""
        return torch.tanh(x) * torch.sigmoid(x)
```

---

## 架构决策记录

### ADR-001: 使用复数参数化

**状态**: 已接受

**背景**:
QuantumArch 需要表示量子态,量子态本质上是复数向量。

**决策**:
使用 PyTorch 的复数张量 (`torch.complex64`, `torch.complex128`) 作为核心数据类型。

**理由**:
1. 符合量子力学数学本质
2. 自然支持模长和相位
3. 可以直接应用酉矩阵变换

**后果**:
- 优点: 语义清晰,数学正确
- 缺点: 内存占用加倍,计算速度略慢

---

### ADR-002: Cayley 参数化保证酉性

**状态**: 已接受

**背景**:
需要保证权重矩阵的酉性约束。

**决策**:
使用 Cayley 变换参数化酉矩阵: `W = (I + iΩ/2)^-1(I - iΩ/2)`, 其中 Ω 是反对称实矩阵。

**理由**:
1. 严格的酉性保证
2. 参数空间可微
3. 计算开销可接受

**后果**:
- 优点: 酉性约束自动满足,无需额外正则化
- 缺点: 需要矩阵求逆,计算量略增

---

**版本**: v1.0
**维护**: 量子架构项目组
**更新**: 2026-03-21
