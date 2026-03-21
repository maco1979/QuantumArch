# QuantumArch 性能优化实践指南

**针对复数运算和 GPU 加速的深度优化策略**

---

## 目录

1. [CUDA 优化](#cuda-优化)
2. [混合精度训练](#混合精度训练)
3. [内核融合优化](#内核融合优化)
4. [内存访问优化](#内存访问优化)
5. [自动微分优化](#自动微分优化)
6. [性能分析与调优](#性能分析与调优)

---

## CUDA 优化

### 复数运算 CUDA Kernel

#### 1. 复数模长计算

```cuda
// quantum_core/kernels/complex_ops.cu
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// 复数模长计算 CUDA kernel
__global__ void complex_abs_kernel(
    const at::complex<float>* __restrict__ input,
    float* __restrict__ output,
    int64_t size
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        at::complex<float> z = input[idx];
        output[idx] = sqrtf(z.real() * z.real() + z.imag() * z.imag());
    }
}

// 批量复数模长计算
torch::Tensor complex_abs_cuda(torch::Tensor input) {
    TORCH_CHECK(input.is_complex(), "输入必须是复数张量");

    const int64_t size = input.numel();
    auto output = torch::empty(input.sizes(), input.options().dtype(torch::kFloat32));

    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    complex_abs_kernel<<<blocks, threads>>>(
        reinterpret_cast<at::complex<float>*>(input.data_ptr()),
        output.data_ptr<float>(),
        size
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA kernel 执行失败");

    return output;
}

// 复数点积 CUDA kernel
__global__ void complex_dot_product_kernel(
    const at::complex<float>* __restrict__ a,
    const at::complex<float>* __restrict__ b,
    at::complex<float>* __restrict__ output,
    int64_t size,
    int64_t stride
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < stride) {
        at::complex<float> sum = 0;

        for (int64_t i = 0; i < size; i++) {
            int64_t offset = i * stride + idx;
            // a * conjugate(b)
            sum += a[offset] * std::conj(b[offset]);
        }

        output[idx] = sum;
    }
}

// Born 归一化 CUDA kernel
__global__ void born_normalize_kernel(
    const at::complex<float>* __restrict__ input,
    float* __restrict__ output,
    int64_t size,
    int64_t stride
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < stride) {
        // 计算模长平方和
        float sum_sq = 0.0f;

        for (int64_t i = 0; i < size; i++) {
            int64_t offset = i * stride + idx;
            at::complex<float> z = input[offset];
            sum_sq += z.real() * z.real() + z.imag() * z.imag();
        }

        // 归一化
        float norm = sqrtf(sum_sq) + 1e-8f;

        for (int64_t i = 0; i < size; i++) {
            int64_t offset = i * stride + idx;
            at::complex<float> z = input[offset];
            output[offset] = (z.real() * z.real() + z.imag() * z.imag()) / (norm * norm);
        }
    }
}
```

#### 2. Python 绑定

```python
# quantum_core/kernels/complex_ops_binding.cpp
#include <torch/extension.h>
#include "complex_ops.cu"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("complex_abs", &complex_abs_cuda, "复数模长计算 (CUDA)");
    m.def("complex_dot_product", &complex_dot_product_cuda, "复数点积 (CUDA)");
    m.def("born_normalize", &born_normalize_cuda, "Born 归一化 (CUDA)");
}
```

#### 3. Setup 配置

```python
# quantum_core/kernels/setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='quantum_kernels',
    ext_modules=[
        CUDAExtension(
            name='quantum_kernels',
            sources=[
                'complex_ops_binding.cpp',
                'complex_ops.cu'
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '--use_fast_math']
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
```

### 优化策略

#### 1. 共享内存优化

```cuda
// 使用共享内存优化矩阵乘法
template<int BLOCK_SIZE>
__global__ void complex_matmul_shared_kernel(
    const at::complex<float>* __restrict__ A,
    const at::complex<float>* __restrict__ B,
    at::complex<float>* __restrict__ C,
    int M, int N, int K
) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ at::complex<float> As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ at::complex<float> Bs[BLOCK_SIZE][BLOCK_SIZE];

    at::complex<float> sum = 0;

    for (int tile = 0; tile < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; tile++) {
        // 加载到共享内存
        if (row < M && tile * BLOCK_SIZE + threadIdx.x < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + tile * BLOCK_SIZE + threadIdx.x];
        }

        if (col < N && tile * BLOCK_SIZE + threadIdx.y < K) {
            Bs[threadIdx.y][threadIdx.x] = B[(tile * BLOCK_SIZE + threadIdx.y) * N + col];
        }

        __syncthreads();

        // 计算
        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

#### 2. Tensor Core 利用

```python
# 使用 Tensor Core 加速矩阵乘法
import torch

def use_tensor_core_matmul():
    """启用 Tensor Core 加速。"""

    # 1. 使用半精度
    x_half = torch.randn(1024, 1024, dtype=torch.float16).cuda()
    w_half = torch.randn(1024, 1024, dtype=torch.float16).cuda()

    # 2. 确保维度对齐 (通常是 8 的倍数)
    if x_half.size(0) % 8 != 0 or x_half.size(1) % 8 != 0:
        # 填充到 8 的倍数
        pad_h = (8 - x_half.size(0) % 8) % 8
        pad_w = (8 - x_half.size(1) % 8) % 8
        x_half = torch.nn.functional.pad(x_half, (0, pad_w, 0, pad_h))

    # 3. 使用 GEMM (通用矩阵乘法)
    y = torch.nn.functional.linear(x_half, w_half)

    # 4. 复数版本 (使用两个实数矩阵)
    x_real = torch.randn(1024, 1024, dtype=torch.float16).cuda()
    x_imag = torch.randn(1024, 1024, dtype=torch.float16).cuda()
    w_real = torch.randn(1024, 1024, dtype=torch.float16).cuda()
    w_imag = torch.randn(1024, 1024, dtype=torch.float16).cuda()

    # 复数乘法: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
    ac = torch.nn.functional.linear(x_real, w_real)
    bd = torch.nn.functional.linear(x_imag, w_imag)
    ad = torch.nn.functional.linear(x_real, w_imag)
    bc = torch.nn.functional.linear(x_imag, w_real)

    y_real = ac - bd
    y_imag = ad + bc

    y_complex = torch.complex(y_real, y_imag)
```

---

## 混合精度训练

### 1. PyTorch AMP 实现

```python
# quantum_core/training/amp.py
"""混合精度训练实现。"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Any, Optional
from .optimizer import QGD


class AMPQuantumArchTrainer:
    """支持混合精度的 QuantumArch 训练器。"""

    def __init__(
        self,
        model: nn.Module,
        optimizer: QGD,
        config: Dict[str, Any]
    ):
        self.model = model
        self.optimizer = optimizer
        self.config = config

        # GradScaler 用于梯度缩放
        self.scaler = GradScaler(
            init_scale=2.**16,
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=2000
        )

        # 监控指标
        self.scale_history = []
        self.skip_steps = 0

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """单步训练 (混合精度)。"""
        self.model.zero_grad()

        # 自动混合精度前向传播
        with autocast():
            result = self.model({'inputs': batch['x']}, training=True)
            loss = self._compute_loss(result, batch['target'])

        # 反向传播 (使用缩放器)
        self.scaler.scale(loss).backward()

        # 梯度裁剪 (在 unscale 之前)
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # 更新参数 (使用缩放器)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # 记录缩放因子
        self.scale_history.append(self.scaler.get_scale())

        # 统计跳过的步数 (梯度溢出)
        if self.scaler.get_scale() < self.scale_history[-2] if len(self.scale_history) > 1 else float('inf'):
            self.skip_steps += 1

        return {
            'loss': loss.item(),
            'scale': self.scaler.get_scale(),
            'skip_rate': self.skip_steps / len(self.scale_history) if self.scale_history else 0.0
        }

    def _compute_loss(self, result: Dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        """计算损失。"""
        output = result['output']

        # 确保 target 维度匹配
        if output.shape != target.shape:
            target = target[..., :output.shape[-1]]

        # MSE 损失
        loss = nn.functional.mse_loss(output, target)

        # 可选: 添加酉性约束正则化
        if self.config.get('unitary_regularization', 0.0) > 0:
            unitary_report = self.model.get_unitarity_report()
            reg_loss = sum(unitary_report.values())
            loss += self.config['unitary_regularization'] * reg_loss

        return loss

    def state_dict(self) -> Dict[str, Any]:
        """保存训练状态。"""
        return {
            'scaler': self.scaler.state_dict(),
            'scale_history': self.scale_history,
            'skip_steps': self.skip_steps
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """加载训练状态。"""
        self.scaler.load_state_dict(state_dict['scaler'])
        self.scale_history = state_dict['scale_history']
        self.skip_steps = state_dict['skip_steps']
```

### 2. 复数混合精度策略

```python
# quantum_core/training/complex_amp.py
"""复数混合精度训练策略。"""

import torch
import torch.nn as nn
from typing import Optional


class ComplexMixedPrecisionStrategy:
    """复数混合精度训练策略。

    策略:
    1. 实部和虚部分别使用 FP16
    2. 梯度累积使用 FP32
    3. 关键操作 (如 Cayley 变换) 使用 FP32
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.fp32_params = set()

        # 识别需要 FP32 的参数
        for name, param in model.named_parameters():
            if self._requires_fp32(name):
                self.fp32_params.add(name)
                # 创建 FP32 副本
                param.data = param.data.float()
            else:
                # 转换为 FP16
                param.data = param.data.half()

    def _requires_fp32(self, param_name: str) -> bool:
        """判断参数是否需要 FP32。"""
        # Cayley 参数 (A 矩阵) 需要高精度
        if 'cayley' in param_name.lower() or 'omega' in param_name.lower():
            return True

        # 归一化层的参数
        if 'norm' in param_name.lower() or 'bias' in param_name.lower():
            return True

        return False

    def convert_to_amp(self, x: torch.Tensor) -> torch.Tensor:
        """将输入转换为混合精度。"""
        if x.is_complex():
            # 复数: 分别处理实部和虚部
            real = x.real.half()
            imag = x.imag.half()
            return torch.complex(real, imag)
        else:
            return x.half()

    def restore_precision(self, x: torch.Tensor) -> torch.Tensor:
        """恢复精度 (用于关键操作)。"""
        if x.is_complex():
            return torch.complex(x.real.float(), x.imag.float())
        else:
            return x.float()

    def sync_fp32_params(self) -> None:
        """同步 FP32 参数到 FP16。"""
        for name, param in self.model.named_parameters():
            if name in self.fp32_params:
                # FP32 参数在计算时使用,同步给 FP16 不会改变值
                pass
```

---

## 内核融合优化

### 1. QSA+QEL 融合内核

```python
# quantum_core/optimized/fused_ops.py
"""融合内核优化。"""

import torch
import torch.nn as nn
from typing import Tuple, Dict


class FusedQSA_QEL(nn.Module):
    """融合 QSA 和 QEL 的优化实现。

    融合优势:
    1. 减少 GPU kernel 启动次数
    2. 减少中间结果的内存读写
    3. 更好的缓存利用率
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        topk_ratio: float = 0.1,
        entanglement_mode: str = 'adaptive'
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.topk_ratio = topk_ratio

        # QSA 投影
        self.Wq = nn.Linear(dim, dim, bias=False)
        self.Wk = nn.Linear(dim, dim, bias=False)
        self.Wv = nn.Linear(dim, dim, bias=False)
        self.Wo = nn.Linear(dim, dim, bias=False)

        # QEL 投影 (用于纠缠)
        self.entangle_weight = nn.Parameter(torch.randn(dim, dim) * 0.02)

        # 纠缠强度预测器
        self.entanglement_predictor = nn.Sequential(
            nn.Linear(dim * 2, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        x: torch.Tensor,
        training: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """融合前向传播。"""
        B, N, D = x.shape

        # ===== 第一阶段: QSA =====
        # 投影
        q = self.Wq(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.Wk(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.Wv(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # 注意力分数
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Top-K 筛选
        k = int(N * self.topk_ratio)
        topk_scores, topk_indices = scores.topk(k, dim=-1)

        # 注意力权重
        weights = torch.softmax(topk_scores, dim=-1)

        # 加权聚合
        qsa_out = weights @ v.gather(
            2,
            topk_indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
        )
        qsa_out = qsa_out.transpose(1, 2).reshape(B, N, D)
        qsa_out = self.Wo(qsa_out)

        # ===== 第二阶段: QEL (与 QSA 融合) =====
        # 预测纠缠强度
        concat = torch.cat([x, qsa_out], dim=-1)  # (B, N, 2D)
        strength = self.entanglement_predictor(concat)  # (B, N, 1)

        # 纠缠变换 (融合)
        # 使用同一个 qsa_out 作为纠缠对象,减少计算
        entangled = x + strength * (qsa_out @ self.entangle_weight.T)

        # ===== 第三阶段: 融合输出 =====
        # 残差连接
        output = x + entangled

        # 度量统计
        metrics = {
            'attention_entropy': self._compute_entropy(weights),
            'entanglement_strength': strength.mean().item(),
            'topk_ratio': k / N
        }

        return output, metrics

    def _compute_entropy(self, weights: torch.Tensor) -> float:
        """计算注意力熵。"""
        entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=-1)
        return entropy.mean().item()
```

### 2. 多层融合

```python
# quantum_core/optimized/fused_block.py
"""多层融合优化。"""

import torch
import torch.nn as nn
from typing import Tuple, Dict


class FusedQuantumBlock(nn.Module):
    """融合多层计算的 QuantumBlock。

    融合策略:
    1. 合并 LayerNorm 和投影
    2. 合并残差连接
    3. 合并多个子模块的计算图
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_dim: int,
        topk_ratio: float = 0.1
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        # 合并的 LayerNorm+投影
        self.qsa_norm_proj = nn.LayerNorm(dim)
        self.qsa_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.qsa_output = nn.Linear(dim, dim, bias=False)

        # FFN 合并
        self.ffn_norm_proj = nn.LayerNorm(dim)
        self.ffn_up = nn.Linear(dim, ffn_dim)
        self.ffn_down = nn.Linear(ffn_dim, dim)

    def forward(
        self,
        x: torch.Tensor,
        training: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """融合前向传播。"""
        metrics = {}

        # ===== QSA + LayerNorm =====
        # 合并: LayerNorm -> QKV 投影
        normed = self.qsa_norm_proj(x)
        qkv = self.qsa_qkv(normed)
        q, k, v = qkv.chunk(3, dim=-1)

        # 注意力计算
        B, N, D = q.shape
        head_dim = D // self.num_heads

        q = q.reshape(B, N, self.num_heads, head_dim).transpose(1, 2)
        k = k.reshape(B, N, self.num_heads, head_dim).transpose(1, 2)
        v = v.reshape(B, N, self.num_heads, head_dim).transpose(1, 2)

        # 简化注意力 (Top-K 可选)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
        weights = torch.softmax(scores, dim=-1)
        attn_out = (weights @ v).transpose(1, 2).reshape(B, N, D)

        # 输出投影 + 残差
        qsa_out = self.qsa_output(attn_out)
        x = x + qsa_out  # 残差连接

        # ===== FFN + LayerNorm =====
        # 合并: LayerNorm -> FFN
        normed = self.ffn_norm_proj(x)
        ffn_out = self.ffn_up(normed)
        ffn_out = torch.gelu(ffn_out)
        ffn_out = self.ffn_down(ffn_out)

        x = x + ffn_out  # 残差连接

        metrics['block_output_norm'] = x.norm().item()

        return x, metrics
```

---

## 内存访问优化

### 1. 内存布局优化

```python
# quantum_core/optimized/memory_layout.py
"""内存布局优化。"""

import torch
import torch.nn as nn
from typing import Tuple


class MemoryLayoutOptimizer:
    """内存布局优化器。

    优化策略:
    1. 确保张量连续存储
    2. 优化访问模式 (cache-friendly)
    3. 减少内存拷贝
    """

    @staticmethod
    def ensure_contiguous(x: torch.Tensor) -> torch.Tensor:
        """确保张量连续存储。"""
        if not x.is_contiguous():
            x = x.contiguous()
        return x

    @staticmethod
    def optimize_attention_layout(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """优化注意力计算的内存布局。

        目标: 使 q, k, v 的内存访问模式对 GPU cache 友好。
        """
        # 1. 确保连续存储
        q = MemoryLayoutOptimizer.ensure_contiguous(q)
        k = MemoryLayoutOptimizer.ensure_contiguous(k)
        v = MemoryLayoutOptimizer.ensure_contiguous(v)

        # 2. 调整维度顺序 (如果需要)
        # PyTorch 默认: (batch, seq_len, dim, num_heads)
        # 优化后: (batch, num_heads, seq_len, dim)
        if q.dim() == 4 and q.shape[1] > q.shape[2]:
            # 当前是 (B, N, H, d), 调整为 (B, H, N, d)
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()

        return q, k, v

    @staticmethod
    def prefetch_tensor(x: torch.Tensor, stream: torch.cuda.Stream) -> None:
        """预取张量到 GPU。"""
        if x.device.type == 'cuda':
            with torch.cuda.stream(stream):
                # 触发内存传输
                _ = x.sum()

    @staticmethod
    def reuse_buffer(
        buffer: torch.Tensor,
        new_shape: Tuple[int, ...]
    ) -> torch.Tensor:
        """复用缓冲区。"""
        if buffer.numel() < new_shape[0] * new_shape[1]:
            # 需要分配更大的缓冲区
            buffer = torch.empty(new_shape, dtype=buffer.dtype, device=buffer.device)
        else:
            # reshape 复用
            buffer = buffer.view(new_shape)[:new_shape[0], :new_shape[1]]

        return buffer
```

### 2. 内存池管理

```python
# quantum_core/optimized/memory_pool.py
"""内存池管理。"""

import torch
from typing import Dict, Optional, Tuple


class MemoryPool:
    """GPU 内存池。

    预分配大块内存,按需分配小块,减少内存碎片。
    """

    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype = torch.complex64,
        max_blocks: int = 10
    ):
        self.device = device
        self.dtype = dtype
        self.max_blocks = max_blocks

        # 内存池: {shape: [buffer1, buffer2, ...]}
        self.pool: Dict[Tuple[int, ...], list] = {}

    def allocate(
        self,
        shape: Tuple[int, ...],
        requires_grad: bool = False
    ) -> torch.Tensor:
        """从内存池分配张量。"""
        shape_key = shape

        if shape_key in self.pool and len(self.pool[shape_key]) > 0:
            # 复用已有缓冲区
            buffer = self.pool[shape_key].pop()
            buffer.requires_grad = requires_grad
            return buffer
        else:
            # 分配新缓冲区
            return torch.empty(shape, dtype=self.dtype, device=self.device, requires_grad=requires_grad)

    def release(self, tensor: torch.Tensor) -> None:
        """释放张量到内存池。"""
        shape_key = tensor.shape

        if shape_key not in self.pool:
            self.pool[shape_key] = []

        # 限制池中块的数量
        if len(self.pool[shape_key]) < self.max_blocks:
            self.pool[shape_key].append(tensor.detach())
        else:
            # 超过限制,直接释放
            del tensor

    def clear(self) -> None:
        """清空内存池。"""
        for shape_key in self.pool:
            for buffer in self.pool[shape_key]:
                del buffer
        self.pool.clear()


# 使用示例
pool = MemoryPool(device=torch.device('cuda'))

# 分配
buffer = pool.allocate((1024, 1024))

# 使用
buffer = torch.randn(1024, 1024, out=buffer, dtype=torch.complex64)

# 释放
pool.release(buffer)
```

---

## 自动微分优化

### 1. 梯度计算优化

```python
# quantum_core/optimized/grad_ops.py
"""梯度计算优化。"""

import torch
import torch.nn as nn
from typing import Optional


class EfficientGradients:
    """高效的梯度计算策略。"""

    @staticmethod
    def detach_temporary(x: torch.Tensor) -> torch.Tensor:
        """临时张量detach,避免不必要的梯度传播。"""
        return x.detach().requires_grad_(x.requires_grad)

    @staticmethod
    def gradient_checkpointing(
        module: nn.Module,
        x: torch.Tensor,
        *args,
        **kwargs
    ) -> torch.Tensor:
        """使用梯度检查点节省内存。"""
        return torch.utils.checkpoint.checkpoint(module, x, *args, **kwargs)

    @staticmethod
    def selective_backward(
        loss: torch.Tensor,
        parameters: list[nn.Parameter],
        retain_graph: bool = False
    ) -> None:
        """选择性反向传播。"""
        loss.backward(retain_graph=retain_graph)

        # 清除不需要的梯度
        for param in parameters:
            if param.grad is not None and not param.requires_grad:
                param.grad = None

    @staticmethod
    def gradient_accumulation(
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        accumulation_steps: int,
        step: int
    ) -> None:
        """梯度累积。"""
        # 缩放损失
        loss = loss / accumulation_steps

        # 反向传播
        loss.backward()

        # 定期更新参数
        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
```

---

## 性能分析与调优

### 1. 性能分析工具

```python
# quantum_core/profiling/profiler.py
"""性能分析工具。"""

import torch
import torch.profiler as profiler
from typing import Optional, Dict, Any


class QuantumArchProfiler:
    """QuantumArch 性能分析器。"""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device
    ):
        self.model = model
        self.device = device

    def profile_forward(
        self,
        input_shape: tuple,
        num_runs: int = 100,
        warmup: int = 10
    ) -> Dict[str, float]:
        """分析前向传播性能。"""
        # 预热
        self.model.eval()
        with torch.no_grad():
            for _ in range(warmup):
                x = torch.randn(*input_shape, dtype=torch.complex64).to(self.device)
                _ = self.model({'inputs': x}, training=False)

        # 计时
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                x = torch.randn(*input_shape, dtype=torch.complex64).to(self.device)

                start_event.record()
                _ = self.model({'inputs': x}, training=False)
                end_event.record()

                torch.cuda.synchronize()
                times.append(start_event.elapsed_time(end_event))

        return {
            'mean_time_ms': sum(times) / len(times),
            'min_time_ms': min(times),
            'max_time_ms': max(times),
            'std_time_ms': (sum((t - sum(times) / len(times))**2 for t in times) / len(times))**0.5
        }

    def profile_with_profiler(
        self,
        input_shape: tuple,
        output_dir: str = './profiler_output'
    ) -> None:
        """使用 PyTorch Profiler 进行详细分析。"""
        self.model.eval()

        x = torch.randn(*input_shape, dtype=torch.complex64).to(self.device)

        with profiler.profile(
            activities=[
                profiler.ProfilerActivity.CPU,
                profiler.ProfilerActivity.CUDA
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            with torch.no_grad():
                _ = self.model({'inputs': x}, training=False)

        # 打印统计
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

        # 导出 Chrome Trace
        prof.export_chrome_trace(f"{output_dir}/trace.json")

        # 导出内存统计
        prof.export_memory_timeline(f"{output_dir}/memory_timeline.html")

    def profile_memory(
        self,
        input_shape: tuple
    ) -> Dict[str, float]:
        """分析内存使用。"""
        self.model.eval()

        torch.cuda.reset_peak_memory_stats()

        with torch.no_grad():
            x = torch.randn(*input_shape, dtype=torch.complex64).to(self.device)
            _ = self.model({'inputs': x}, training=False)

        return {
            'allocated_mb': torch.cuda.memory_allocated(self.device) / 1024**2,
            'reserved_mb': torch.cuda.memory_reserved(self.device) / 1024**2,
            'peak_allocated_mb': torch.cuda.max_memory_allocated(self.device) / 1024**2,
            'peak_reserved_mb': torch.cuda.max_memory_reserved(self.device) / 1024**2
        }
```

### 2. 性能调优建议

```python
# quantum_core/profiling/tuning.py
"""性能调优建议。"""

from typing import Dict, Any


class PerformanceTuner:
    """性能调优器。"""

    @staticmethod
    def analyze_profiling_results(results: Dict[str, Any]) -> list[str]:
        """分析性能分析结果并给出建议。"""
        suggestions = []

        # CUDA 时间分析
        cuda_time = results.get('cuda_time_total', 0)
        cpu_time = results.get('cpu_time_total', 0)

        if cuda_time > 0 and cpu_time / cuda_time > 0.5:
            suggestions.append("CPU 占用过高,考虑优化数据传输和预处理")

        # 内存分析
        memory_mb = results.get('memory_allocated_mb', 0)
        if memory_mb > 8000:  # 8GB
            suggestions.append("内存占用过高,考虑:")
            suggestions.append("  - 使用梯度检查点")
            suggestions.append("  - 减小 batch size")
            suggestions.append("  - 使用混合精度训练")

        # Kernel 分析
        if 'kernel_stats' in results:
            slow_kernels = results['kernel_stats'][:3]
            for kernel in slow_kernels:
                suggestions.append(f"慢速 Kernel: {kernel['name']} ({kernel['time']:.2f}ms)")

        return suggestions

    @staticmethod
    def recommend_batch_size(
        model: nn.Module,
        device: torch.device,
        initial_batch_size: int = 32,
        max_memory_mb: int = 16000
    ) -> int:
        """推荐最优 batch size。"""
        torch.cuda.reset_peak_memory_stats()

        # 二分查找
        low, high = 1, initial_batch_size * 2
        best_batch_size = 1

        while low <= high:
            mid = (low + high) // 2

            try:
                x = torch.randn(mid, 128, 512, dtype=torch.complex64).to(device)
                _ = model({'inputs': x}, training=False)
                torch.cuda.synchronize()

                memory_mb = torch.cuda.max_memory_allocated(device) / 1024**2

                if memory_mb < max_memory_mb:
                    best_batch_size = mid
                    low = mid + 1
                else:
                    high = mid - 1

                torch.cuda.empty_cache()
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    high = mid - 1
                    torch.cuda.empty_cache()
                else:
                    raise

        return best_batch_size
```

---

**版本**: v1.0
**维护**: 量子架构项目组
**更新**: 2026-03-21
