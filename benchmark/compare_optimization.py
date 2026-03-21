"""
性能优化对比测试

对比优化前后的性能差异：
1. QSA Attention
2. 复数运算
3. 融合算子

运行: python benchmark/compare_optimization.py
"""

import torch
import time
import sys
from typing import Dict, Any
import json

sys.path.insert(0, r"e:\量子架构")

from quantum_core import (
    QuantumSuperpositionAttention as QSA_Original,
    born_normalize,
    von_neumann_entropy,
    ComplexEmbedding,
)
from quantum_core.attention_optimized import (
    QuantumSuperpositionAttention as QSA_Optimized,
    fused_qkv_attention,
)
from quantum_core.complex_ops_optimized import (
    born_normalize as born_normalize_optimized,
    von_neumann_entropy as von_neumann_entropy_optimized,
    fused_born_entropy,
    fused_attention_score,
    PerformanceProfiler,
)


def benchmark_function(func, *args, warmup=10, num_runs=100, **kwargs):
    """基准测试函数。"""
    # 预热
    for _ in range(warmup):
        _ = func(*args, **kwargs)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # 计时
    start = time.perf_counter()
    for _ in range(num_runs):
        _ = func(*args, **kwargs)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    elapsed = time.perf_counter() - start
    return elapsed / num_runs * 1000  # ms


def test_qsa_optimization(device):
    """测试 QSA 优化效果。"""
    print("\n[1/4] QSA Attention 优化对比...")
    
    batch_size = 4
    seq_len = 128
    dim = 256
    num_heads = 8
    
    # 创建输入
    x = torch.randn(batch_size, seq_len, dim, dtype=torch.complex64, device=device)
    
    # 原始版本
    qsa_orig = QSA_Original(dim=dim, num_heads=num_heads, mode='topk', topk_ratio=0.1).to(device)
    time_orig = benchmark_function(qsa_orig, x, training=True, warmup=5, num_runs=20)
    
    # 优化版本
    qsa_opt = QSA_Optimized(dim=dim, num_heads=num_heads, mode='topk', topk_ratio=0.1).to(device)
    time_opt = benchmark_function(qsa_opt, x, training=True, warmup=5, num_runs=20)
    
    speedup = time_orig / time_opt
    
    print(f"  原始版本: {time_orig:.2f}ms")
    print(f"  优化版本: {time_opt:.2f}ms")
    print(f"  加速比: {speedup:.2f}x")
    
    return {'original_ms': time_orig, 'optimized_ms': time_opt, 'speedup': speedup}


def test_born_normalize(device):
    """测试 Born 归一化优化。"""
    print("\n[2/4] Born 归一化优化对比...")
    
    batch = 8
    seq = 64
    dim = 512
    
    # 创建输入
    z = torch.randn(batch, seq, dim, dtype=torch.complex64, device=device)
    
    # 原始版本
    time_orig = benchmark_function(born_normalize, z, dim=-1, warmup=10, num_runs=100)
    
    # 优化版本
    time_opt = benchmark_function(born_normalize_optimized, z, dim=-1, warmup=10, num_runs=100)
    
    speedup = time_orig / time_opt
    
    print(f"  原始版本: {time_orig:.4f}ms")
    print(f"  优化版本: {time_opt:.4f}ms")
    print(f"  加速比: {speedup:.2f}x")
    
    return {'original_ms': time_orig, 'optimized_ms': time_opt, 'speedup': speedup}


def test_von_neumann_entropy(device):
    """测试冯诺依曼熵优化。"""
    print("\n[3/4] 冯诺依曼熵优化对比...")
    
    batch = 8
    seq = 64
    dim = 512
    
    # 创建归一化概率
    probs = torch.rand(batch, seq, dim, device=device)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    
    # 原始版本
    time_orig = benchmark_function(von_neumann_entropy, probs, dim=-1, warmup=10, num_runs=100)
    
    # 优化版本
    time_opt = benchmark_function(von_neumann_entropy_optimized, probs, dim=-1, warmup=10, num_runs=100)
    
    speedup = time_orig / time_opt
    
    print(f"  原始版本: {time_orig:.4f}ms")
    print(f"  优化版本: {time_opt:.4f}ms")
    print(f"  加速比: {speedup:.2f}x")
    
    return {'original_ms': time_orig, 'optimized_ms': time_opt, 'speedup': speedup}


def test_fused_operators(device):
    """测试融合算子。"""
    print("\n[4/4] 融合算子性能测试...")
    
    batch = 4
    num_heads = 8
    seq_len = 64
    head_dim = 32
    
    # 创建 Q, K, V
    Q = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.complex64, device=device)
    K = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.complex64, device=device)
    V = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.complex64, device=device)
    
    scale = head_dim ** -0.5
    
    # 融合算子
    time_fused = benchmark_function(
        fused_qkv_attention, Q, K, V, scale, 0.1, True,
        warmup=10, num_runs=50
    )
    
    print(f"  融合算子: {time_fused:.2f}ms")
    
    return {'fused_ms': time_fused}


def test_memory_usage(device):
    """测试内存使用。"""
    print("\n[内存] 内存使用对比...")
    
    batch_size = 8
    seq_len = 256
    dim = 512
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    
    # 创建输入
    x = torch.randn(batch_size, seq_len, dim, dtype=torch.complex64, device=device)
    
    # 原始 QSA
    qsa_orig = QSA_Original(dim=dim, num_heads=8, mode='topk').to(device)
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    _ = qsa_orig(x, training=True)
    
    if torch.cuda.is_available():
        mem_orig = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
    else:
        mem_orig = 0
    
    # 优化 QSA
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    qsa_opt = QSA_Optimized(dim=dim, num_heads=8, mode='topk').to(device)
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    _ = qsa_opt(x, training=True)
    
    if torch.cuda.is_available():
        mem_opt = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
    else:
        mem_opt = 0
    
    print(f"  原始版本: {mem_orig:.2f} MB")
    print(f"  优化版本: {mem_opt:.2f} MB")
    if mem_orig > 0:
        print(f"  内存减少: {(1 - mem_opt/mem_orig)*100:.1f}%")
    
    return {'original_mb': mem_orig, 'optimized_mb': mem_opt}


def test_torch_compile(device):
    """测试 torch.compile 加速。"""
    print("\n[torch.compile] 编译优化测试...")
    
    batch_size = 4
    seq_len = 64
    dim = 256
    
    x = torch.randn(batch_size, seq_len, dim, dtype=torch.complex64, device=device)
    
    # 未编译
    qsa = QSA_Optimized(dim=dim, num_heads=8, mode='topk').to(device)
    time_no_compile = benchmark_function(qsa, x, training=False, warmup=3, num_runs=10)
    
    # 编译版本（如果支持）
    if hasattr(torch, 'compile'):
        try:
            qsa_compiled = torch.compile(qsa, mode='default')
            # 编译需要时间，这里只测一次
            _ = qsa_compiled(x, training=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            time_compile = benchmark_function(qsa_compiled, x, training=False, warmup=3, num_runs=10)
            speedup = time_no_compile / time_compile
            print(f"  未编译: {time_no_compile:.2f}ms")
            print(f"  编译后: {time_compile:.2f}ms")
            print(f"  加速比: {speedup:.2f}x")
            return {'no_compile_ms': time_no_compile, 'compile_ms': time_compile, 'speedup': speedup}
        except Exception as e:
            print(f"  torch.compile 不支持: {e}")
            return {}
    else:
        print("  PyTorch 版本不支持 torch.compile")
        return {}


def main():
    """主函数。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("="*70)
    print("  QuantumArch 性能优化对比测试")
    print("="*70)
    print(f"设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    results = {}
    
    # 运行所有测试
    results['qsa'] = test_qsa_optimization(device)
    results['born_normalize'] = test_born_normalize(device)
    results['entropy'] = test_von_neumann_entropy(device)
    results['fused'] = test_fused_operators(device)
    results['memory'] = test_memory_usage(device)
    results['compile'] = test_torch_compile(device)
    
    # 汇总
    print("\n" + "="*70)
    print("  优化效果汇总")
    print("="*70)
    
    total_speedup = 1.0
    count = 0
    
    for name, result in results.items():
        if 'speedup' in result and result['speedup'] > 0:
            print(f"{name}: {result['speedup']:.2f}x 加速")
            total_speedup *= result['speedup']
            count += 1
    
    if count > 0:
        avg_speedup = total_speedup ** (1/count)
        print(f"\n平均加速比: {avg_speedup:.2f}x")
    
    # 保存结果
    with open('benchmark_optimization_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: benchmark_optimization_results.json")
    
    return results


if __name__ == "__main__":
    main()
