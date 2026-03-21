"""
QuantumArch 完整性能基准测试

测试项目:
1. 各模块延迟和吞吐量
2. 内存占用
3. GPU 利用率
4. 扩展性测试
5. 与 Transformer 对比

运行: python benchmark/full_benchmark.py
"""

import torch
import time
import json
import sys
import psutil
import os
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
import numpy as np

sys.path.insert(0, r"e:\量子架构")

from quantum_core import (
    QuantumArch,
    QuantumSuperpositionAttention,
    QuantumEntanglementLayer,
    QuantumCollapseInference,
    QuantumFFN,
    QuantumBlock,
    CayleyLinear,
    ComplexLayerNorm,
)
from quantum_core.embedding import ComplexEmbedding


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    name: str
    batch_size: int
    seq_len: int
    dim: int
    latency_ms: float
    throughput_samples_per_sec: float
    memory_mb: float
    gpu_utilization_percent: float = 0.0


class PerformanceBenchmark:
    """性能基准测试类"""

    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.results: List[BenchmarkResult] = []
        
    def get_memory_usage(self) -> float:
        """获取当前内存使用 (MB)"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

    def benchmark_cayley_linear(self, dim: int, batch_size: int = 1) -> BenchmarkResult:
        """测试 CayleyLinear 性能"""
        layer = CayleyLinear(dim, dim).to(self.device)
        x = torch.randn(batch_size, dim, dtype=torch.complex64, device=self.device)
        
        # 预热
        for _ in range(10):
            _ = layer(x)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # 计时
        num_runs = 100
        start = time.perf_counter()
        
        for _ in range(num_runs):
            _ = layer(x)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        elapsed = time.perf_counter() - start
        latency_ms = (elapsed / num_runs) * 1000
        throughput = batch_size * num_runs / elapsed
        
        return BenchmarkResult(
            name="CayleyLinear",
            batch_size=batch_size,
            seq_len=1,
            dim=dim,
            latency_ms=latency_ms,
            throughput_samples_per_sec=throughput,
            memory_mb=self.get_memory_usage(),
        )

    def benchmark_qsa(
        self, 
        dim: int, 
        num_heads: int, 
        seq_len: int, 
        batch_size: int,
        mode: str = "topk",
        topk_ratio: float = 0.1,
    ) -> BenchmarkResult:
        """测试 QSA 性能"""
        qsa = QuantumSuperpositionAttention(
            dim=dim,
            num_heads=num_heads,
            mode=mode,
            topk_ratio=topk_ratio,
        ).to(self.device)
        
        x = torch.randn(batch_size, seq_len, dim, dtype=torch.complex64, device=self.device)
        
        # 预热
        for _ in range(5):
            _ = qsa(x, training=True)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # 计时
        num_runs = 20
        start = time.perf_counter()
        
        for _ in range(num_runs):
            _ = qsa(x, training=True)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        elapsed = time.perf_counter() - start
        latency_ms = (elapsed / num_runs) * 1000
        throughput = batch_size * num_runs / elapsed
        
        return BenchmarkResult(
            name=f"QSA_{mode}",
            batch_size=batch_size,
            seq_len=seq_len,
            dim=dim,
            latency_ms=latency_ms,
            throughput_samples_per_sec=throughput,
            memory_mb=self.get_memory_usage(),
        )

    def benchmark_full_model(
        self,
        dim: int,
        num_layers: int,
        num_heads: int,
        seq_len: int,
        batch_size: int,
    ) -> BenchmarkResult:
        """测试完整模型性能"""
        model = QuantumArch(
            dim=dim,
            num_layers=num_layers,
            num_heads=num_heads,
            vocab_size=0,
            direct_input=True,
        ).to(self.device)
        
        x = torch.randn(batch_size, seq_len, dim, device=self.device)
        
        # 预热
        for _ in range(3):
            _ = model(x, training=False)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # 计时
        num_runs = 10
        start = time.perf_counter()
        
        for _ in range(num_runs):
            _ = model(x, training=False)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        elapsed = time.perf_counter() - start
        latency_ms = (elapsed / num_runs) * 1000
        throughput = batch_size * num_runs / elapsed
        
        return BenchmarkResult(
            name="QuantumArch_Full",
            batch_size=batch_size,
            seq_len=seq_len,
            dim=dim,
            latency_ms=latency_ms,
            throughput_samples_per_sec=throughput,
            memory_mb=self.get_memory_usage(),
        )

    def benchmark_scaling(
        self,
        dim: int,
        num_heads: int,
        seq_lens: List[int],
        batch_size: int = 4,
    ) -> List[BenchmarkResult]:
        """测试序列长度扩展性"""
        results = []
        
        for seq_len in seq_lens:
            result = self.benchmark_qsa(
                dim=dim,
                num_heads=num_heads,
                seq_len=seq_len,
                batch_size=batch_size,
                mode="topk",
            )
            results.append(result)
            print(f"  seq_len={seq_len}: {result.latency_ms:.2f}ms")
        
        return results

    def compare_transformer(
        self,
        dim: int,
        num_heads: int,
        seq_len: int,
        batch_size: int,
    ) -> Dict[str, BenchmarkResult]:
        """与标准 Transformer 对比"""
        import torch.nn as nn
        
        # 标准 Transformer 注意力
        class StandardAttention(nn.Module):
            def __init__(self, dim, num_heads):
                super().__init__()
                self.num_heads = num_heads
                self.head_dim = dim // num_heads
                self.Wq = nn.Linear(dim, dim)
                self.Wk = nn.Linear(dim, dim)
                self.Wv = nn.Linear(dim, dim)
                self.Wo = nn.Linear(dim, dim)
            
            def forward(self, x):
                B, N, D = x.shape
                Q = self.Wq(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
                K = self.Wk(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
                V = self.Wv(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
                
                attn = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
                attn = attn.transpose(1, 2).contiguous().view(B, N, D)
                return self.Wo(attn)
        
        # 标准 Transformer
        transformer = StandardAttention(dim, num_heads).to(self.device)
        x = torch.randn(batch_size, seq_len, dim, device=self.device)
        
        # 预热
        for _ in range(5):
            _ = transformer(x)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # 计时
        num_runs = 20
        start = time.perf_counter()
        
        for _ in range(num_runs):
            _ = transformer(x)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        elapsed = time.perf_counter() - start
        transformer_latency = (elapsed / num_runs) * 1000
        transformer_throughput = batch_size * num_runs / elapsed
        
        transformer_result = BenchmarkResult(
            name="Transformer_Standard",
            batch_size=batch_size,
            seq_len=seq_len,
            dim=dim,
            latency_ms=transformer_latency,
            throughput_samples_per_sec=transformer_throughput,
            memory_mb=self.get_memory_usage(),
        )
        
        # QuantumArch
        qa_result = self.benchmark_qsa(
            dim=dim,
            num_heads=num_heads,
            seq_len=seq_len,
            batch_size=batch_size,
            mode="topk",
        )
        
        return {
            "transformer": transformer_result,
            "quantumarch": qa_result,
        }

    def run_all_benchmarks(self) -> Dict[str, Any]:
        """运行所有基准测试"""
        print("\n" + "="*70)
        print("  QuantumArch 性能基准测试")
        print("="*70)
        
        results = {}
        
        # 1. CayleyLinear
        print("\n[1/5] CayleyLinear 基准测试...")
        results["cayley"] = self.benchmark_cayley_linear(dim=512, batch_size=32)
        print(f"  延迟: {results['cayley'].latency_ms:.3f}ms")
        print(f"  吞吐量: {results['cayley'].throughput_samples_per_sec:.2f} samples/s")
        
        # 2. QSA 不同模式
        print("\n[2/5] QSA 模式对比...")
        for mode in ["full", "topk"]:
            result = self.benchmark_qsa(
                dim=256, num_heads=8, seq_len=128, batch_size=4, mode=mode
            )
            results[f"qsa_{mode}"] = result
            print(f"  {mode}: {result.latency_ms:.2f}ms")
        
        # 3. 完整模型
        print("\n[3/5] 完整模型基准测试...")
        results["full_model"] = self.benchmark_full_model(
            dim=256, num_layers=6, num_heads=8, seq_len=64, batch_size=4
        )
        print(f"  延迟: {results['full_model'].latency_ms:.2f}ms")
        print(f"  吞吐量: {results['full_model'].throughput_samples_per_sec:.2f} samples/s")
        
        # 4. 扩展性测试
        print("\n[4/5] 序列长度扩展性测试...")
        results["scaling"] = self.benchmark_scaling(
            dim=256, num_heads=8, seq_lens=[32, 64, 128, 256], batch_size=4
        )
        
        # 5. 与 Transformer 对比
        print("\n[5/5] 与 Transformer 对比...")
        comparison = self.compare_transformer(
            dim=256, num_heads=8, seq_len=128, batch_size=4
        )
        results["comparison"] = comparison
        
        speedup = comparison["transformer"].latency_ms / comparison["quantumarch"].latency_ms
        print(f"  Transformer: {comparison['transformer'].latency_ms:.2f}ms")
        print(f"  QuantumArch: {comparison['quantumarch'].latency_ms:.2f}ms")
        print(f"  加速比: {speedup:.2f}x")
        
        return results

    def save_results(self, results: Dict[str, Any], output_path: str = "benchmark_results.json"):
        """保存结果到文件"""
        # 转换 dataclass 为 dict
        def convert(obj):
            if isinstance(obj, BenchmarkResult):
                return asdict(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(i) for i in obj]
            return obj
        
        converted = convert(results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(converted, f, indent=2, ensure_ascii=False)
        
        print(f"\n结果已保存到: {output_path}")


def main():
    """主函数"""
    print(f"设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    benchmark = PerformanceBenchmark()
    results = benchmark.run_all_benchmarks()
    benchmark.save_results(results)
    
    print("\n" + "="*70)
    print("  基准测试完成!")
    print("="*70)


if __name__ == "__main__":
    main()
