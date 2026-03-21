"""
性能基准测试框架
对比 QSA/QEL/QCI 与标准 Transformer 组件的计算效率
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import math


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    test_name: str
    quantum_time_ms: float
    baseline_time_ms: float
    speedup: float
    quantum_memory_mb: float
    baseline_memory_mb: float
    memory_ratio: float
    metrics: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# 标准 Transformer 组件（Baseline）
# ============================================================================

class StandardAttention(nn.Module):
    """标准多头注意力 (O(n²))"""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class StandardFFN(nn.Module):
    """标准前馈网络"""

    def __init__(self, dim: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class StandardBlock(nn.Module):
    """标准 Transformer Block (Pre-Norm)"""

    def __init__(self, dim: int, num_heads: int, ffn_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = StandardAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = StandardFFN(dim, ffn_dim)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ============================================================================
# 量子架构组件
# ============================================================================

try:
    from quantum_core import (
        QuantumSuperpositionAttention, QuantumEntanglementLayer,
        QuantumCollapseInference, QuantumFFN, QuantumBlock, check_unitarity
    )
    QUANTUM_CORE_AVAILABLE = True
except ImportError:
    QUANTUM_CORE_AVAILABLE = False
    print("警告: quantum_core 模块不可用，量子组件测试将跳过")


# ============================================================================
# 基准测试工具
# ============================================================================

class BenchmarkRunner:
    """基准测试运行器"""

    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.results: List[BenchmarkResult] = []

    def measure_time(self, model: nn.Module, x: torch.Tensor,
                     warmup: int = 10, repeats: int = 100) -> Tuple[float, float]:
        """测量模型前向传播时间"""
        model = model.to(self.device)
        x = x.to(self.device)

        # 预热
        for _ in range(warmup):
            with torch.no_grad():
                _ = model(x)

        if self.device == 'cuda':
            torch.cuda.synchronize()

        # 计时
        start = time.time()
        for _ in range(repeats):
            with torch.no_grad():
                _ = model(x)
        if self.device == 'cuda':
            torch.cuda.synchronize()

        elapsed = (time.time() - start) / repeats

        # 测量内存
        if self.device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                _ = model(x)
            if self.device == 'cuda':
                torch.cuda.synchronize()
            memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        else:
            memory_mb = 0.0

        return elapsed * 1000, memory_mb  # ms, MB

    def benchmark_qsa_vs_attention(self, dims: List[int], seq_lengths: List[int]):
        """对比 QSA 与标准注意力"""
        print("\n" + "=" * 80)
        print("QSA vs 标准注意力 基准测试")
        print("=" * 80)

        for dim in dims:
            for seq_len in seq_lengths:
                num_heads = 8
                topk_ratio = 0.1

                # 标准注意力
                attn = StandardAttention(dim, num_heads)
                x = torch.randn(2, seq_len, dim)
                baseline_time, baseline_mem = self.measure_time(attn, x)

                # QSA
                if QUANTUM_CORE_AVAILABLE:
                    qsa = QuantumSuperpositionAttention(
                        dim=dim,
                        num_heads=num_heads,
                        topk_ratio=topk_ratio,
                        mode='topk'
                    )
                    x_complex = torch.randn(2, seq_len, dim, dtype=torch.cfloat)
                    quantum_time, quantum_mem = self.measure_time(qsa, x_complex)

                    speedup = baseline_time / quantum_time
                    memory_ratio = quantum_mem / baseline_mem if baseline_mem > 0 else 0

                    result = BenchmarkResult(
                        test_name=f"QSA_vs_Attention_d{dim}_n{seq_len}",
                        quantum_time_ms=quantum_time,
                        baseline_time_ms=baseline_time,
                        speedup=speedup,
                        quantum_memory_mb=quantum_mem,
                        baseline_memory_mb=baseline_mem,
                        memory_ratio=memory_ratio,
                        metrics={
                            'dim': dim,
                            'seq_len': seq_len,
                            'num_heads': num_heads,
                            'topk_ratio': topk_ratio
                        }
                    )
                    self.results.append(result)

                    print(f"  dim={dim}, seq_len={seq_len}:")
                    print(f"    基准: {baseline_time:.3f}ms, {baseline_mem:.1f}MB")
                    print(f"    QSA:   {quantum_time:.3f}ms, {quantum_mem:.1f}MB")
                    print(f"    加速:  {speedup:.2f}×, 内存比: {memory_ratio:.2f}×")

    def benchmark_qel_vs_residual(self, dims: List[int], seq_lengths: List[int]):
        """对比 QEL 与残差连接"""
        print("\n" + "=" * 80)
        print("QEL vs 残差连接 基准测试")
        print("=" * 80)

        for dim in dims:
            for seq_len in seq_lengths:
                # 残差连接 (简单作为基线)
                x = torch.randn(2, seq_len, dim)
                start = time.time()
                for _ in range(100):
                    _ = x + x  # 最简残差
                residual_time = (time.time() - start) / 100 * 1000  # ms

                # QEL
                if QUANTUM_CORE_AVAILABLE:
                    qel = QuantumEntanglementLayer(dim=dim, num_qubits=min(8, dim // 2))
                    x_complex = torch.randn(2, seq_len, dim, dtype=torch.cfloat)
                    quantum_time, quantum_mem = self.measure_time(qel, x_complex)

                    speedup = residual_time / quantum_time

                    result = BenchmarkResult(
                        test_name=f"QEL_vs_Residual_d{dim}_n{seq_len}",
                        quantum_time_ms=quantum_time,
                        baseline_time_ms=residual_time,
                        speedup=speedup,
                        quantum_memory_mb=quantum_mem,
                        baseline_memory_mb=0.0,
                        memory_ratio=0.0,
                        metrics={
                            'dim': dim,
                            'seq_len': seq_len
                        }
                    )
                    self.results.append(result)

                    print(f"  dim={dim}, seq_len={seq_len}:")
                    print(f"    残差: {residual_time:.3f}ms")
                    print(f"    QEL:  {quantum_time:.3f}ms")
                    print(f"    比率:  {quantum_time / residual_time:.2f}×")

    def benchmark_ffn_q_vs_ffn(self, dims: List[int]):
        """对比 FFN_Q 与标准 FFN"""
        print("\n" + "=" * 80)
        print("FFN_Q vs 标准 FFN 基准测试")
        print("=" * 80)

        for dim in dims:
            ffn_dim = dim * 4
            seq_len = 128

            # 标准 FFN
            ffn = StandardFFN(dim, ffn_dim)
            x = torch.randn(2, seq_len, dim)
            baseline_time, baseline_mem = self.measure_time(ffn, x)

            # FFN_Q
            if QUANTUM_CORE_AVAILABLE:
                ffn_q = QuantumFFN(dim=dim, ffn_dim=ffn_dim, activation='modrelu')
                x_complex = torch.randn(2, seq_len, dim, dtype=torch.cfloat)
                quantum_time, quantum_mem = self.measure_time(ffn_q, x_complex)

                speedup = baseline_time / quantum_time
                memory_ratio = quantum_mem / baseline_mem if baseline_mem > 0 else 0

                result = BenchmarkResult(
                    test_name=f"FFN_Q_vs_FFN_d{dim}",
                    quantum_time_ms=quantum_time,
                    baseline_time_ms=baseline_time,
                    speedup=speedup,
                    quantum_memory_mb=quantum_mem,
                    baseline_memory_mb=baseline_mem,
                    memory_ratio=memory_ratio,
                    metrics={
                        'dim': dim,
                        'ffn_dim': ffn_dim
                    }
                )
                self.results.append(result)

                print(f"  dim={dim}:")
                print(f"    基准: {baseline_time:.3f}ms, {baseline_mem:.1f}MB")
                print(f"    FFN_Q: {quantum_time:.3f}ms, {quantum_mem:.1f}MB")
                print(f"    加速:  {speedup:.2f}×, 内存比: {memory_ratio:.2f}×")

    def benchmark_qci_early_exit(self, dims: List[int], seq_lengths: List[int]):
        """测试 QCI 早退机制的效果"""
        if not QUANTUM_CORE_AVAILABLE:
            return

        print("\n" + "=" * 80)
        print("QCI 早退机制 效果测试")
        print("=" * 80)

        for dim in dims:
            for seq_len in seq_lengths:
                qci = QuantumCollapseInference(dim=dim)
                x = torch.randn(2, seq_len, dim, dtype=torch.cfloat)

                # 正常模式
                qci.training = False
                with torch.no_grad():
                    output_normal, metrics_normal = qci(x)

                # 低阈值模式（更多早退）
                qci.threshold = 0.3
                with torch.no_grad():
                    output_aggressive, metrics_aggressive = qci(x)

                early_exit_rate = metrics_aggressive.get('early_exit_rate', 0)

                print(f"  dim={dim}, seq_len={seq_len}:")
                print(f"    早退率 (threshold=0.3): {early_exit_rate:.2%}")

                # 测量时间差异
                normal_time, _ = self.measure_time(
                    QuantumCollapseInference(dim=dim),
                    torch.randn(2, seq_len, dim, dtype=torch.cfloat),
                    repeats=50
                )

                result = BenchmarkResult(
                    test_name=f"QCI_EarlyExit_d{dim}_n{seq_len}",
                    quantum_time_ms=normal_time,
                    baseline_time_ms=normal_time,  # 基线即自己
                    speedup=1.0,
                    quantum_memory_mb=0.0,
                    baseline_memory_mb=0.0,
                    memory_ratio=0.0,
                    metrics={
                        'dim': dim,
                        'seq_len': seq_len,
                        'early_exit_rate': early_exit_rate
                    }
                )
                self.results.append(result)

    def benchmark_block_comparison(self, configs: List[Dict[str, int]]):
        """完整 QuantumBlock vs StandardBlock 对比"""
        if not QUANTUM_CORE_AVAILABLE:
            return

        print("\n" + "=" * 80)
        print("完整 Block 对比 基准测试")
        print("=" * 80)

        for cfg in configs:
            dim = cfg['dim']
            num_heads = cfg['num_heads']
            ffn_dim = cfg['ffn_dim']
            seq_len = cfg['seq_len']

            # 标准 Block
            standard_block = StandardBlock(dim, num_heads, ffn_dim)
            x = torch.randn(2, seq_len, dim)
            baseline_time, baseline_mem = self.measure_time(standard_block, x)

            # QuantumBlock
            quantum_block = QuantumBlock(
                dim=dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                topk_ratio=0.1,
                collapse_enabled=False  # 先不测试坍缩
            )
            x_complex = torch.randn(2, seq_len, dim, dtype=torch.cfloat)
            quantum_time, quantum_mem = self.measure_time(quantum_block, x_complex)

            speedup = baseline_time / quantum_time
            memory_ratio = quantum_mem / baseline_mem if baseline_mem > 0 else 0

            result = BenchmarkResult(
                test_name=f"Block_Comparison_d{dim}_n{seq_len}",
                quantum_time_ms=quantum_time,
                baseline_time_ms=baseline_time,
                speedup=speedup,
                quantum_memory_mb=quantum_mem,
                baseline_memory_mb=baseline_mem,
                memory_ratio=memory_ratio,
                metrics=cfg
            )
            self.results.append(result)

            print(f"  dim={dim}, seq_len={seq_len}:")
            print(f"    标准: {baseline_time:.3f}ms, {baseline_mem:.1f}MB")
            print(f"    量子: {quantum_time:.3f}ms, {quantum_mem:.1f}MB")
            print(f"    加速:  {speedup:.2f}×, 内存比: {memory_ratio:.2f}×")

    def run_all_benchmarks(self):
        """运行所有基准测试"""
        print("=" * 80)
        print("量子架构性能基准测试")
        print("=" * 80)
        print(f"设备: {self.device}")
        print()

        # 测试配置
        dims = [128, 256, 512]
        seq_lengths = [128, 512, 1024]

        # 运行各类测试
        self.benchmark_qsa_vs_attention(dims, seq_lengths)
        self.benchmark_qel_vs_residual(dims, seq_lengths)
        self.benchmark_ffn_q_vs_ffn(dims)
        self.benchmark_qci_early_exit(dims, seq_lengths)

        block_configs = [
            {'dim': 128, 'num_heads': 4, 'ffn_dim': 512, 'seq_len': 128},
            {'dim': 256, 'num_heads': 8, 'ffn_dim': 1024, 'seq_len': 256},
        ]
        self.benchmark_block_comparison(block_configs)

        # 生成报告
        self.generate_report()

    def generate_report(self):
        """生成测试报告"""
        print("\n" + "=" * 80)
        print("基准测试摘要")
        print("=" * 80)

        # 分类汇总
        qsa_results = [r for r in self.results if 'QSA' in r.test_name]
        qel_results = [r for r in self.results if 'QEL' in r.test_name]
        ffn_results = [r for r in self.results if 'FFN' in r.test_name]
        block_results = [r for r in self.results if 'Block' in r.test_name]

        def summarize(name, results):
            if not results:
                return
            avg_speedup = sum(r.speedup for r in results) / len(results)
            max_speedup = max(r.speedup for r in results)
            min_speedup = min(r.speedup for r in results)
            print(f"\n{name}:")
            print(f"  平均加速: {avg_speedup:.2f}×")
            print(f"  最大加速: {max_speedup:.2f}×")
            print(f"  最小加速: {min_speedup:.2f}×")

        summarize("QSA vs 注意力", qsa_results)
        summarize("QEL vs 残差", qel_results)
        summarize("FFN_Q vs FFN", ffn_results)
        summarize("完整 Block", block_results)

        # 保存 JSON 报告
        output_dir = Path('./benchmark_results')
        output_dir.mkdir(exist_ok=True)

        report = {
            'device': self.device,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'results': [
                {
                    'test_name': r.test_name,
                    'quantum_time_ms': r.quantum_time_ms,
                    'baseline_time_ms': r.baseline_time_ms,
                    'speedup': r.speedup,
                    'quantum_memory_mb': r.quantum_memory_mb,
                    'baseline_memory_mb': r.baseline_memory_mb,
                    'memory_ratio': r.memory_ratio,
                    'metrics': r.metrics
                }
                for r in self.results
            ],
            'summary': {
                'total_tests': len(self.results),
                'avg_speedup': sum(r.speedup for r in self.results) / len(self.results) if self.results else 0
            }
        }

        report_file = output_dir / f'benchmark_{int(time.time())}.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\n报告已保存: {report_file}")


def main():
    """主函数"""
    runner = BenchmarkRunner()
    runner.run_all_benchmarks()


if __name__ == '__main__':
    main()
