"""
量子架构性能分析脚本
QuantumArch Performance Profiler

对量子架构各模块进行详细的性能画像：
- 算子级别耗时分析
- 内存峰值追踪
- 复数运算效率评估
- FLOPs 估算（量子门 vs 标准线性层）
"""

import time
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Callable, Tuple
from contextlib import contextmanager
from collections import defaultdict
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


# ─── 计时上下文管理器 ──────────────────────────────────────────────────────────

@contextmanager
def timer(name: str, results: Dict[str, float]):
    """精确计时上下文管理器"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000  # ms
        results[name] = elapsed


# ─── 内存追踪 ──────────────────────────────────────────────────────────────────

class MemoryTracker:
    """GPU/CPU 内存峰值追踪"""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.snapshots: List[Dict] = []
    
    def snapshot(self, label: str = "") -> Dict:
        info = {"label": label, "timestamp": time.time()}
        if self.device == "cuda" and torch.cuda.is_available():
            info["allocated_mb"] = torch.cuda.memory_allocated() / 1024**2
            info["reserved_mb"] = torch.cuda.memory_reserved() / 1024**2
            info["max_allocated_mb"] = torch.cuda.max_memory_allocated() / 1024**2
        else:
            try:
                import psutil
                process = psutil.Process()
                info["rss_mb"] = process.memory_info().rss / 1024**2
            except ImportError:
                info["rss_mb"] = -1
        
        self.snapshots.append(info)
        return info
    
    def reset(self):
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.snapshots.clear()


# ─── FLOPs 估算 ────────────────────────────────────────────────────────────────

def estimate_attention_flops(
    batch_size: int,
    seq_len: int,
    d_model: int,
    n_heads: int,
    is_complex: bool = False,
) -> Dict[str, int]:
    """
    估算注意力机制的 FLOPs
    
    标准注意力主要 FLOPs 来源：
    1. QKV 投影: 3 * 2 * B * L * d^2
    2. 注意力矩阵: 2 * B * H * L^2 * (d/H)
    3. 加权求和: 2 * B * H * L^2 * (d/H)
    4. 输出投影: 2 * B * L * d^2
    """
    d_k = d_model // n_heads
    
    # 对于复数运算，实际乘法数量增加约4倍（复数乘法 = 4实数乘法）
    complex_factor = 4 if is_complex else 1
    
    qkv_proj = 3 * 2 * batch_size * seq_len * d_model * d_model * complex_factor
    attn_scores = 2 * batch_size * n_heads * seq_len * seq_len * d_k * complex_factor
    attn_weighted = 2 * batch_size * n_heads * seq_len * seq_len * d_k * complex_factor
    out_proj = 2 * batch_size * seq_len * d_model * d_model * complex_factor
    
    total = qkv_proj + attn_scores + attn_weighted + out_proj
    
    return {
        "qkv_projection": qkv_proj,
        "attention_scores": attn_scores,
        "attention_weighted_sum": attn_weighted,
        "output_projection": out_proj,
        "total": total,
        "total_gflops": total / 1e9,
        "is_complex": is_complex,
    }


def estimate_layer_params(d_model: int, n_heads: int, ffn_ratio: int = 4) -> Dict[str, int]:
    """估算单个 Transformer/量子架构层的参数量"""
    d_ff = d_model * ffn_ratio
    
    # 注意力层：QKV + Out 投影（复数架构参数量翻倍）
    attn_params_real = 4 * d_model * d_model  # 标准
    attn_params_complex = 8 * d_model * d_model  # 复数（实+虚）
    
    # FFN 层
    ffn_params = 2 * d_model * d_ff
    
    # LayerNorm
    ln_params = 4 * d_model  # 2个 LN，每个有 scale + bias
    
    return {
        "attention_real": attn_params_real,
        "attention_complex": attn_params_complex,
        "ffn": ffn_params,
        "layer_norm": ln_params,
        "total_real": attn_params_real + ffn_params + ln_params,
        "total_complex": attn_params_complex + ffn_params + ln_params,
    }


# ─── 模块级别 Profiler ─────────────────────────────────────────────────────────

class ModuleProfiler:
    """
    模块级别性能分析器
    
    通过 PyTorch hook 机制自动记录每个模块的前向时间。
    """
    
    def __init__(self):
        self.timings: Dict[str, List[float]] = defaultdict(list)
        self._hooks: List = []
        self._start_times: Dict[str, float] = {}
    
    def attach(self, model: nn.Module) -> "ModuleProfiler":
        """挂载到模型的所有模块"""
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # 只挂叶节点
                h_pre = module.register_forward_pre_hook(self._make_pre_hook(name))
                h_post = module.register_forward_hook(self._make_post_hook(name))
                self._hooks.extend([h_pre, h_post])
        return self
    
    def _make_pre_hook(self, name: str):
        def hook(module, input):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self._start_times[name] = time.perf_counter()
        return hook
    
    def _make_post_hook(self, name: str):
        def hook(module, input, output):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            if name in self._start_times:
                elapsed = (time.perf_counter() - self._start_times[name]) * 1000
                self.timings[name].append(elapsed)
        return hook
    
    def detach(self):
        """解除所有 hook"""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
    
    def summary(self, top_k: int = 10) -> List[Dict]:
        """返回耗时 top-k 的模块列表"""
        stats = []
        for name, times in self.timings.items():
            arr = np.array(times)
            stats.append({
                "module": name,
                "mean_ms": round(float(arr.mean()), 4),
                "std_ms": round(float(arr.std()), 4),
                "total_ms": round(float(arr.sum()), 4),
                "calls": len(times),
            })
        
        return sorted(stats, key=lambda x: x["total_ms"], reverse=True)[:top_k]
    
    def print_summary(self, top_k: int = 10):
        """打印性能摘要"""
        stats = self.summary(top_k)
        print(f"\n{'='*65}")
        print(f"模块性能分析 (Top-{top_k})")
        print(f"{'='*65}")
        print(f"{'Module':<35} {'Mean(ms)':>8} {'Std(ms)':>8} {'Total(ms)':>10}")
        print(f"{'-'*35}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}")
        for s in stats:
            print(
                f"{s['module'][:35]:<35} "
                f"{s['mean_ms']:>8.4f} "
                f"{s['std_ms']:>8.4f} "
                f"{s['total_ms']:>10.2f}"
            )


# ─── 综合性能报告 ──────────────────────────────────────────────────────────────

def generate_performance_report(
    d_model: int = 256,
    n_layers: int = 6,
    n_heads: int = 8,
    seq_lengths: List[int] = [128, 256, 512],
    batch_size: int = 4,
    output_path: str = "benchmark/performance_report.json",
) -> Dict:
    """生成完整的性能分析报告"""
    
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "d_model": d_model,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "batch_size": batch_size,
        },
        "parameter_analysis": estimate_layer_params(d_model, n_heads),
        "flops_analysis": {},
        "scaling_analysis": [],
    }
    
    # FLOPs 分析
    for seq_len in seq_lengths:
        report["flops_analysis"][f"seq_{seq_len}"] = {
            "standard_attention": estimate_attention_flops(
                batch_size, seq_len, d_model, n_heads, is_complex=False
            ),
            "quantum_attention": estimate_attention_flops(
                batch_size, seq_len, d_model, n_heads, is_complex=True
            ),
        }
    
    # 扩展性分析：O(n) 复杂度验证
    baseline_flops = report["flops_analysis"][f"seq_{seq_lengths[0]}"]["standard_attention"]["total"]
    for seq_len in seq_lengths:
        flops = report["flops_analysis"][f"seq_{seq_len}"]["standard_attention"]["total"]
        ratio = flops / baseline_flops
        expected_ratio = (seq_len / seq_lengths[0]) ** 2  # O(n²)
        report["scaling_analysis"].append({
            "seq_len": seq_len,
            "actual_flops_ratio": round(ratio, 3),
            "expected_o_n2_ratio": round(expected_ratio, 3),
        })
    
    # 保存报告
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 性能报告已生成: {out_path}")
    
    # 打印摘要
    params = report["parameter_analysis"]
    print(f"\n📊 参数量分析 (d_model={d_model}, FFN_ratio=4)")
    print(f"  标准架构单层参数: {params['total_real']:,}")
    print(f"  量子架构单层参数: {params['total_complex']:,} (×{params['total_complex']/params['total_real']:.1f})")
    print(f"  标准架构总参数:   {params['total_real'] * n_layers:,}")
    print(f"  量子架构总参数:   {params['total_complex'] * n_layers:,}")
    
    return report


if __name__ == "__main__":
    report = generate_performance_report(
        d_model=256,
        n_layers=6,
        n_heads=8,
        seq_lengths=[128, 256, 512, 1024],
        batch_size=4,
    )
