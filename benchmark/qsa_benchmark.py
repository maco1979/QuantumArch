"""
QSA（量子叠加注意力）性能基准测试
Quantum Superposition Attention Performance Benchmark

对比 QSA 与标准 Scaled Dot-Product Attention 的：
- 前向传播速度
- 内存占用
- 数值精度（与标准注意力的输出差异）
- 不同序列长度下的扩展性
"""

import time
import torch
import torch.nn.functional as F
import json
from pathlib import Path
from typing import Dict, List, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def standard_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """标准缩放点积注意力（实数版本，用于对比基线）"""
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attn_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, v)


def benchmark_attention_forward(
    batch_size: int,
    seq_len: int,
    d_model: int,
    n_heads: int,
    n_warmup: int = 5,
    n_runs: int = 20,
    device: str = "cpu",
) -> Dict:
    """
    对比注意力机制前向传播性能
    
    Args:
        batch_size: 批次大小
        seq_len: 序列长度
        d_model: 模型维度
        n_heads: 注意力头数
        n_warmup: 预热次数（不计入统计）
        n_runs: 正式测试次数
        device: 运算设备
    
    Returns:
        包含各项指标的字典
    """
    d_k = d_model // n_heads
    dev = torch.device(device)
    
    # 生成实数 Q/K/V（对齐基线）
    q_real = torch.randn(batch_size, n_heads, seq_len, d_k, device=dev)
    k_real = torch.randn(batch_size, n_heads, seq_len, d_k, device=dev)
    v_real = torch.randn(batch_size, n_heads, seq_len, d_k, device=dev)
    
    # 生成复数 Q/K/V（用于量子注意力）
    q_cplx = torch.complex(q_real, torch.randn_like(q_real))
    k_cplx = torch.complex(k_real, torch.randn_like(k_real))
    v_cplx = torch.complex(v_real, torch.randn_like(v_real))
    
    # ─── 基线：标准注意力（实数）──────────────────────────
    for _ in range(n_warmup):
        _ = standard_attention(q_real, k_real, v_real)
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    t0 = time.perf_counter()
    for _ in range(n_runs):
        out_std = standard_attention(q_real, k_real, v_real)
        if device == "cuda":
            torch.cuda.synchronize()
    t_std = (time.perf_counter() - t0) / n_runs * 1000  # ms
    
    # ─── 量子注意力：复数点积 + Born法则归一化 ─────────────
    def quantum_attention_forward(q, k, v):
        # 复数内积：⟨q|k⟩ = Σ conj(q) * k
        scores = torch.matmul(q.conj(), k.transpose(-2, -1))
        # Born法则概率
        probs = scores.abs() ** 2
        probs = probs / (probs.sum(dim=-1, keepdim=True).clamp(min=1e-8))
        # 加权求和
        return torch.matmul(probs.to(v.dtype), v)
    
    for _ in range(n_warmup):
        _ = quantum_attention_forward(q_cplx, k_cplx, v_cplx)
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    t0 = time.perf_counter()
    for _ in range(n_runs):
        out_qsa = quantum_attention_forward(q_cplx, k_cplx, v_cplx)
        if device == "cuda":
            torch.cuda.synchronize()
    t_qsa = (time.perf_counter() - t0) / n_runs * 1000  # ms
    
    # ─── 内存统计 ─────────────────────────────────────────
    mem_std = q_real.element_size() * q_real.numel() * 3  # Q+K+V（字节）
    mem_qsa = q_cplx.element_size() * q_cplx.numel() * 3
    
    # ─── 数值比较（实部对比）──────────────────────────────
    # 用实数输入做QSA，与标准注意力对比
    q_c = torch.complex(q_real, torch.zeros_like(q_real))
    k_c = torch.complex(k_real, torch.zeros_like(k_real))
    v_c = torch.complex(v_real, torch.zeros_like(v_real))
    out_qsa_real = quantum_attention_forward(q_c, k_c, v_c).real
    
    # 归一化后对比
    out_std_norm = F.normalize(out_std, dim=-1)
    out_qsa_norm = F.normalize(out_qsa_real, dim=-1)
    cosine_sim = (out_std_norm * out_qsa_norm).sum(-1).mean().item()
    
    return {
        "config": {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "d_model": d_model,
            "n_heads": n_heads,
            "device": device,
        },
        "timing_ms": {
            "standard_attention": round(t_std, 4),
            "quantum_attention": round(t_qsa, 4),
            "speedup_ratio": round(t_std / t_qsa, 4),
        },
        "memory_bytes": {
            "standard_qkv": mem_std,
            "quantum_qkv": mem_qsa,
            "overhead_ratio": round(mem_qsa / mem_std, 2),
        },
        "accuracy": {
            "output_cosine_similarity": round(cosine_sim, 6),
        },
    }


def run_scalability_test(
    seq_lengths: List[int] = [64, 128, 256, 512, 1024],
    batch_size: int = 4,
    d_model: int = 256,
    n_heads: int = 8,
    device: str = "cpu",
) -> List[Dict]:
    """
    序列长度扩展性测试
    
    Args:
        seq_lengths: 测试的序列长度列表
        batch_size: 批次大小
        d_model: 模型维度
        n_heads: 注意力头数
        device: 运算设备
    
    Returns:
        各序列长度的基准结果列表
    """
    results = []
    print(f"\n{'='*60}")
    print(f"QSA vs 标准注意力 扩展性测试 (d_model={d_model}, heads={n_heads})")
    print(f"{'='*60}")
    print(f"{'seq_len':>8} | {'Standard(ms)':>12} | {'QSA(ms)':>10} | {'Speedup':>8} | {'CosSim':>8}")
    print(f"{'-'*8}-+-{'-'*12}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}")
    
    for seq_len in seq_lengths:
        try:
            result = benchmark_attention_forward(
                batch_size=batch_size,
                seq_len=seq_len,
                d_model=d_model,
                n_heads=n_heads,
                device=device,
            )
            results.append(result)
            
            t = result["timing_ms"]
            a = result["accuracy"]
            print(
                f"{seq_len:>8} | {t['standard_attention']:>12.4f} | "
                f"{t['quantum_attention']:>10.4f} | "
                f"{t['speedup_ratio']:>8.4f} | "
                f"{a['output_cosine_similarity']:>8.4f}"
            )
        except Exception as e:
            print(f"{seq_len:>8} | ERROR: {e}")
    
    return results


def save_benchmark_results(results: List[Dict], output_path: str = "benchmark/qsa_benchmark.json"):
    """保存基准测试结果到JSON文件"""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output, "w", encoding="utf-8") as f:
        json.dump({
            "benchmark": "QSA vs Standard Attention",
            "timestamp": time.strftime("%Y-%m-%d %Human:%M:%S"),
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n基准结果已保存至: {output}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"运行设备: {device}")
    
    # 扩展性测试
    results = run_scalability_test(
        seq_lengths=[64, 128, 256, 512],
        batch_size=4,
        d_model=256,
        n_heads=8,
        device=device,
    )
    
    # 保存结果
    save_benchmark_results(results)
