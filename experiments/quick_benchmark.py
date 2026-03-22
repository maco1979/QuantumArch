"""
QuantumArch 轻量实验基准 - 快速对比（单次前向+单步训练）

用于快速获取性能数据，无需长时间训练。
"""

import sys
import os
import time
import math
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

device = torch.device("cpu")

print("=" * 70)
print("  QuantumArch vs Standard Transformer 快速实验")
print("=" * 70)
print(f"  PyTorch: {torch.__version__}")
print(f"  Device: {device}")
print()


# ============================================================================
# 1. 环境信息 & 参数量对比
# ============================================================================
from quantum_core import QuantumArch

configs = [
    {"dim": 64, "layers": 2, "heads": 4, "name": "Small"},
    {"dim": 128, "layers": 4, "heads": 4, "name": "Medium"},
    {"dim": 128, "layers": 6, "heads": 8, "name": "Large"},
]

print("=" * 70)
print("  1. 模型参数量对比")
print("=" * 70)
print(f"  {'配置':<10} {'QuantumArch':<18} {'Transformer':<18} {'QA/ST 比值':<12}")
print("-" * 70)

param_results = []
for cfg in configs:
    dim, layers, heads, name = cfg["dim"], cfg["layers"], cfg["heads"], cfg["name"]
    vocab = 100
    ffn_dim = dim * 4

    qa = QuantumArch(
        vocab_size=vocab,
        dim=dim,
        num_layers=layers,
        num_heads=heads,
        ffn_dim=ffn_dim,
        max_seq_len=256,
        topk_ratio=0.15,
        collapse_enabled=True,
        dropout=0.0,
        qsa_mode="topk",
        output_dim=vocab,
    )

    class StdTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(vocab, dim)
            self.pos = nn.Embedding(256, dim)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=ffn_dim,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
            self.head = nn.Linear(dim, vocab)

        def forward(self, x):
            B, N = x.shape
            p = torch.arange(N, device=x.device).unsqueeze(0)
            h = self.emb(x) + self.pos(p)
            h = self.encoder(h)
            return self.head(h)

    st = StdTransformer()

    qa_params = sum(p.numel() for p in qa.parameters())
    st_params = sum(p.numel() for p in st.parameters())
    ratio = qa_params / st_params

    param_results.append(
        {"name": name, "qa_params": qa_params, "st_params": st_params, "ratio": ratio}
    )

    print(f"  {name:<10} {qa_params:<18,} {st_params:<18,} {ratio:<12.2f}×")

# ============================================================================
# 2. 前向传播延迟对比
# ============================================================================
print("\n" + "=" * 70)
print("  2. 前向传播延迟对比 (batch=4, seq=32)")
print("=" * 70)

seq_len = 32
batch = 4
vocab = 100

latency_results = []
for cfg in configs:
    dim, layers, heads, name = cfg["dim"], cfg["layers"], cfg["heads"], cfg["name"]
    ffn_dim = dim * 4

    qa = QuantumArch(
        vocab_size=vocab,
        dim=dim,
        num_layers=layers,
        num_heads=heads,
        ffn_dim=ffn_dim,
        max_seq_len=256,
        topk_ratio=0.15,
        collapse_enabled=True,
        dropout=0.0,
        qsa_mode="topk",
        output_dim=vocab,
    ).eval()

    class StdTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(vocab, dim)
            self.pos = nn.Embedding(256, dim)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=ffn_dim,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
            self.head = nn.Linear(dim, vocab)

        def forward(self, x):
            B, N = x.shape
            p = torch.arange(N, device=x.device).unsqueeze(0)
            h = self.emb(x) + self.pos(p)
            h = self.encoder(h)
            return self.head(h)

    st = StdTransformer().eval()

    tids = torch.randint(0, vocab, (batch, seq_len))

    # 预热
    with torch.no_grad():
        for _ in range(5):
            _ = qa({"token_ids": tids}, training=False)
            _ = st(tids)

    # 计时
    times_qa = []
    times_st = []
    for _ in range(20):
        with torch.no_grad():
            t0 = time.perf_counter()
            _ = qa({"token_ids": tids}, training=False)
            t1 = time.perf_counter()
            _ = st(tids)
            t2 = time.perf_counter()
        times_qa.append((t1 - t0) * 1000)
        times_st.append((t2 - t1) * 1000)

    avg_qa = np.mean(times_qa)
    avg_st = np.mean(times_st)
    speedup = avg_st / avg_qa

    latency_results.append({"name": name, "qa_ms": avg_qa, "st_ms": avg_st, "speedup": speedup})

    print(
        f"  {name:<10} QA: {avg_qa:.2f}ms  ST: {avg_st:.2f}ms  "
        f"比率: {1/speedup:.2f}× (ST/QA: {speedup:.2f}×)"
    )

# ============================================================================
# 3. 反向传播梯度质量
# ============================================================================
print("\n" + "=" * 70)
print("  3. 梯度质量对比 (单步训练)")
print("=" * 70)

dim = 64
layers = 2
heads = 4
vocab = 100
ffn_dim = dim * 4
batch = 8
seq_len = 16

qa = QuantumArch(
    vocab_size=vocab,
    dim=dim,
    num_layers=layers,
    num_heads=heads,
    ffn_dim=ffn_dim,
    max_seq_len=256,
    topk_ratio=0.15,
    collapse_enabled=True,
    dropout=0.0,
    qsa_mode="topk",
    output_dim=vocab,
)


class StdTransformer2(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(256, dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=ffn_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.head = nn.Linear(dim, vocab)

    def forward(self, x):
        B, N = x.shape
        p = torch.arange(N, device=x.device).unsqueeze(0)
        h = self.emb(x) + self.pos(p)
        h = self.encoder(h)
        return self.head(h)


st = StdTransformer2()

tids = torch.randint(0, vocab, (batch, seq_len))
labels = torch.randint(0, vocab, (batch, seq_len))

# QA 训练步
qa.train()
qa_out = qa({"token_ids": tids}, training=True)
qa_loss = F.cross_entropy(qa_out["output"].view(-1, vocab), labels.view(-1))
qa_loss.backward()

qa_grad_norms = []
for name, p in qa.named_parameters():
    if p.grad is not None:
        qa_grad_norms.append(p.grad.norm().item())

# ST 训练步
st.train()
st_out = st(tids)
st_loss = F.cross_entropy(st_out.view(-1, vocab), labels.view(-1))
st_loss.backward()

st_grad_norms = []
for name, p in st.named_parameters():
    if p.grad is not None:
        st_grad_norms.append(p.grad.norm().item())

print(f"  QuantumArch:")
print(f"    Loss: {qa_loss.item():.4f}")
print(
    f"    梯度范数: mean={np.mean(qa_grad_norms):.6f}, "
    f"median={np.median(qa_grad_norms):.6f}, "
    f"max={np.max(qa_grad_norms):.6f}, "
    f"梯度数={len(qa_grad_norms)}"
)
print(f"  StandardTransformer:")
print(f"    Loss: {st_loss.item():.4f}")
print(
    f"    梯度范数: mean={np.mean(st_grad_norms):.6f}, "
    f"median={np.median(st_grad_norms):.6f}, "
    f"max={np.max(st_grad_norms):.6f}, "
    f"梯度数={len(st_grad_norms)}"
)

# ============================================================================
# 4. 酉性约束验证
# ============================================================================
print("\n" + "=" * 70)
print("  4. QuantumArch 酉性约束验证")
print("=" * 70)

qa.eval()
report = qa.get_unitarity_report()
all_pass = all(v < 1e-2 for v in report.values())
for k, v in report.items():
    status = "PASS" if v < 1e-2 else "WARN" if v < 0.1 else "FAIL"
    print(f"  [{status}] {k}: {v:.2e}")
print(f"  结果: {'ALL PASS' if all_pass else 'NEEDS ATTENTION'}")

# ============================================================================
# 5. 5步训练损失趋势
# ============================================================================
print("\n" + "=" * 70)
print("  5. 5步训练损失趋势")
print("=" * 70)

from quantum_core import QGD

qa.train()
st.train()

qa_opt = QGD(qa.parameters(), mod_lr=5e-4, phase_lr=5e-4)
st_opt = torch.optim.AdamW(st.parameters(), lr=5e-4, weight_decay=0.01)

qa_losses = []
st_losses = []

for step in range(5):
    tids = torch.randint(0, vocab, (batch, seq_len))
    labels = torch.randint(0, vocab, (batch, seq_len))

    qa_opt.zero_grad()
    qa_out = qa({"token_ids": tids}, training=True)
    qa_loss = F.cross_entropy(qa_out["output"].view(-1, vocab), labels.view(-1))
    qa_loss.backward()
    qa_opt.step()
    qa_losses.append(qa_loss.item())

    st_opt.zero_grad()
    st_out = st(tids)
    st_loss = F.cross_entropy(st_out.view(-1, vocab), labels.view(-1))
    st_loss.backward()
    st_opt.step()
    st_losses.append(st_loss.item())

print(f"  Step | QA Loss  | ST Loss  | QA 趋势 | ST 趋势")
print(f"  -----|----------|----------|---------|---------")
for i in range(5):
    qa_trend = "↓" if i > 0 and qa_losses[i] < qa_losses[i - 1] else "↑" if i > 0 else " "
    st_trend = "↓" if i > 0 and st_losses[i] < st_losses[i - 1] else "↑" if i > 0 else " "
    print(
        f"  {i+1:4d} | {qa_losses[i]:.4f}   | {st_losses[i]:.4f}   | {qa_trend}        | {st_trend}"
    )

qa_decline = (qa_losses[0] - qa_losses[-1]) / qa_losses[0] * 100
st_decline = (st_losses[0] - st_losses[-1]) / st_losses[0] * 100
print(f"\n  QA 损失下降: {qa_decline:.1f}%  ({qa_losses[0]:.4f} → {qa_losses[-1]:.4f})")
print(f"  ST 损失下降: {st_decline:.1f}%  ({st_losses[0]:.4f} → {st_losses[-1]:.4f})")

# ============================================================================
# 6. 序列长度扩展性
# ============================================================================
print("\n" + "=" * 70)
print("  6. 序列长度扩展性 (dim=64, 2层, batch=4)")
print("=" * 70)

dim = 64
layers = 2
heads = 4
seq_lens = [16, 32, 64, 128]

qa_scale = QuantumArch(
    vocab_size=vocab,
    dim=dim,
    num_layers=layers,
    num_heads=heads,
    ffn_dim=ffn_dim,
    max_seq_len=256,
    topk_ratio=0.15,
    collapse_enabled=True,
    dropout=0.0,
    qsa_mode="topk",
    output_dim=vocab,
).eval()


class StdTransformer3(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(256, dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=ffn_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.head = nn.Linear(dim, vocab)

    def forward(self, x):
        B, N = x.shape
        p = torch.arange(N, device=x.device).unsqueeze(0)
        h = self.emb(x) + self.pos(p)
        h = self.encoder(h)
        return self.head(h)


st_scale = StdTransformer3().eval()

print(f"  {'seq_len':<10} {'QA (ms)':<12} {'ST (ms)':<12} {'QA/ST':<10}")
print(f"  {'-'*44}")

scaling_data = []
for sl in seq_lens:
    tids = torch.randint(0, vocab, (4, sl))

    with torch.no_grad():
        # 预热
        _ = qa_scale({"token_ids": tids}, training=False)
        _ = st_scale(tids)

        # 计时
        times_qa = []
        times_st = []
        for _ in range(10):
            t0 = time.perf_counter()
            _ = qa_scale({"token_ids": tids}, training=False)
            t1 = time.perf_counter()
            _ = st_scale(tids)
            t2 = time.perf_counter()
            times_qa.append((t1 - t0) * 1000)
            times_st.append((t2 - t1) * 1000)

    avg_qa = np.mean(times_qa)
    avg_st = np.mean(times_st)
    ratio = avg_qa / avg_st if avg_st > 0 else 0

    scaling_data.append({"seq_len": sl, "qa_ms": avg_qa, "st_ms": avg_st, "ratio": ratio})
    print(f"  {sl:<10} {avg_qa:<12.2f} {avg_st:<12.2f} {ratio:<10.2f}")

# ============================================================================
# 7. QCI 早退效率
# ============================================================================
print("\n" + "=" * 70)
print("  7. QCI 自适应早退效率")
print("=" * 70)

from quantum_core import QuantumCollapseInference

dim = 64
qci_low = QuantumCollapseInference(dim=dim, tau_low=0.3, tau_high=1.0, adaptive_tau=True)
qci_high = QuantumCollapseInference(dim=dim, tau_low=0.7, tau_high=2.0, adaptive_tau=True)

x = torch.randn(4, 32, dim, dtype=torch.complex64)

qci_low.eval()
qci_high.eval()
with torch.no_grad():
    _, m_low = qci_low(x, training=False)
    _, m_high = qci_high(x, training=False)

print(f"  低阈值 (τ_low=0.3):")
print(f"    early_exit_rate = {m_low['collapse_early_exit_rate']:.2%}")
print(f"    entropy = {m_low['collapse_entropy']:.4f}")
print(f"  高阈值 (τ_low=0.7):")
print(f"    early_exit_rate = {m_high['collapse_early_exit_rate']:.2%}")
print(f"    entropy = {m_high['collapse_entropy']:.4f}")

# ============================================================================
# 保存结果
# ============================================================================
print("\n" + "=" * 70)
print("  实验完成！保存结果...")
print("=" * 70)

all_results = {
    "timestamp": datetime.now().isoformat(),
    "environment": {
        "pytorch": torch.__version__,
        "device": str(device),
        "python": sys.version.split()[0],
    },
    "parameters": param_results,
    "latency": latency_results,
    "gradient_quality": {
        "qa_loss": qa_losses,
        "st_loss": st_losses,
        "qa_grad_mean": float(np.mean(qa_grad_norms)),
        "st_grad_mean": float(np.mean(st_grad_norms)),
    },
    "scaling": scaling_data,
    "qci_efficiency": {
        "low_threshold": {
            "early_exit_rate": m_low["collapse_early_exit_rate"],
            "entropy": m_low["collapse_entropy"],
        },
        "high_threshold": {
            "early_exit_rate": m_high["collapse_early_exit_rate"],
            "entropy": m_high["collapse_entropy"],
        },
    },
}

output_dir = Path(__file__).parent
json_path = output_dir / "experiment_results.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

print(f"  JSON 结果: {json_path}")
