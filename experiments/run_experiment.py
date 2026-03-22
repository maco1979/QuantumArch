"""
QuantumArch 真实实验基准 - 对比标准 Transformer

实验设计：
1. 复制记忆任务 (Copy Memory) - 测试序列建模能力
2. 加法任务 (Addition) - 测试数值推理能力
3. 排序任务 (Sorting) - 测试序列理解能力

每个实验对比：
- QuantumArch（复数量子架构）
- Standard Transformer（同参数量实数架构）

评估指标：训练损失、验证准确率、参数量、FLOPs

运行: python experiments/run_experiment.py
"""

import sys
import os
import time
import math
import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# 标准 Transformer 基准模型
# ============================================================================


class StandardTransformer(nn.Module):
    """标准 Transformer 编码器（与 QuantumArch 同参数量对比）"""

    def __init__(
        self,
        dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        ffn_dim: Optional[int] = None,
        max_seq_len: int = 512,
        vocab_size: int = 100,
        output_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.output_dim = output_dim or vocab_size
        ffn_dim = ffn_dim or dim * 4

        # Token + 位置嵌入
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.dropout = nn.Dropout(dropout)

        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出头
        self.output_head = nn.Linear(dim, self.output_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        B, N = token_ids.shape
        positions = torch.arange(N, device=token_ids.device).unsqueeze(0)

        x = self.token_emb(token_ids) + self.pos_emb(positions)
        x = self.dropout(x)
        x = self.encoder(x)
        return self.output_head(x)


# ============================================================================
# 数据生成器
# ============================================================================


class CopyMemoryTask:
    """复制记忆任务：记忆序列中的标记并在指定位置回忆。

    序列格式: [BOS] [item_1] [item_2] ... [item_k] [DELIM] [pad...pad] [query_1] ... [query_k] [EOS]

    模型需要在看到 query 时输出对应的 item。
    """

    def __init__(self, vocab_size: int = 10, num_items: int = 6, total_len: int = 20):
        self.vocab_size = vocab_size
        self.num_items = num_items
        self.total_len = total_len
        # 特殊 token: 0=BOS, 1=EOS, 2=DELIM, 3=PAD, 4..4+vocab_size-1 = 数据 token
        self.bos = 0
        self.eos = 1
        self.delim = 2
        self.pad = 3
        self.data_start = 4
        self.total_tokens = 4 + vocab_size  # BOS+EOS+DELIM+PAD + data tokens

    def generate_batch(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """生成一批训练数据。"""
        items_per_sample = []
        sequences = []
        targets = []

        for _ in range(batch_size):
            # 随机选择 num_items 个数据 token
            items = torch.randint(
                self.data_start, self.data_start + self.vocab_size, (self.num_items,)
            )
            items_per_sample.append(items)

            # 构建序列: [BOS, item_1..item_k, DELIM, PAD...PAD, item_1..item_k, EOS]
            recall_start = self.num_items + 2  # BOS + items + DELIM
            padding_len = max(
                0, self.total_len - 2 - self.num_items * 2 - 1
            )  # -2 for BOS/EOS, -1 for DELIM
            seq = [self.bos]
            seq.extend(items.tolist())
            seq.append(self.delim)
            seq.extend([self.pad] * padding_len)
            seq.extend(items.tolist())
            seq.append(self.eos)

            # 目标: 每个位置预测下一个 token（teacher forcing）
            # 对于 recall 位置，目标是对应的 item
            target = seq[1:] + [self.eos]  # shift right

            sequences.append(seq)
            targets.append(target)

        # Pad to same length
        max_len = max(len(s) for s in sequences)
        padded_seqs = torch.full((batch_size, max_len), self.pad, dtype=torch.long, device=device)
        padded_targets = torch.full(
            (batch_size, max_len), self.pad, dtype=torch.long, device=device
        )

        for i, (seq, tgt) in enumerate(zip(sequences, targets)):
            padded_seqs[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
            padded_targets[i, : len(tgt)] = torch.tensor(tgt, dtype=torch.long, device=device)

        return {
            "input_ids": padded_seqs,
            "labels": padded_targets,
            "num_items": self.num_items,
        }

    def evaluate_accuracy(
        self, logits: torch.Tensor, labels: torch.Tensor, num_items: int
    ) -> Tuple[float, float]:
        """计算 recall 区域的准确率（核心指标）和整体准确率。"""
        predictions = logits.argmax(dim=-1)

        # 整体准确率（排除 PAD）
        mask = labels != self.pad
        if mask.sum() == 0:
            return 0.0, 0.0
        overall_acc = (predictions[mask] == labels[mask]).float().mean().item()

        # Recall 区域准确率
        B, N = labels.shape
        recall_start = num_items + 2  # BOS + items + DELIM
        recall_positions = torch.arange(
            recall_start, recall_start + num_items, device=labels.device
        )
        recall_positions = recall_positions[recall_positions < N]  # 防止越界
        if len(recall_positions) == 0:
            return overall_acc, 0.0

        recall_preds = predictions[:, recall_positions]
        recall_labels = labels[:, recall_positions]
        recall_acc = (recall_preds == recall_labels).float().mean().item()

        return overall_acc, recall_acc


class AdditionTask:
    """加法任务：给定两个数字的序列表示，预测它们的和。

    序列格式: [BOS] [d1_0] [d1_1] ... [d1_k] [PLUS] [d2_0] ... [d2_k] [DELIM] [r_0] [r_1] ... [PAD] [EOS]
    """

    def __init__(self, max_num: int = 1000, total_len: int = 20):
        self.max_num = max_num
        self.total_len = total_len
        # 特殊 token
        self.bos = 0
        self.eos = 1
        self.delim = 2
        self.pad = 3
        self.plus = 4
        self.digit_start = 5  # digits 0-9
        self.total_tokens = 15  # enough for digits + special tokens

    def _num_to_digits(self, num: int) -> List[int]:
        """数字转 digit token 列表（高位在前）。"""
        if num == 0:
            return [self.digit_start]
        digits = []
        while num > 0:
            digits.append(self.digit_start + (num % 10))
            num //= 10
        return digits[::-1]

    def generate_batch(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        sequences = []
        targets = []

        for _ in range(batch_size):
            a = torch.randint(0, self.max_num, (1,)).item()
            b = torch.randint(0, self.max_num, (1,)).item()
            result = a + b

            digits_a = self._num_to_digits(a)
            digits_b = self._num_to_digits(b)
            digits_r = self._num_to_digits(result)

            # 构建序列
            seq = (
                [self.bos]
                + digits_a
                + [self.plus]
                + digits_b
                + [self.delim]
                + digits_r
                + [self.eos]
            )
            target = seq[1:] + [self.eos]

            # Pad
            while len(seq) < self.total_len:
                seq.append(self.pad)
                target.append(self.pad)

            sequences.append(seq[: self.total_len])
            targets.append(target[: self.total_len])

        input_ids = torch.tensor(sequences, dtype=torch.long, device=device)
        labels = torch.tensor(targets, dtype=torch.long, device=device)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "results": None,  # 结果编码在序列中
        }

    def evaluate_accuracy(
        self, logits: torch.Tensor, labels: torch.Tensor, **kwargs
    ) -> Tuple[float, float]:
        predictions = logits.argmax(dim=-1)
        mask = labels != self.pad
        if mask.sum() == 0:
            return 0.0, 0.0
        overall_acc = (predictions[mask] == labels[mask]).float().mean().item()
        # 加法任务中 overall_acc 就是核心指标
        return overall_acc, overall_acc


class SymbolSortingTask:
    """符号排序任务：对序列中的符号按值排序。

    序列格式: [BOS] [s1] [s2] ... [sk] [DELIM] [sorted_s1] ... [sorted_sk] [EOS]
    """

    def __init__(self, num_symbols: int = 8, symbol_range: int = 20, total_len: int = 24):
        self.num_symbols = num_symbols
        self.symbol_range = symbol_range
        self.total_len = total_len
        self.bos = 0
        self.eos = 1
        self.delim = 2
        self.pad = 3
        self.sym_start = 4
        self.total_tokens = 4 + symbol_range

    def generate_batch(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        sequences = []
        targets = []

        for _ in range(batch_size):
            symbols = torch.randint(
                self.sym_start, self.sym_start + self.symbol_range, (self.num_symbols,)
            )
            sorted_symbols = symbols.sort()[0]

            seq = (
                [self.bos] + symbols.tolist() + [self.delim] + sorted_symbols.tolist() + [self.eos]
            )
            target = seq[1:] + [self.eos]

            while len(seq) < self.total_len:
                seq.append(self.pad)
                target.append(self.pad)

            sequences.append(seq[: self.total_len])
            targets.append(target[: self.total_len])

        input_ids = torch.tensor(sequences, dtype=torch.long, device=device)
        labels = torch.tensor(targets, dtype=torch.long, device=device)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "num_symbols": self.num_symbols,
        }

    def evaluate_accuracy(
        self, logits: torch.Tensor, labels: torch.Tensor, num_symbols: int = 0, **kwargs
    ) -> Tuple[float, float]:
        predictions = logits.argmax(dim=-1)
        mask = labels != self.pad
        if mask.sum() == 0:
            return 0.0, 0.0
        overall_acc = (predictions[mask] == labels[mask]).float().mean().item()

        # Sort 区域准确率
        B, N = labels.shape
        sort_start = num_symbols + 2  # BOS + symbols + DELIM
        sort_positions = torch.arange(sort_start, sort_start + num_symbols, device=labels.device)
        sort_positions = sort_positions[sort_positions < N]
        if len(sort_positions) == 0:
            return overall_acc, 0.0
        sort_preds = predictions[:, sort_positions]
        sort_labels = labels[:, sort_positions]
        sort_acc = (sort_preds == sort_labels).float().mean().item()

        return overall_acc, sort_acc


# ============================================================================
# 实验运行器
# ============================================================================


@dataclass
class ExperimentConfig:
    """实验配置"""

    name: str
    task: str  # 'copy', 'addition', 'sorting'
    dim: int = 128
    num_layers: int = 4
    num_heads: int = 4
    ffn_dim: Optional[int] = None
    batch_size: int = 32
    num_epochs: int = 30
    learning_rate: float = 3e-4
    max_seq_len: int = 64
    eval_interval: int = 5
    seed: int = 42


@dataclass
class ExperimentResult:
    """实验结果"""

    model_name: str
    task_name: str
    total_params: int
    final_train_loss: float
    final_eval_acc: float
    final_task_acc: float
    convergence_epoch: int  # 达到 90% task accuracy 的 epoch
    time_seconds: float
    history: List[Dict[str, float]] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_single_experiment(
    model: nn.Module,
    model_name: str,
    task,
    task_name: str,
    config: ExperimentConfig,
    device: torch.device,
) -> ExperimentResult:
    """运行单个模型在单个任务上的训练实验。"""

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    num_classes = task.total_tokens if hasattr(task, "total_tokens") else config.max_seq_len

    model.to(device)
    model.train()

    history = []
    convergence_epoch = -1
    start_time = time.time()

    for epoch in range(config.num_epochs):
        model.train()
        epoch_losses = []
        epoch_overall_accs = []
        epoch_task_accs = []
        num_batches = 0

        # 训练
        for _ in range(30):  # 30 batches per epoch
            batch = task.generate_batch(config.batch_size, device)

            if model_name == "QuantumArch":
                # QuantumArch 使用字典输入
                out = model({"token_ids": batch["input_ids"]}, training=True)
                logits = out["output"]
            else:
                logits = model(batch["input_ids"])

            # 计算损失（排除 PAD 位置）
            labels = batch["labels"]
            loss_mask = labels != task.pad
            if loss_mask.sum() == 0:
                continue

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=task.pad,
            )

            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_losses.append(loss.item())
            num_batches += 1

            # 评估
            if num_batches % 25 == 0:
                model.eval()
                with torch.no_grad():
                    eval_batch = task.generate_batch(config.batch_size, device)
                    if model_name == "QuantumArch":
                        eval_out = model({"token_ids": eval_batch["input_ids"]}, training=False)
                        eval_logits = eval_out["output"]
                    else:
                        eval_logits = model(eval_batch["input_ids"])

                    eval_labels = eval_batch["labels"]
                    eval_kwargs = {}
                    if "num_items" in eval_batch:
                        eval_kwargs["num_items"] = eval_batch["num_items"]
                    if "num_symbols" in eval_batch:
                        eval_kwargs["num_symbols"] = eval_batch["num_symbols"]

                    overall_acc, task_acc = task.evaluate_accuracy(
                        eval_logits, eval_labels, **eval_kwargs
                    )
                    epoch_overall_accs.append(overall_acc)
                    epoch_task_accs.append(task_acc)
                model.train()

        avg_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        avg_overall_acc = sum(epoch_overall_accs) / max(len(epoch_overall_accs), 1)
        avg_task_acc = sum(epoch_task_accs) / max(len(epoch_task_accs), 1)

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "eval_overall_acc": avg_overall_acc,
                "eval_task_acc": avg_task_acc,
            }
        )

        # 检查收敛
        if convergence_epoch < 0 and avg_task_acc >= 0.90:
            convergence_epoch = epoch + 1

        if (epoch + 1) % config.eval_interval == 0 or epoch == 0 or epoch == config.num_epochs - 1:
            print(
                f"    Epoch {epoch+1:3d}/{config.num_epochs}: "
                f"loss={avg_loss:.4f}, overall_acc={avg_overall_acc:.4f}, "
                f"task_acc={avg_task_acc:.4f}"
            )

    elapsed = time.time() - start_time
    final_loss = history[-1]["train_loss"] if history else float("inf")
    final_overall = history[-1]["eval_overall_acc"] if history else 0.0
    final_task = history[-1]["eval_task_acc"] if history else 0.0

    return ExperimentResult(
        model_name=model_name,
        task_name=task_name,
        total_params=count_parameters(model),
        final_train_loss=final_loss,
        final_eval_acc=final_overall,
        final_task_acc=final_task,
        convergence_epoch=convergence_epoch,
        time_seconds=elapsed,
        history=history,
        config=asdict(config),
    )


def run_all_experiments(device: torch.device) -> List[ExperimentResult]:
    """运行所有实验。"""
    results = []

    # ── 实验配置 ──
    experiments = [
        ExperimentConfig(
            name="copy_memory",
            task="copy",
            dim=64,
            num_layers=2,
            num_heads=4,
            batch_size=16,
            num_epochs=20,
            learning_rate=5e-4,
            max_seq_len=32,
        ),
        ExperimentConfig(
            name="addition",
            task="addition",
            dim=64,
            num_layers=2,
            num_heads=4,
            batch_size=16,
            num_epochs=20,
            learning_rate=5e-4,
            max_seq_len=32,
        ),
        ExperimentConfig(
            name="sorting",
            task="sorting",
            dim=64,
            num_layers=2,
            num_heads=4,
            batch_size=16,
            num_epochs=20,
            learning_rate=5e-4,
            max_seq_len=32,
        ),
    ]

    for config in experiments:
        print(f"\n{'='*70}")
        print(f"  Task: {config.name}")
        print(
            f"  Config: dim={config.dim}, layers={config.num_layers}, "
            f"heads={config.num_heads}, epochs={config.num_epochs}"
        )
        print(f"{'='*70}")

        # 创建任务
        if config.task == "copy":
            task = CopyMemoryTask(vocab_size=8, num_items=6, total_len=24)
        elif config.task == "addition":
            task = AdditionTask(max_num=500, total_len=20)
        elif config.task == "sorting":
            task = SymbolSortingTask(num_symbols=6, symbol_range=16, total_len=22)
        else:
            raise ValueError(f"Unknown task: {config.task}")

        total_tokens = task.total_tokens
        print(f"  Total tokens: {total_tokens}")

        # ── QuantumArch ──
        print(f"\n  --- QuantumArch ---")
        from quantum_core import QuantumArch, QGD

        qa_model = QuantumArch(
            vocab_size=total_tokens,
            dim=config.dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            max_seq_len=config.max_seq_len,
            topk_ratio=0.15,
            collapse_enabled=True,
            tau_low=0.5,
            tau_high=1.5,
            dropout=0.1,
            qsa_mode="topk",
            output_dim=total_tokens,
        )

        qa_result = run_single_experiment(
            qa_model, "QuantumArch", task, config.name, config, device
        )
        results.append(qa_result)

        # ── Standard Transformer ──
        print(f"\n  --- Standard Transformer ---")
        st_model = StandardTransformer(
            dim=config.dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            max_seq_len=config.max_seq_len,
            vocab_size=total_tokens,
            output_dim=total_tokens,
            dropout=0.1,
        )

        st_result = run_single_experiment(
            st_model, "StandardTransformer", task, config.name, config, device
        )
        results.append(st_result)

    return results


def generate_report(results: List[ExperimentResult]) -> str:
    """生成实验报告。"""
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("  QuantumArch vs Standard Transformer - 实验报告")
    report_lines.append(f"  日期: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report_lines.append("=" * 70)

    # 按任务分组
    tasks = {}
    for r in results:
        if r.task_name not in tasks:
            tasks[r.task_name] = {}
        tasks[r.task_name][r.model_name] = r

    report_lines.append("\n" + "-" * 70)
    report_lines.append(
        f"  {'任务':<20} {'模型':<22} {'参数量':<12} {'最终损失':<12} "
        f"{'Task Acc':<12} {'收敛Epoch':<12} {'时间(s)':<10}"
    )
    report_lines.append("-" * 70)

    summary_data = []

    for task_name in ["copy_memory", "addition", "sorting"]:
        if task_name not in tasks:
            continue
        models = tasks[task_name]

        qa = models.get("QuantumArch")
        st = models.get("StandardTransformer")

        if qa:
            report_lines.append(
                f"  {task_name:<20} {'QuantumArch':<22} {qa.total_params:<12,} "
                f"{qa.final_train_loss:<12.4f} {qa.final_task_acc:<12.4f} "
                f"{qa.convergence_epoch if qa.convergence_epoch > 0 else 'N/A':<12} "
                f"{qa.time_seconds:<10.1f}"
            )
            summary_data.append({"task": task_name, "model": "QuantumArch", "result": qa})

        if st:
            report_lines.append(
                f"  {task_name:<20} {'StandardTransformer':<22} {st.total_params:<12,} "
                f"{st.final_train_loss:<12.4f} {st.final_task_acc:<12.4f} "
                f"{st.convergence_epoch if st.convergence_epoch > 0 else 'N/A':<12} "
                f"{st.time_seconds:<10.1f}"
            )
            summary_data.append({"task": task_name, "model": "StandardTransformer", "result": st})

        # 对比
        if qa and st:
            report_lines.append("  " + "-" * 100)
            param_ratio = qa.total_params / st.total_params if st.total_params > 0 else 0
            acc_diff = qa.final_task_acc - st.final_task_acc
            report_lines.append(
                f"  {'':20} {'对比':<22} 参数比={param_ratio:.2f}×  "
                f"Acc差={acc_diff:+.4f}  "
                f"{'QA优' if acc_diff > 0 else 'ST优' if acc_diff < 0 else '持平'}"
            )
            report_lines.append("  " + "-" * 100)

    # 汇总
    report_lines.append("\n" + "=" * 70)
    report_lines.append("  关键发现")
    report_lines.append("=" * 70)

    qa_wins = 0
    st_wins = 0
    ties = 0
    for task_name in tasks:
        models = tasks[task_name]
        qa = models.get("QuantumArch")
        st = models.get("StandardTransformer")
        if qa and st:
            diff = qa.final_task_acc - st.final_task_acc
            if diff > 0.01:
                qa_wins += 1
            elif diff < -0.01:
                st_wins += 1
            else:
                ties += 1

    report_lines.append(f"  QuantumArch 胜出: {qa_wins} 个任务")
    report_lines.append(f"  StandardTransformer 胜出: {st_wins} 个任务")
    report_lines.append(f"  持平: {ties} 个任务")

    # 平均指标
    qa_accs = [r.final_task_acc for r in results if r.model_name == "QuantumArch"]
    st_accs = [r.final_task_acc for r in results if r.model_name == "StandardTransformer"]
    if qa_accs:
        report_lines.append(f"\n  QuantumArch 平均 Task Accuracy: {np.mean(qa_accs):.4f}")
    if st_accs:
        report_lines.append(f"  StandardTransformer 平均 Task Accuracy: {np.mean(st_accs):.4f}")

    report_lines.append("\n" + "=" * 70)
    report_lines.append("  注: 实验在 CPU 上运行，使用合成任务评估基础序列建模能力。")
    report_lines.append("  QuantumArch 使用复数表示和量子启发机制，参数量为 Transformer 的 ~2×")
    report_lines.append("  (复数参数 = 实部 + 虚部，每个参数存储 2 个浮点数)。")
    report_lines.append("=" * 70)

    return "\n".join(report_lines)


def save_results_json(results: List[ExperimentResult], output_path: str):
    """保存结果为 JSON。"""
    data = {
        "timestamp": datetime.now().isoformat(),
        "results": [asdict(r) for r in results],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def main():
    print("=" * 70)
    print("  QuantumArch 实验基准 - 对比标准 Transformer")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  设备: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  PyTorch: {torch.__version__}")
    print()

    # 运行实验
    results = run_all_experiments(device)

    # 生成报告
    report = generate_report(results)
    print("\n" + report)

    # 保存结果
    output_dir = Path(__file__).parent.parent / "experiments"
    output_dir.mkdir(exist_ok=True)

    report_path = output_dir / "experiment_results.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    json_path = output_dir / "experiment_results.json"
    save_results_json(results, str(json_path))

    print(f"\n  报告已保存: {report_path}")
    print(f"  JSON 已保存: {json_path}")

    return results


if __name__ == "__main__":
    main()
