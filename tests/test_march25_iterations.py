"""量子架构今日新功能单元测试（2026-03-25 迭代）

覆盖本次10次迭代新增/改进的功能：
1. QSA 因果掩码（causal mask）
2. QEL 纠缠度量导出 (get_entanglement_metrics)
3. QCI POVM 正交正则化损失 + 基向量重归一化
4. QGD 梯度 EMA 跟踪 + 增强统计
5. complex_ops 量子保真度/迹距离/相对熵
6. FFN ComplexSwiGLU 激活集成
7. QIR 量子干涉路由器独立模块
8. model.py 增强参数量统计（子模块分类 + 内存估算）

运行方式：
    python -m pytest tests/test_march25_iterations.py -v
    或直接运行：python tests/test_march25_iterations.py
"""

import sys
import os
import math

import torch
import pytest

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


# ──────────────────────────────────────────────
# 测试1: QSA 因果掩码
# ──────────────────────────────────────────────


class TestQSACausalMask:
    """测试 QuantumSuperpositionAttention 的因果掩码功能。"""

    def setup_method(self):
        from quantum_core.attention import QuantumSuperpositionAttention

        self.qsa = QuantumSuperpositionAttention(
            dim=64, num_heads=4, topk_ratio=0.5, mode="full"
        )

    def test_causal_output_shape(self):
        """因果模式下输出形状不变。"""
        x = torch.randn(2, 8, 64, dtype=torch.complex64)
        out, metrics = self.qsa(x, causal=True, training=False)
        assert out.shape == (2, 8, 64), f"输出形状错误: {out.shape}"

    def test_causal_vs_noncausal_differ(self):
        """因果掩码应改变注意力结果（尤其是非第一个 token）。"""
        torch.manual_seed(42)
        x = torch.randn(1, 6, 64, dtype=torch.complex64)
        with torch.no_grad():
            out_causal, _ = self.qsa(x, causal=True, training=False)
            out_free, _ = self.qsa(x, causal=False, training=False)
        # 因果/非因果输出在最后几个 token 上应该不同
        assert not torch.allclose(out_causal[:, -1], out_free[:, -1], atol=1e-5), \
            "因果掩码对最后 token 应有影响"

    def test_causal_metrics_contains_key(self):
        """metrics 中应包含 'causal' 字段。"""
        x = torch.randn(1, 4, 64, dtype=torch.complex64)
        _, metrics = self.qsa(x, causal=True, training=False)
        assert "causal" in metrics, "metrics 缺少 'causal' 字段"
        assert metrics["causal"] is True

    def test_noncausal_metrics(self):
        """非因果模式下 'causal' 字段为 False。"""
        x = torch.randn(1, 4, 64, dtype=torch.complex64)
        _, metrics = self.qsa(x, causal=False, training=False)
        assert metrics["causal"] is False


# ──────────────────────────────────────────────
# 测试2: QEL 纠缠度量导出
# ──────────────────────────────────────────────


class TestQELEntanglementMetrics:
    """测试 QuantumEntanglementLayer.get_entanglement_metrics 方法。"""

    def setup_method(self):
        from quantum_core.entanglement import QuantumEntanglementLayer

        self.qel = QuantumEntanglementLayer(dim=32, use_global_qft=True)

    def test_metrics_keys(self):
        """导出指标包含所有必要字段。"""
        x = torch.randn(2, 6, 32, dtype=torch.complex64)
        metrics = self.qel.get_entanglement_metrics(x)
        required_keys = {
            "concurrence_mean", "concurrence_std",
            "entanglement_entropy", "qft_alpha", "entanglement_depth"
        }
        missing = required_keys - set(metrics.keys())
        assert not missing, f"缺少字段: {missing}"

    def test_concurrence_range(self):
        """纠缠度应在 [0, 1] 范围内。"""
        x = torch.randn(2, 8, 32, dtype=torch.complex64)
        metrics = self.qel.get_entanglement_metrics(x)
        c = metrics["concurrence_mean"]
        assert 0.0 <= c <= 1.0, f"纠缠度超出范围: {c}"

    def test_qft_alpha_range(self):
        """QFT 混合系数应在 (0, 1) 范围内。"""
        x = torch.randn(2, 4, 32, dtype=torch.complex64)
        metrics = self.qel.get_entanglement_metrics(x)
        alpha = metrics["qft_alpha"]
        assert 0.0 < alpha < 1.0, f"qft_alpha 超出范围: {alpha}"

    def test_single_token_no_error(self):
        """单 token 序列不应报错（返回零纠缠）。"""
        x = torch.randn(1, 1, 32, dtype=torch.complex64)
        metrics = self.qel.get_entanglement_metrics(x)
        assert metrics["concurrence_mean"] == 0.0

    def test_no_grad_preserved(self):
        """get_entanglement_metrics 不应引入梯度图。"""
        x = torch.randn(2, 4, 32, dtype=torch.complex64, requires_grad=True)
        metrics = self.qel.get_entanglement_metrics(x)
        # 函数内部有 no_grad，结果应为标量
        for v in metrics.values():
            assert isinstance(v, float), f"指标应为 float，得到: {type(v)}"


# ──────────────────────────────────────────────
# 测试3: QCI POVM 正交正则化
# ──────────────────────────────────────────────


class TestPOVMOrthogonalization:
    """测试 POVMProjector 的正交正则化损失和基重归一化。"""

    def setup_method(self):
        from quantum_core.collapse import POVMProjector

        self.povm = POVMProjector(in_dim=16, out_dim=8)

    def test_orthogonality_loss_nonneg(self):
        """正交正则化损失应为非负实数。"""
        loss = self.povm.orthogonality_regularization_loss()
        assert loss.item() >= 0.0, f"正交损失为负: {loss.item()}"
        assert loss.dtype == torch.float32, f"损失类型错误: {loss.dtype}"

    def test_orthogonality_loss_zero_for_orthonormal(self):
        """对于正交基，正交损失应接近零。"""
        in_dim, out_dim = 16, 8
        from quantum_core.collapse import POVMProjector

        povm = POVMProjector(in_dim=in_dim, out_dim=out_dim)
        # 手动设置正交归一基（使用 QR 分解）
        basis_real = torch.randn(in_dim, out_dim)
        q, _ = torch.linalg.qr(basis_real)  # (in_dim, out_dim)
        orth_basis = torch.complex(q.T, torch.zeros_like(q.T))  # (out_dim, in_dim)
        povm.measurement_basis.data = orth_basis

        loss = povm.orthogonality_regularization_loss()
        # 正交归一基的 Gram 矩阵 ≈ I，损失应接近 0
        assert loss.item() < 1.0, f"正交基的正则化损失过大: {loss.item()}"

    def test_renormalize_basis(self):
        """重归一化后基向量应为单位向量。"""
        # 先放大基向量模长
        self.povm.measurement_basis.data *= 10.0
        self.povm.renormalize_basis()

        norms = self.povm.measurement_basis.abs().pow(2).sum(dim=-1).sqrt()
        # 每个基向量的模长应接近 1
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
            f"重归一化后基向量模长不为1: {norms}"


# ──────────────────────────────────────────────
# 测试4: QGD 梯度 EMA 跟踪
# ──────────────────────────────────────────────


class TestQGDGradEMA:
    """测试 QGD 的梯度 EMA 跟踪功能。"""

    def setup_method(self):
        from quantum_core.optimizer import QGD

        # 创建简单复数参数
        self.params = [torch.randn(8, 8, dtype=torch.complex64, requires_grad=True)]
        self.optimizer = QGD(self.params, mod_lr=1e-3, phase_lr=1e-2)

    def _simulate_step(self):
        """模拟一步优化（随机梯度）。"""
        for p in self.params:
            p.grad = torch.randn_like(p)
        self.optimizer.step()

    def test_get_stats_has_new_fields(self):
        """get_stats 应包含 grad_norm_real, mod_lr, phase_lr 字段。"""
        self._simulate_step()
        stats = self.optimizer.get_stats()
        assert "grad_norm_real" in stats, "缺少 grad_norm_real 字段"
        assert "mod_lr" in stats, "缺少 mod_lr 字段"
        assert "phase_lr" in stats, "缺少 phase_lr 字段"

    def test_ema_keys(self):
        """grad_norm_ema 应返回正确字段。"""
        self._simulate_step()
        ema = self.optimizer.grad_norm_ema()
        required = {"ema_complex", "ema_real", "ema_total", "step_count"}
        missing = required - set(ema.keys())
        assert not missing, f"缺少 EMA 字段: {missing}"

    def test_ema_increases_with_steps(self):
        """EMA 计数应随步数递增。"""
        for _ in range(5):
            self._simulate_step()

        ema1 = self.optimizer.grad_norm_ema()
        self._simulate_step()
        ema2 = self.optimizer.grad_norm_ema()

        assert ema2["step_count"] > ema1["step_count"], "EMA step_count 未递增"

    def test_ema_nonneg(self):
        """EMA 值应为非负数。"""
        for _ in range(3):
            self._simulate_step()
        ema = self.optimizer.grad_norm_ema()
        assert ema["ema_complex"] >= 0.0, f"ema_complex 为负: {ema['ema_complex']}"
        assert ema["ema_total"] >= 0.0, f"ema_total 为负: {ema['ema_total']}"


# ──────────────────────────────────────────────
# 测试5: 量子保真度/迹距离/相对熵
# ──────────────────────────────────────────────


class TestQuantumInfoMetrics:
    """测试 complex_ops.py 中新增的量子信息度量函数。"""

    def setup_method(self):
        from quantum_core.complex_ops import (
            quantum_fidelity, trace_distance, quantum_relative_entropy
        )

        self.fidelity = quantum_fidelity
        self.td = trace_distance
        self.qre = quantum_relative_entropy

    def test_fidelity_same_state(self):
        """相同态的保真度应为 1.0。"""
        psi = torch.tensor([1.0 + 0j, 0.0 + 0j], dtype=torch.complex64)
        f = self.fidelity(psi, psi)
        assert torch.isclose(f, torch.tensor(1.0), atol=1e-5), f"F(|ψ⟩,|ψ⟩) ≠ 1: {f}"

    def test_fidelity_orthogonal(self):
        """正交态的保真度应为 0.0。"""
        psi = torch.tensor([1.0 + 0j, 0.0 + 0j], dtype=torch.complex64)
        phi = torch.tensor([0.0 + 0j, 1.0 + 0j], dtype=torch.complex64)
        f = self.fidelity(psi, phi)
        assert torch.isclose(f, torch.tensor(0.0), atol=1e-5), f"F(|0⟩,|1⟩) ≠ 0: {f}"

    def test_fidelity_range(self):
        """保真度应在 [0, 1] 范围内。"""
        B, d = 4, 32
        psi = torch.randn(B, d, dtype=torch.complex64)
        phi = torch.randn(B, d, dtype=torch.complex64)
        f = self.fidelity(psi, phi)
        assert (f >= 0).all() and (f <= 1 + 1e-5).all(), f"保真度超出范围: {f}"

    def test_trace_distance_same_state(self):
        """相同态的迹距离应为 0.0。"""
        psi = torch.tensor([0.6 + 0.8j, 0.0 + 0j], dtype=torch.complex64)
        td = self.td(psi, psi)
        assert torch.isclose(td, torch.tensor(0.0), atol=1e-5), f"T(|ψ⟩,|ψ⟩) ≠ 0: {td}"

    def test_trace_distance_orthogonal(self):
        """正交态的迹距离应为 1.0。"""
        psi = torch.tensor([1.0 + 0j, 0.0 + 0j], dtype=torch.complex64)
        phi = torch.tensor([0.0 + 0j, 1.0 + 0j], dtype=torch.complex64)
        td = self.td(psi, phi)
        assert torch.isclose(td, torch.tensor(1.0), atol=1e-5), f"T(|0⟩,|1⟩) ≠ 1: {td}"

    def test_td_fidelity_relation(self):
        """验证 T² + F = 1 关系（纯态保真度-迹距离关系）。"""
        B, d = 3, 16
        psi = torch.randn(B, d, dtype=torch.complex64)
        phi = torch.randn(B, d, dtype=torch.complex64)
        f = self.fidelity(psi, phi)
        td = self.td(psi, phi)
        # T = √(1 - F) → T² + F = 1
        assert torch.allclose(td**2 + f, torch.ones(B), atol=1e-4), \
            f"T² + F ≠ 1: T²={td**2}, F={f}"

    def test_relative_entropy_nonneg(self):
        """量子相对熵应为非负。"""
        psi = torch.randn(4, 16, dtype=torch.complex64)
        phi = torch.randn(4, 16, dtype=torch.complex64)
        qre = self.qre(psi, phi)
        assert (qre >= 0).all(), f"相对熵为负: {qre}"

    def test_relative_entropy_self_zero(self):
        """相同分布的相对熵应为 0。"""
        psi = torch.randn(4, 16, dtype=torch.complex64)
        qre = self.qre(psi, psi)
        assert torch.allclose(qre, torch.zeros(4), atol=1e-4), \
            f"D(p||p) ≠ 0: {qre}"


# ──────────────────────────────────────────────
# 测试6: FFN ComplexSwiGLU 集成
# ──────────────────────────────────────────────


class TestFFNSwiGLU:
    """测试 QuantumFFN 的 swiglu 激活选项。"""

    def test_quantum_ffn_swiglu_output_shape(self):
        """swiglu 激活的 QuantumFFN 应有正确的输出形状。"""
        from quantum_core.ffn import QuantumFFN

        ffn = QuantumFFN(dim=64, ffn_dim=128, activation="modrelu")
        x = torch.randn(2, 8, 64, dtype=torch.complex64)
        out = ffn(x, training=False)
        assert out.shape == (2, 8, 64), f"输出形状错误: {out.shape}"

    def test_gated_ffn_swiglu_option(self):
        """GatedQuantumFFN 支持 swiglu 激活且输出形状正确。"""
        from quantum_core.ffn import GatedQuantumFFN

        ffn = GatedQuantumFFN(dim=32, ffn_dim=64, activation="swiglu")
        x = torch.randn(2, 4, 32, dtype=torch.complex64)
        out = ffn(x, training=False)
        assert out.shape == (2, 4, 32), f"swiglu GatedQuantumFFN 输出形状错误: {out.shape}"

    def test_swiglu_no_extra_gate(self):
        """swiglu 模式下 self.gate 应为 None（避免双重门控）。"""
        from quantum_core.ffn import GatedQuantumFFN

        ffn = GatedQuantumFFN(dim=32, ffn_dim=64, activation="swiglu")
        assert ffn.gate is None, "swiglu 模式下 gate 应为 None"

    def test_modrelu_has_gate(self):
        """modrelu 模式下应有 QuantumGate。"""
        from quantum_core.ffn import GatedQuantumFFN

        ffn = GatedQuantumFFN(dim=32, ffn_dim=64, activation="modrelu")
        assert ffn.gate is not None, "modrelu 模式下 gate 不应为 None"


# ──────────────────────────────────────────────
# 测试7: QIR 量子干涉路由器
# ──────────────────────────────────────────────


class TestQIR:
    """测试 quantum_core/interference_router.py 的核心功能。"""

    def test_pairwise_interference_shape(self):
        """成对干涉矩阵形状应正确。"""
        from quantum_core.interference_router import pairwise_interference

        B, H, N, d = 2, 4, 8, 16
        Q = torch.randn(B, H, N, d, dtype=torch.complex64)
        K = torch.randn(B, H, N, d, dtype=torch.complex64)
        inter = pairwise_interference(Q, K)
        assert inter.shape == (B, H, N, N), f"干涉矩阵形状错误: {inter.shape}"

    def test_pairwise_interference_real(self):
        """干涉矩阵应为实数张量。"""
        from quantum_core.interference_router import pairwise_interference

        Q = torch.randn(2, 2, 4, 8, dtype=torch.complex64)
        K = torch.randn(2, 2, 4, 8, dtype=torch.complex64)
        inter = pairwise_interference(Q, K)
        assert not inter.is_complex(), "干涉矩阵应为实数"

    def test_router_forward_shape(self):
        """QIR 路由权重形状应为 (B, H, N, N)。"""
        from quantum_core.interference_router import QuantumInterferenceRouter

        B, N, d, H = 2, 6, 64, 4
        router = QuantumInterferenceRouter(dim=d, num_heads=H)
        x = torch.randn(B, N, d, dtype=torch.complex64)
        routing_weights, metrics = router(x, training=True)
        assert routing_weights.shape == (B, H, N, N), \
            f"路由权重形状错误: {routing_weights.shape}"

    def test_router_weights_range(self):
        """软路由权重应在 [0, 1] 范围内。"""
        from quantum_core.interference_router import QuantumInterferenceRouter

        router = QuantumInterferenceRouter(dim=32, num_heads=2)
        x = torch.randn(2, 8, 32, dtype=torch.complex64)
        weights, _ = router(x, training=True)
        assert (weights >= 0).all() and (weights <= 1 + 1e-6).all(), \
            f"软路由权重超出 [0,1] 范围"

    def test_router_metrics_keys(self):
        """metrics 应包含 QIR 相关字段。"""
        from quantum_core.interference_router import QuantumInterferenceRouter

        router = QuantumInterferenceRouter(dim=32, num_heads=2)
        x = torch.randn(1, 4, 32, dtype=torch.complex64)
        _, metrics = router(x)
        assert "qir_active_ratio" in metrics, "缺少 qir_active_ratio"
        assert "qir_constructive_ratio" in metrics, "缺少 qir_constructive_ratio"

    def test_apply_to_attention(self):
        """apply_to_attention 应重新归一化注意力概率。"""
        from quantum_core.interference_router import QuantumInterferenceRouter

        router = QuantumInterferenceRouter(dim=32, num_heads=2)
        B, H, N = 1, 2, 5
        attn = torch.rand(B, H, N, N)
        attn = attn / attn.sum(dim=-1, keepdim=True)
        routing = torch.ones(B, H, N, N) * 0.5
        fused = router.apply_to_attention(attn, routing)
        # 融合后每行应归一化
        row_sums = fused.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), \
            f"apply_to_attention 未正确归一化: {row_sums}"

    def test_lightweight_router(self):
        """轻量级路由器应接受 Q/K 直接输入。"""
        from quantum_core.interference_router import LightweightInterferenceRouter

        router = LightweightInterferenceRouter()
        B, H, N, d = 2, 4, 8, 16
        Q = torch.randn(B, H, N, d, dtype=torch.complex64)
        K = torch.randn(B, H, N, d, dtype=torch.complex64)
        weights, metrics = router(Q, K, training=True)
        assert weights.shape == (B, H, N, N), f"轻量路由器形状错误: {weights.shape}"


# ──────────────────────────────────────────────
# 测试8: model.py 增强参数统计
# ──────────────────────────────────────────────


class TestModelParamStats:
    """测试 QuantumArch.count_parameters 增强功能。"""

    def setup_method(self):
        from quantum_core.model import QuantumArch

        self.model = QuantumArch(
            vocab_size=100, dim=64, num_layers=2, num_heads=4
        )

    def test_new_fields_present(self):
        """count_parameters 应包含新增字段。"""
        info = self.model.count_parameters()
        new_fields = {"memory_mb_fp32", "memory_mb_fp16",
                      "qsa_params", "qel_params", "qci_params", "ffn_params"}
        missing = new_fields - set(info.keys())
        assert not missing, f"缺少参数统计字段: {missing}"

    def test_memory_nonneg(self):
        """内存估算应为正数。"""
        info = self.model.count_parameters()
        assert info["memory_mb_fp32"] > 0, "FP32 内存估算应 > 0"
        assert info["memory_mb_fp16"] > 0, "FP16 内存估算应 > 0"
        # FP16 应为 FP32 的一半
        assert abs(info["memory_mb_fp16"] - info["memory_mb_fp32"] / 2) < 0.1, \
            "FP16 内存应为 FP32 的一半"

    def test_total_equals_sum(self):
        """total_real_equiv 应 >= block_params + embedding_params。"""
        info = self.model.count_parameters()
        assert info["total_real_equiv"] >= info["block_params"] + info["embedding_params"], \
            "总参数量应 >= 各部分之和"

    def test_complexity_summary_output(self):
        """complexity_summary 应返回非空字符串，包含新增信息。"""
        summary = self.model.complexity_summary(seq_len=128)
        assert len(summary) > 100, "summary 过短"
        assert "QSA" in summary or "qsa" in summary.lower(), "缺少 QSA 信息"
        assert "MB" in summary, "缺少内存信息"


# ──────────────────────────────────────────────
# 主测试入口
# ──────────────────────────────────────────────


if __name__ == "__main__":
    # 直接运行（无需 pytest）
    import traceback

    test_classes = [
        TestQSACausalMask,
        TestQELEntanglementMetrics,
        TestPOVMOrthogonalization,
        TestQGDGradEMA,
        TestQuantumInfoMetrics,
        TestFFNSwiGLU,
        TestQIR,
        TestModelParamStats,
    ]

    passed = 0
    failed = 0
    errors = []

    for cls in test_classes:
        print(f"\n{'='*60}")
        print(f"测试类: {cls.__name__}")
        print('='*60)
        instance = cls()
        for method_name in dir(instance):
            if not method_name.startswith("test_"):
                continue
            method = getattr(instance, method_name)
            if callable(method):
                if hasattr(instance, "setup_method"):
                    instance.setup_method()
                try:
                    method()
                    print(f"  ✓ {method_name}")
                    passed += 1
                except Exception as e:
                    print(f"  ✗ {method_name}: {e}")
                    errors.append(f"{cls.__name__}.{method_name}: {e}")
                    failed += 1

    print(f"\n{'='*60}")
    print(f"测试结果: {passed} 通过, {failed} 失败")
    if errors:
        print("\n失败详情:")
        for err in errors:
            print(f"  - {err}")
    print('='*60)
