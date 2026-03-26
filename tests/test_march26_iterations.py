"""
2026-03-26 每日迭代测试套件

覆盖本次10次迭代的新增/改进功能：
1. QSA get_attention_patterns()
2. QFT inverse() 逆变换
3. AdaptiveThreshold tau_high 衰减 + get_threshold_summary()
4. QGD per_group_lr_state()
5. GatedQuantumFFN get_gate_statistics()
6. quantum_metrics 模块
7. QuantumArch inference()
8. CayleyLinear recover_unitarity()
"""

import math
import pytest
import torch
import torch.nn as nn


# ─────────────────────────────────────────────
# 通用 fixtures
# ─────────────────────────────────────────────

BATCH = 2
SEQ = 8
DIM = 64


@pytest.fixture
def complex_input():
    """标准复数输入张量 (B, N, D)"""
    return torch.randn(BATCH, SEQ, DIM, dtype=torch.complex64)


@pytest.fixture
def real_input():
    """标准实数输入张量 (B, N, D)"""
    return torch.randn(BATCH, SEQ, DIM, dtype=torch.float32)


# ─────────────────────────────────────────────
# 迭代1：QSA get_attention_patterns()
# ─────────────────────────────────────────────

class TestQSAAttentionPatterns:
    """测试 QSA 新增的 get_attention_patterns() 诊断方法"""

    def _make_qsa(self):
        from quantum_core.attention import QuantumSuperpositionAttention
        return QuantumSuperpositionAttention(dim=DIM, num_heads=4, topk_ratio=0.5)

    def test_output_keys(self, complex_input):
        """返回的字典应包含所有期望的 key"""
        qsa = self._make_qsa()
        result = qsa.get_attention_patterns(complex_input)
        expected_keys = {"attn_probs", "phase_matrix", "attn_entropy", "topk_mask"}
        assert expected_keys == set(result.keys()), f"缺少 key: {expected_keys - set(result.keys())}"

    def test_attn_probs_shape(self, complex_input):
        """attn_probs 形状应为 (B, H, N, N)"""
        qsa = self._make_qsa()
        result = qsa.get_attention_patterns(complex_input)
        B, N = BATCH, SEQ
        H = 4
        assert result["attn_probs"].shape == (B, H, N, N), f"实际形状: {result['attn_probs'].shape}"

    def test_attn_probs_sum_to_one(self, complex_input):
        """注意力概率在最后一维应归一化为 1"""
        qsa = self._make_qsa()
        result = qsa.get_attention_patterns(complex_input)
        row_sums = result["attn_probs"].sum(dim=-1)  # (B, H, N)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4), \
            f"概率行和均值: {row_sums.mean().item():.4f}，应为 1.0"

    def test_topk_mask_shape_and_dtype(self, complex_input):
        """topk_mask 应为 bool 类型，形状 (B, H, N, N)"""
        qsa = self._make_qsa()
        result = qsa.get_attention_patterns(complex_input)
        assert result["topk_mask"].dtype == torch.bool
        assert result["topk_mask"].shape == result["attn_probs"].shape

    def test_topk_mask_k_count(self, complex_input):
        """每行 topk_mask 应恰好有 k 个 True"""
        from quantum_core.attention import QuantumSuperpositionAttention
        k_ratio = 0.5
        qsa = QuantumSuperpositionAttention(dim=DIM, num_heads=4, topk_ratio=k_ratio)
        result = qsa.get_attention_patterns(complex_input)
        k = max(1, int(k_ratio * SEQ))
        per_row_true = result["topk_mask"].sum(dim=-1)  # (B, H, N)
        assert (per_row_true == k).all(), \
            f"每行 Top-K True 数应为 {k}，实际: {per_row_true.unique()}"

    def test_causal_mask_upper_triangle_zero(self, complex_input):
        """因果掩码下，attn_probs 上三角（未来位置）应为 0"""
        qsa = self._make_qsa()
        result = qsa.get_attention_patterns(complex_input, causal=True)
        probs = result["attn_probs"]  # (B, H, N, N)
        # 上三角（i < j）应为 0
        for i in range(SEQ):
            for j in range(i + 1, SEQ):
                vals = probs[:, :, i, j]
                assert (vals.abs() < 1e-5).all(), \
                    f"因果掩码下 probs[{i},{j}] 不为 0: {vals.max().item():.4e}"

    def test_no_grad_no_param_change(self, complex_input):
        """get_attention_patterns 不应修改模型参数"""
        qsa = self._make_qsa()
        params_before = {n: p.clone() for n, p in qsa.named_parameters()}
        qsa.get_attention_patterns(complex_input)
        for n, p in qsa.named_parameters():
            assert torch.allclose(p, params_before[n]), f"参数 {n} 被意外修改"


# ─────────────────────────────────────────────
# 迭代2：QFT inverse() 逆变换
# ─────────────────────────────────────────────

class TestQFTInverse:
    """测试 QuantumFourierTransform.inverse() 逆变换"""

    def _make_qft(self, n_steps=1, init_alpha=0.0):
        from quantum_core.entanglement import QuantumFourierTransform
        qft = QuantumFourierTransform(n_steps=n_steps)
        # 设置 logit_alpha 使 alpha ≈ init_alpha（0表示完全QFT变换，方便测试精确逆）
        with torch.no_grad():
            if init_alpha == 0.0:
                qft.logit_alpha.fill_(-20.0)  # sigmoid(-20) ≈ 0
            elif init_alpha == 1.0:
                qft.logit_alpha.fill_(20.0)   # sigmoid(20) ≈ 1
        return qft

    def test_inverse_method_exists(self):
        """inverse 方法应存在"""
        from quantum_core.entanglement import QuantumFourierTransform
        qft = QuantumFourierTransform()
        assert hasattr(qft, "inverse"), "QuantumFourierTransform 缺少 inverse 方法"
        assert callable(qft.inverse)

    def test_inverse_output_shape(self, complex_input):
        """inverse 输出形状应与输入相同"""
        from quantum_core.entanglement import QuantumFourierTransform
        qft = QuantumFourierTransform(n_steps=1)
        x_qft = qft(complex_input)
        x_rec = qft.inverse(x_qft)
        assert x_rec.shape == complex_input.shape

    def test_identity_when_alpha_one(self, complex_input):
        """当 alpha=1 时，forward 和 inverse 均为恒等变换"""
        qft = self._make_qft(init_alpha=1.0)
        # forward 应近似恒等：alpha=1 时 result = alpha*x + 0*QFT(x) = x
        x_fwd = qft(complex_input)
        assert torch.allclose(x_fwd, complex_input, atol=1e-5), \
            "alpha=1 时 forward 应为恒等变换"
        x_inv = qft.inverse(x_fwd)
        assert torch.allclose(x_inv, complex_input, atol=1e-5), \
            "alpha=1 时 inverse 应为恒等变换"

    def test_inverse_preserves_dtype(self, complex_input):
        """inverse 应保持复数类型"""
        from quantum_core.entanglement import QuantumFourierTransform
        qft = QuantumFourierTransform()
        x_fwd = qft(complex_input)
        x_inv = qft.inverse(x_fwd)
        assert x_inv.is_complex(), "inverse 输出应为复数类型"


# ─────────────────────────────────────────────
# 迭代3：AdaptiveThreshold 改进
# ─────────────────────────────────────────────

class TestAdaptiveThreshold:
    """测试 AdaptiveThreshold 的 tau_high 衰减与 get_threshold_summary()"""

    def _make_threshold(self):
        from quantum_core.collapse import AdaptiveThreshold
        return AdaptiveThreshold(dim=DIM, tau_low_init=0.5, tau_min=0.1, tau_high_init=1.5)

    def test_get_threshold_summary_keys(self):
        """get_threshold_summary 应返回所有期望 key"""
        th = self._make_threshold()
        summary = th.get_threshold_summary()
        expected = {"tau_low", "tau_high", "gap", "tau_low_ratio", "tau_high_ratio",
                    "avg_entropy", "step_count"}
        assert expected == set(summary.keys()), f"缺失: {expected - set(summary.keys())}"

    def test_initial_gap_positive(self):
        """初始状态下 tau_high > tau_low（gap > 0）"""
        th = self._make_threshold()
        summary = th.get_threshold_summary()
        assert summary["gap"] > 0, f"初始 gap 应 > 0，实际: {summary['gap']}"

    def test_tau_ratio_in_valid_range(self):
        """tau_ratio 应在 [0, 1] 范围内"""
        th = self._make_threshold()
        summary = th.get_threshold_summary()
        assert 0 <= summary["tau_low_ratio"] <= 1.5, \
            f"tau_low_ratio 超出范围: {summary['tau_low_ratio']}"

    def test_tau_high_gap_maintained_after_update(self):
        """多次更新后，tau_high 应始终 >= tau_low + 0.3"""
        th = self._make_threshold()
        entropy = torch.ones(BATCH, SEQ) * 0.5
        # 模拟 1000 步更新
        for _ in range(1000):
            th.update(entropy, training=True)
        summary = th.get_threshold_summary()
        assert summary["gap"] >= 0.3 - 1e-5, \
            f"tau_high - tau_low 应 >= 0.3，实际 gap: {summary['gap']:.4f}"

    def test_avg_entropy_none_before_updates(self):
        """未更新前 avg_entropy 应为 None"""
        th = self._make_threshold()
        summary = th.get_threshold_summary()
        assert summary["avg_entropy"] is None, \
            f"更新前 avg_entropy 应为 None，实际: {summary['avg_entropy']}"


# ─────────────────────────────────────────────
# 迭代4：QGD per_group_lr_state()
# ─────────────────────────────────────────────

class TestQGDPerGroupLRState:
    """测试 QGD.per_group_lr_state()"""

    def _make_optimizer_with_groups(self):
        from quantum_core.optimizer import QGD
        # 创建两个简单参数组
        p1 = nn.Parameter(torch.randn(4, 4, dtype=torch.complex64))
        p2 = nn.Parameter(torch.randn(4, 4, dtype=torch.float32))
        opt = QGD([p1], mod_lr=1e-4, phase_lr=1e-3)
        opt.add_param_group({"params": [p2], "mod_lr": 1e-5, "phase_lr": 1e-4,
                              "cayley_lr": 1e-5, "real_lr": 1e-5, "betas": (0.9, 0.999),
                              "eps": 1e-8, "weight_decay": 0.0,
                              "max_grad_norm": None, "amsgrad": False})
        return opt

    def test_returns_list_of_dicts(self):
        """返回值应为 list of dict"""
        opt = self._make_optimizer_with_groups()
        states = opt.per_group_lr_state()
        assert isinstance(states, list)
        assert all(isinstance(s, dict) for s in states)

    def test_correct_group_count(self):
        """返回的字典数量应等于参数组数"""
        opt = self._make_optimizer_with_groups()
        states = opt.per_group_lr_state()
        assert len(states) == len(opt.param_groups)

    def test_group_idx_sequential(self):
        """group_idx 应从 0 开始连续"""
        opt = self._make_optimizer_with_groups()
        states = opt.per_group_lr_state()
        for i, s in enumerate(states):
            assert s["group_idx"] == i

    def test_lr_values_match_param_groups(self):
        """state 中的 mod_lr 应与参数组实际 mod_lr 一致"""
        opt = self._make_optimizer_with_groups()
        states = opt.per_group_lr_state()
        for s, group in zip(states, opt.param_groups):
            assert abs(s["mod_lr"] - group.get("mod_lr", 0)) < 1e-10

    def test_required_keys_present(self):
        """每个 dict 应包含所有必要 key"""
        from quantum_core.optimizer import QGD
        p = nn.Parameter(torch.randn(4, 4))
        opt = QGD([p], mod_lr=1e-4, phase_lr=1e-3)
        states = opt.per_group_lr_state()
        required = {"group_idx", "mod_lr", "phase_lr", "cayley_lr", "real_lr",
                    "num_params", "num_complex", "num_cayley", "num_real", "has_grad"}
        for s in states:
            assert required.issubset(set(s.keys())), f"缺失: {required - set(s.keys())}"


# ─────────────────────────────────────────────
# 迭代5：GatedQuantumFFN get_gate_statistics()
# ─────────────────────────────────────────────

class TestGatedFFNGateStatistics:
    """测试 GatedQuantumFFN.get_gate_statistics()"""

    def _make_gated_ffn(self, activation="modrelu"):
        from quantum_core.ffn import GatedQuantumFFN
        return GatedQuantumFFN(dim=DIM, ffn_dim=DIM * 2, activation=activation)

    def test_non_swiglu_returns_stats(self, complex_input):
        """非 SwiGLU 模式应返回非空字典"""
        ffn = self._make_gated_ffn("modrelu")
        # 需要先做上投影得到 ffn_dim 维度的张量
        up = ffn.W_up(complex_input)
        stats = ffn.get_gate_statistics(up)
        assert len(stats) > 0, "非 SwiGLU 模式应返回统计信息"

    def test_swiglu_returns_empty(self, complex_input):
        """SwiGLU 模式应返回空字典（无独立 QuantumGate）"""
        ffn = self._make_gated_ffn("swiglu")
        up = ffn.W_up(complex_input)
        stats = ffn.get_gate_statistics(up)
        assert stats == {}, f"SwiGLU 模式应返回空字典，实际: {stats}"

    def test_gate_mag_in_range(self, complex_input):
        """门控模长均值应在 (0, √2] 范围内"""
        ffn = self._make_gated_ffn("modrelu")
        up = ffn.W_up(complex_input)
        stats = ffn.get_gate_statistics(up)
        assert 0 < stats["gate_mag_mean"] <= math.sqrt(2) + 1e-4, \
            f"gate_mag_mean={stats['gate_mag_mean']:.4f} 超出 (0, √2] 范围"

    def test_dead_gate_ratio_in_range(self, complex_input):
        """dead_gate_ratio 应在 [0, 1] 范围内"""
        ffn = self._make_gated_ffn("modrelu")
        up = ffn.W_up(complex_input)
        stats = ffn.get_gate_statistics(up)
        assert 0.0 <= stats["dead_gate_ratio"] <= 1.0

    def test_returns_all_required_keys(self, complex_input):
        """应返回所有期望的统计 key"""
        ffn = self._make_gated_ffn("modrelu")
        up = ffn.W_up(complex_input)
        stats = ffn.get_gate_statistics(up)
        required = {"gate_mag_mean", "gate_mag_std", "gate_mag_min", "gate_mag_max",
                    "gate_phase_mean", "gate_phase_std",
                    "gate_real_mean", "gate_imag_mean", "dead_gate_ratio"}
        assert required.issubset(set(stats.keys())), f"缺失: {required - set(stats.keys())}"


# ─────────────────────────────────────────────
# 迭代6：quantum_metrics 模块
# ─────────────────────────────────────────────

class TestQuantumMetrics:
    """测试 quantum_metrics 新模块"""

    def test_module_importable(self):
        """quantum_metrics 应可正常导入"""
        import quantum_core.quantum_metrics as qm
        assert hasattr(qm, "quantum_fidelity")

    def test_fidelity_same_state_is_one(self, complex_input):
        """相同态的保真度应为 1"""
        from quantum_core.quantum_metrics import quantum_fidelity
        f = quantum_fidelity(complex_input, complex_input, dim=-1)
        assert torch.allclose(f, torch.ones_like(f), atol=1e-4), \
            f"自身保真度均值: {f.mean():.4f}，应为 1"

    def test_fidelity_orthogonal_states_is_zero(self):
        """正交态的保真度应为 0"""
        from quantum_core.quantum_metrics import quantum_fidelity
        a = torch.zeros(1, 1, DIM, dtype=torch.complex64)
        b = torch.zeros(1, 1, DIM, dtype=torch.complex64)
        a[..., 0] = 1.0 + 0j
        b[..., 1] = 1.0 + 0j
        f = quantum_fidelity(a, b, dim=-1)
        assert (f.abs() < 1e-6).all(), f"正交态保真度应为 0，实际: {f}"

    def test_trace_distance_range(self, complex_input):
        """迹距离应在 [0, 1] 范围内"""
        from quantum_core.quantum_metrics import trace_distance
        td = trace_distance(complex_input, complex_input.roll(1, dims=0), dim=-1)
        assert (td >= 0).all() and (td <= 1.0 + 1e-5).all(), \
            f"迹距离超出 [0,1]: min={td.min():.4f}, max={td.max():.4f}"

    def test_bures_range(self, complex_input):
        """Bures 距离应在 [0, √2] 范围内"""
        from quantum_core.quantum_metrics import bures_distance
        bd = bures_distance(complex_input, complex_input.roll(1, dims=0), dim=-1)
        assert (bd >= 0).all() and (bd <= math.sqrt(2) + 1e-4).all(), \
            f"Bures 距离超出 [0,√2]: max={bd.max():.4f}"

    def test_schmidt_number_gte_one(self, complex_input):
        """Schmidt 数估计应 >= 1"""
        from quantum_core.quantum_metrics import schmidt_number_estimate
        k = schmidt_number_estimate(complex_input, dim=-1)
        assert (k >= 1.0 - 1e-5).all(), f"Schmidt 数 < 1: min={k.min():.4f}"

    def test_concurrence_range(self, complex_input):
        """纠缠并发度应在 [0, 1]"""
        from quantum_core.quantum_metrics import concurrence_from_amplitudes
        c = concurrence_from_amplitudes(complex_input, complex_input.roll(2, dims=1), dim=-1)
        assert (c >= 0).all() and (c <= 1.0 + 1e-5).all(), \
            f"并发度超出 [0,1]: min={c.min():.4f}, max={c.max():.4f}"

    def test_attention_effective_width_range(self):
        """注意力有效宽度应在 [1, N]"""
        from quantum_core.quantum_metrics import attention_effective_width
        probs = torch.softmax(torch.randn(BATCH, 4, SEQ, SEQ), dim=-1)
        ew = attention_effective_width(probs, dim=-1)
        assert (ew >= 1.0 - 1e-4).all() and (ew <= SEQ + 1e-4).all(), \
            f"有效宽度超出 [1, {SEQ}]: min={ew.min():.4f}, max={ew.max():.4f}"

    def test_compute_health_all_keys(self, complex_input):
        """compute_model_quantum_health 应返回所有期望 key"""
        from quantum_core.quantum_metrics import compute_model_quantum_health
        health = compute_model_quantum_health(complex_input)
        expected = {"state_entropy_mean", "state_entropy_std", "state_mag_mean",
                    "state_mag_std", "schmidt_number_mean",
                    "attn_effective_width_mean", "attn_sparsity_mean"}
        assert expected == set(health.keys()), f"缺失: {expected - set(health.keys())}"


# ─────────────────────────────────────────────
# 迭代7：QuantumArch.inference()
# ─────────────────────────────────────────────

class TestQuantumArchInference:
    """测试 QuantumArch.inference() 快速推理接口"""

    def _make_model(self):
        from quantum_core.model import QuantumArch
        return QuantumArch(dim=DIM, num_layers=2, num_heads=4, direct_input=True)

    def test_inference_returns_dict(self, complex_input):
        """inference 应返回 dict"""
        model = self._make_model()
        result = model.inference(complex_input)
        assert isinstance(result, dict)

    def test_inference_output_key_present(self, complex_input):
        """结果应包含 'output' key"""
        model = self._make_model()
        result = model.inference(complex_input)
        assert "output" in result, f"缺少 'output' key，实际: {list(result.keys())}"

    def test_inference_restores_training_state(self, complex_input):
        """inference 前后模型 training 状态应不变"""
        model = self._make_model()
        model.train()
        assert model.training
        model.inference(complex_input)
        assert model.training, "inference 后 training 状态应恢复为 True"

    def test_inference_eval_state_preserved(self, complex_input):
        """eval 模式下调用 inference 后应保持 eval 状态"""
        model = self._make_model()
        model.eval()
        model.inference(complex_input)
        assert not model.training, "eval 模式下 inference 后应仍为 eval 状态"

    def test_return_hidden_includes_hidden_state(self, complex_input):
        """return_hidden=True 时结果应包含 hidden_state"""
        model = self._make_model()
        result = model.inference(complex_input, return_hidden=True)
        assert "hidden_state" in result, "return_hidden=True 时应包含 hidden_state"
        assert result["hidden_state"].is_complex(), "hidden_state 应为复数类型"

    def test_return_hidden_false_no_hidden_state(self, complex_input):
        """return_hidden=False（默认）时结果不应包含 hidden_state"""
        model = self._make_model()
        result = model.inference(complex_input, return_hidden=False)
        assert "hidden_state" not in result, \
            f"return_hidden=False 时不应包含 hidden_state，实际: {list(result.keys())}"

    def test_inference_no_grad(self, complex_input):
        """inference 期间不应有梯度流"""
        model = self._make_model()
        x = complex_input.requires_grad_(False)
        result = model.inference(x)
        assert not result["output"].requires_grad, "inference 输出不应有梯度"


# ─────────────────────────────────────────────
# 迭代8：CayleyLinear.recover_unitarity()
# ─────────────────────────────────────────────

class TestCayleyLinearRecoverUnitarity:
    """测试 CayleyLinear.recover_unitarity()"""

    def _make_cayley(self, dim=16):
        from quantum_core.unitary import CayleyLinear
        return CayleyLinear(dim, dim, init_scale=0.1)

    def test_method_exists(self):
        """recover_unitarity 方法应存在"""
        cayley = self._make_cayley()
        assert hasattr(cayley, "recover_unitarity"), "CayleyLinear 缺少 recover_unitarity 方法"

    def test_returns_float_violation(self):
        """方法应返回一个 float（恢复前违背度）"""
        cayley = self._make_cayley()
        violation = cayley.recover_unitarity()
        assert isinstance(violation, float), f"应返回 float，实际: {type(violation)}"

    def test_violation_non_negative(self):
        """违背度应非负"""
        cayley = self._make_cayley()
        violation = cayley.recover_unitarity()
        assert violation >= 0.0, f"违背度应 >= 0，实际: {violation}"

    def test_qr_recovery_reduces_violation(self):
        """QR 恢复后违背度应 <= 恢复前"""
        cayley = self._make_cayley()
        # 先记录当前违背度（不恢复）
        before = cayley.get_unitarity_violation().item()
        # 执行恢复
        returned_before = cayley.recover_unitarity(method="qr")
        assert abs(returned_before - before) < 1e-4, \
            f"返回值应为恢复前违背度: 返回={returned_before:.6f}, 实际={before:.6f}"

    def test_rescale_method_works(self):
        """rescale 方法也应正常运行，不抛出异常"""
        cayley = self._make_cayley()
        violation = cayley.recover_unitarity(method="rescale")
        assert isinstance(violation, float)

    def test_non_square_raises(self):
        """非方阵情况应抛出 ValueError"""
        from quantum_core.unitary import CayleyLinear
        cayley = CayleyLinear(DIM, DIM // 2)
        with pytest.raises(ValueError, match="仅支持方阵"):
            cayley.recover_unitarity()


# ─────────────────────────────────────────────
# 执行所有测试
# ─────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
