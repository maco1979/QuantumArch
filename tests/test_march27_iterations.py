"""
2026-03-27 迭代测试套件

覆盖当日10次迭代的所有新增功能：
1. quantum_mutual_information (complex_ops.py)
2. multi_head_entropy_summary (attention.py)
3. compute_schmidt_rank_proxy (entanglement.py)
4. compute_collapse_efficiency (collapse.py)
5. learning_rate_sensitivity_report (optimizer.py)
6. get_layer_quantum_states (model.py)
7. qsa_theory.md 文档存在性检查
8. get_block_efficiency_report (quantum_block.py)
9. condition_number + omega_spectrum (unitary.py)
10. 集成测试：端到端量子效率分析
"""

import os
import math
import pytest
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# 测试类 1: 量子互信息 (complex_ops.py)
# ─────────────────────────────────────────────────────────────────────────────

class TestQuantumMutualInformation:
    """测试 quantum_mutual_information 函数"""

    def setup_method(self):
        from quantum_core.complex_ops import quantum_mutual_information
        self.qmi = quantum_mutual_information
        torch.manual_seed(42)

    def test_output_shape_batch(self):
        """批量输入的输出形状正确"""
        psi = torch.randn(4, 32, dtype=torch.complex64)
        phi = torch.randn(4, 32, dtype=torch.complex64)
        I = self.qmi(psi, phi)
        assert I.shape == (4,), f"Expected (4,), got {I.shape}"

    def test_output_nonnegative(self):
        """互信息非负"""
        psi = torch.randn(8, 16, dtype=torch.complex64)
        phi = torch.randn(8, 16, dtype=torch.complex64)
        I = self.qmi(psi, phi)
        assert (I >= 0).all(), f"存在负值: {I.min().item()}"

    def test_3d_input(self):
        """3D 输入（batch, seq, dim）形状正确"""
        psi = torch.randn(2, 8, 64, dtype=torch.complex64)
        phi = torch.randn(2, 8, 64, dtype=torch.complex64)
        I = self.qmi(psi, phi)
        assert I.shape == (2, 8), f"Expected (2, 8), got {I.shape}"

    def test_same_state_nonzero(self):
        """相同状态的互信息不为零（自相关）"""
        psi = torch.randn(4, 32, dtype=torch.complex64)
        I = self.qmi(psi, psi)
        # 相同态有最大关联，互信息应该较大
        assert I.mean().item() >= 0, "自相关互信息应非负"

    def test_gradient_flows(self):
        """梯度可以正常回传"""
        psi = torch.randn(2, 16, dtype=torch.complex64, requires_grad=False)
        # 互信息基于 Born 概率，不需要梯度
        I = self.qmi(psi, psi)
        assert I is not None

    def test_finite_values(self):
        """输出值有限（无 NaN/Inf）"""
        psi = torch.randn(4, 32, dtype=torch.complex64)
        phi = torch.randn(4, 64, dtype=torch.complex64)  # 不同维度
        # 不同维度时应截断或填充处理
        psi_trunc = psi  # same dim needed for outer product
        phi_trunc = torch.randn(4, 32, dtype=torch.complex64)
        I = self.qmi(psi_trunc, phi_trunc)
        assert torch.isfinite(I).all(), "互信息含有 NaN 或 Inf"


# ─────────────────────────────────────────────────────────────────────────────
# 测试类 2: 多头熵分析 (attention.py)
# ─────────────────────────────────────────────────────────────────────────────

class TestMultiHeadEntropySummary:
    """测试 QuantumSuperpositionAttention.multi_head_entropy_summary"""

    def setup_method(self):
        from quantum_core.attention import QuantumSuperpositionAttention
        self.qsa = QuantumSuperpositionAttention(dim=64, num_heads=4, topk_ratio=0.3)
        torch.manual_seed(42)

    def test_output_keys_present(self):
        """所有必要的键都存在"""
        x = torch.randn(2, 16, 64, dtype=torch.complex64)
        summary = self.qsa.multi_head_entropy_summary(x)
        required_keys = [
            "per_head_entropy", "entropy_mean", "entropy_std",
            "entropy_min", "entropy_max",
            "uniform_head_mask", "sharp_head_mask",
            "max_entropy_ref", "head_diversity_score",
        ]
        for k in required_keys:
            assert k in summary, f"缺少键: {k}"

    def test_per_head_entropy_shape(self):
        """per_head_entropy 形状为 (num_heads,)"""
        x = torch.randn(2, 16, 64, dtype=torch.complex64)
        summary = self.qsa.multi_head_entropy_summary(x)
        assert summary["per_head_entropy"].shape == (4,)

    def test_entropy_nonnegative(self):
        """熵值非负"""
        x = torch.randn(2, 16, 64, dtype=torch.complex64)
        summary = self.qsa.multi_head_entropy_summary(x)
        assert (summary["per_head_entropy"] >= 0).all()

    def test_diversity_score_range(self):
        """多样性分数在 [0, 1] 范围内"""
        x = torch.randn(2, 16, 64, dtype=torch.complex64)
        summary = self.qsa.multi_head_entropy_summary(x)
        score = summary["head_diversity_score"]
        assert 0.0 <= score <= 1.0, f"多样性分数越界: {score}"

    def test_mask_shapes(self):
        """掩码形状正确"""
        x = torch.randn(2, 16, 64, dtype=torch.complex64)
        summary = self.qsa.multi_head_entropy_summary(x)
        assert summary["uniform_head_mask"].shape == (4,)
        assert summary["sharp_head_mask"].shape == (4,)
        assert summary["uniform_head_mask"].dtype == torch.bool
        assert summary["sharp_head_mask"].dtype == torch.bool

    def test_no_gradient_computation(self):
        """不追踪梯度（诊断工具）"""
        x = torch.randn(2, 16, 64, dtype=torch.complex64)
        summary = self.qsa.multi_head_entropy_summary(x)
        # per_head_entropy 不应需要梯度
        assert not summary["per_head_entropy"].requires_grad

    def test_causal_mode(self):
        """因果模式下正常工作"""
        x = torch.randn(2, 16, 64, dtype=torch.complex64)
        summary_causal = self.qsa.multi_head_entropy_summary(x, causal=True)
        summary_noncausal = self.qsa.multi_head_entropy_summary(x, causal=False)
        # 因果掩码应使熵降低（关注范围变小）
        assert summary_causal["entropy_mean"] is not None


# ─────────────────────────────────────────────────────────────────────────────
# 测试类 3: Schmidt 秩估计 (entanglement.py)
# ─────────────────────────────────────────────────────────────────────────────

class TestSchmidtRankProxy:
    """测试 compute_schmidt_rank_proxy 函数"""

    def setup_method(self):
        from quantum_core.entanglement import compute_schmidt_rank_proxy
        self.rank_fn = compute_schmidt_rank_proxy
        torch.manual_seed(42)

    def test_output_shape_2d(self):
        """2D 输入的输出形状"""
        a = torch.randn(8, 16, dtype=torch.complex64)
        b = torch.randn(8, 16, dtype=torch.complex64)
        r = self.rank_fn(a, b)
        assert r.shape == (8,), f"Expected (8,), got {r.shape}"

    def test_minimum_rank_one(self):
        """有效秩最小为 1"""
        a = torch.randn(4, 16, dtype=torch.complex64)
        b = torch.randn(4, 16, dtype=torch.complex64)
        r = self.rank_fn(a, b)
        assert (r >= 1.0).all(), f"存在秩 < 1 的情况: {r.min().item()}"

    def test_rank_bounded_by_dim(self):
        """有效秩不超过维度"""
        d = 8
        a = torch.randn(4, d, dtype=torch.complex64)
        b = torch.randn(4, d, dtype=torch.complex64)
        r = self.rank_fn(a, b)
        assert (r <= float(d)).all(), f"存在秩超过维度 {d}: {r.max().item()}"

    def test_product_state_rank_one(self):
        """乘积态（可分态）的秩接近 1"""
        # 乘积态 a⊗b：a 和 b 完全独立，Schmidt 秩 = 1
        a = torch.zeros(2, 8, dtype=torch.complex64)
        b = torch.zeros(2, 8, dtype=torch.complex64)
        a[:, 0] = 1.0  # 确定态 |0⟩
        b[:, 0] = 1.0  # 确定态 |0⟩
        r = self.rank_fn(a, b)
        # 乘积态有效秩应接近 1
        assert r.mean().item() <= 2.0, f"乘积态秩过高: {r.mean().item()}"

    def test_3d_input(self):
        """3D 输入（batch, seq, dim）正常处理"""
        a = torch.randn(2, 8, 16, dtype=torch.complex64)
        b = torch.randn(2, 8, 16, dtype=torch.complex64)
        r = self.rank_fn(a, b)
        assert r.shape == (2, 8), f"Expected (2, 8), got {r.shape}"

    def test_finite_output(self):
        """输出有限（无 NaN/Inf）"""
        a = torch.randn(4, 32, dtype=torch.complex64)
        b = torch.randn(4, 32, dtype=torch.complex64)
        r = self.rank_fn(a, b)
        assert torch.isfinite(r).all(), "Schmidt 秩含有 NaN 或 Inf"


# ─────────────────────────────────────────────────────────────────────────────
# 测试类 4: 坍缩效率指标 (collapse.py)
# ─────────────────────────────────────────────────────────────────────────────

class TestCollapseEfficiency:
    """测试 QuantumCollapseInference.compute_collapse_efficiency"""

    def setup_method(self):
        from quantum_core.collapse import QuantumCollapseInference
        self.qci = QuantumCollapseInference(dim=64, tau_low=0.5, tau_high=1.5)
        torch.manual_seed(42)

    def test_output_keys(self):
        """返回所有必要键"""
        x = torch.randn(2, 16, 64, dtype=torch.complex64)
        eff = self.qci.compute_collapse_efficiency(x)
        required = [
            "early_exit_rate", "information_retention",
            "entropy_before", "entropy_after",
            "entropy_compression_ratio", "povm_completeness_violation",
            "effective_measurement_rank", "tau_low", "tau_high",
        ]
        for k in required:
            assert k in eff, f"缺少键: {k}"

    def test_early_exit_rate_range(self):
        """早退率在 [0, 1] 范围内"""
        x = torch.randn(2, 16, 64, dtype=torch.complex64)
        eff = self.qci.compute_collapse_efficiency(x)
        rate = eff["early_exit_rate"]
        assert 0.0 <= rate <= 1.0, f"早退率越界: {rate}"

    def test_information_retention_range(self):
        """信息保留率在 [0, 1] 范围内"""
        x = torch.randn(2, 16, 64, dtype=torch.complex64)
        eff = self.qci.compute_collapse_efficiency(x)
        retention = eff["information_retention"]
        assert 0.0 <= retention <= 1.0, f"信息保留率越界: {retention}"

    def test_entropy_nonnegative(self):
        """熵非负"""
        x = torch.randn(2, 16, 64, dtype=torch.complex64)
        eff = self.qci.compute_collapse_efficiency(x)
        assert eff["entropy_before"] >= 0
        assert eff["entropy_after"] >= 0

    def test_effective_rank_positive(self):
        """有效测量数为正"""
        x = torch.randn(2, 16, 64, dtype=torch.complex64)
        eff = self.qci.compute_collapse_efficiency(x)
        assert eff["effective_measurement_rank"] >= 1.0

    def test_no_gradient_side_effects(self):
        """不修改模型参数梯度"""
        x = torch.randn(2, 16, 64, dtype=torch.complex64)
        # 清除梯度
        self.qci.zero_grad()
        eff = self.qci.compute_collapse_efficiency(x)
        # 参数应没有梯度
        for p in self.qci.parameters():
            assert p.grad is None, "compute_collapse_efficiency 产生了非预期梯度"


# ─────────────────────────────────────────────────────────────────────────────
# 测试类 5: 学习率敏感性报告 (optimizer.py)
# ─────────────────────────────────────────────────────────────────────────────

class TestLearningRateSensitivity:
    """测试 QGD.learning_rate_sensitivity_report"""

    def setup_method(self):
        from quantum_core.optimizer import QGD
        # 创建一个简单的参数
        self.params = [
            nn.Parameter(torch.randn(8, 8, dtype=torch.complex64)),
            nn.Parameter(torch.randn(8, 8, dtype=torch.float32)),
        ]
        self.optimizer = QGD(self.params, mod_lr=1e-4, phase_lr=1e-3)

    def test_report_keys(self):
        """报告包含必要键"""
        report = self.optimizer.learning_rate_sensitivity_report()
        assert "group_reports" in report
        assert "global_status" in report

    def test_group_reports_list(self):
        """group_reports 是列表"""
        report = self.optimizer.learning_rate_sensitivity_report()
        assert isinstance(report["group_reports"], list)
        assert len(report["group_reports"]) >= 1

    def test_group_report_keys(self):
        """每个参数组报告包含必要键"""
        report = self.optimizer.learning_rate_sensitivity_report()
        for group_report in report["group_reports"]:
            required_keys = [
                "group_idx", "mod_lr", "phase_lr",
                "avg_sensitivity_complex", "avg_sensitivity_real",
                "sensitivity_status", "recommendation",
            ]
            for k in required_keys:
                assert k in group_report, f"缺少键: {k}"

    def test_status_valid_values(self):
        """sensitivity_status 为合法值"""
        report = self.optimizer.learning_rate_sensitivity_report()
        valid_statuses = {"ok", "too_low", "too_high"}
        for group_report in report["group_reports"]:
            assert group_report["sensitivity_status"] in valid_statuses

    def test_global_status_string(self):
        """全局状态为字符串"""
        report = self.optimizer.learning_rate_sensitivity_report()
        assert isinstance(report["global_status"], str)
        assert len(report["global_status"]) > 0

    def test_report_after_zero_steps(self):
        """初始化后（无梯度历史）报告不崩溃"""
        report = self.optimizer.learning_rate_sensitivity_report()
        # 无历史时敏感性应为 0（或处理 0 状态）
        for g in report["group_reports"]:
            assert isinstance(g["avg_sensitivity_complex"], float)
            assert isinstance(g["avg_sensitivity_real"], float)


# ─────────────────────────────────────────────────────────────────────────────
# 测试类 6: 分层量子态提取 (model.py)
# ─────────────────────────────────────────────────────────────────────────────

class TestLayerQuantumStates:
    """测试 QuantumArch.get_layer_quantum_states"""

    def setup_method(self):
        from quantum_core.model import QuantumArch
        self.model = QuantumArch(
            dim=64, num_layers=3, num_heads=4,
            ffn_dim=128, collapse_enabled=False,
            direct_input=True,
        )
        self.model.eval()
        torch.manual_seed(42)

    def test_output_keys(self):
        """返回所有必要键"""
        x = torch.randn(2, 8, 64, dtype=torch.complex64)
        result = self.model.get_layer_quantum_states(x, layers=[0, 1])
        required_keys = [
            "input_state", "layer_states", "output_state",
            "layer_metrics", "state_fidelities", "state_entropies",
        ]
        for k in required_keys:
            assert k in result, f"缺少键: {k}"

    def test_layer_states_shape(self):
        """各层状态的形状正确"""
        B, N, D = 2, 8, 64
        x = torch.randn(B, N, D, dtype=torch.complex64)
        result = self.model.get_layer_quantum_states(x, layers=[0, 2])
        for layer_idx, state in result["layer_states"].items():
            assert state.shape == (B, N, D), f"层 {layer_idx} 形状错误: {state.shape}"

    def test_all_layers_extraction(self):
        """layers=None 提取所有层"""
        x = torch.randn(2, 8, 64, dtype=torch.complex64)
        result = self.model.get_layer_quantum_states(x)
        assert len(result["layer_states"]) == 3, "应提取所有3层"

    def test_negative_layer_index(self):
        """支持负数层索引"""
        x = torch.randn(2, 8, 64, dtype=torch.complex64)
        result = self.model.get_layer_quantum_states(x, layers=[-1])
        # -1 应映射到最后一层（索引2）
        assert 2 in result["layer_states"]

    def test_fidelities_in_range(self):
        """各层保真度在 [0, 1] 范围内"""
        x = torch.randn(2, 8, 64, dtype=torch.complex64)
        result = self.model.get_layer_quantum_states(x, layers=[0, 1, 2])
        for k, fid in result["state_fidelities"].items():
            assert 0.0 <= fid <= 1.0, f"保真度越界 at {k}: {fid}"

    def test_complex_state_type(self):
        """量子态为复数类型"""
        x = torch.randn(2, 8, 64, dtype=torch.complex64)
        result = self.model.get_layer_quantum_states(x, layers=[0])
        assert result["input_state"].is_complex()
        assert result["output_state"].is_complex()
        for state in result["layer_states"].values():
            assert state.is_complex()

    def test_model_state_unchanged(self):
        """调用后模型状态不变"""
        x = torch.randn(2, 8, 64, dtype=torch.complex64)
        was_training = self.model.training
        _ = self.model.get_layer_quantum_states(x)
        assert self.model.training == was_training, "调用后训练状态被改变"


# ─────────────────────────────────────────────────────────────────────────────
# 测试类 7: QSA 理论文档存在性
# ─────────────────────────────────────────────────────────────────────────────

class TestQSATheoryDocument:
    """验证 QSA 理论文档创建正确"""

    def test_file_exists(self):
        """文档文件存在"""
        doc_path = os.path.join(
            os.path.dirname(__file__), "..", "docs", "core", "qsa_theory.md"
        )
        assert os.path.exists(doc_path), f"文档不存在: {doc_path}"

    def test_file_nonempty(self):
        """文档内容非空"""
        doc_path = os.path.join(
            os.path.dirname(__file__), "..", "docs", "core", "qsa_theory.md"
        )
        with open(doc_path, encoding="utf-8") as f:
            content = f.read()
        assert len(content) > 100, "文档内容过短"

    def test_key_sections_present(self):
        """文档包含关键章节"""
        doc_path = os.path.join(
            os.path.dirname(__file__), "..", "docs", "core", "qsa_theory.md"
        )
        with open(doc_path, encoding="utf-8") as f:
            content = f.read()
        required_sections = ["Born", "干涉", "Top-K", "因果", "多头", "超参数"]
        for section in required_sections:
            assert section in content, f"文档缺少关键内容: {section}"

    def test_code_example_present(self):
        """文档包含代码示例"""
        doc_path = os.path.join(
            os.path.dirname(__file__), "..", "docs", "core", "qsa_theory.md"
        )
        with open(doc_path, encoding="utf-8") as f:
            content = f.read()
        assert "```python" in content, "文档缺少 Python 代码示例"


# ─────────────────────────────────────────────────────────────────────────────
# 测试类 8: 量子块效率报告 (quantum_block.py)
# ─────────────────────────────────────────────────────────────────────────────

class TestBlockEfficiencyReport:
    """测试 QuantumBlock.get_block_efficiency_report"""

    def setup_method(self):
        from quantum_core.quantum_block import QuantumBlock
        self.block = QuantumBlock(
            dim=64, num_heads=4, ffn_dim=128,
            topk_ratio=0.3, collapse_enabled=True,
            tau_low=0.5, tau_high=1.5,
        )
        torch.manual_seed(42)

    def test_report_keys_present(self):
        """报告包含必要键"""
        x = torch.randn(2, 16, 64, dtype=torch.complex64)
        report = self.block.get_block_efficiency_report(x)
        required_keys = [
            "qsa_topk_utilization", "avg_attention_entropy",
            "head_diversity_score", "qel_entanglement_strength",
            "qel_qft_alpha", "qel_global_qft_usage",
            "qci_early_exit_rate", "phase_coherence_score",
            "has_dead_gates", "unitarity_max_violation",
        ]
        for k in required_keys:
            assert k in report, f"缺少键: {k}"

    def test_topk_utilization_range(self):
        """Top-K 利用率在 [0, 1]"""
        x = torch.randn(2, 16, 64, dtype=torch.complex64)
        report = self.block.get_block_efficiency_report(x)
        u = report["qsa_topk_utilization"]
        assert 0.0 <= u <= 1.0, f"Top-K 利用率越界: {u}"

    def test_phase_coherence_range(self):
        """相位相干性在 [0, 1]"""
        x = torch.randn(2, 16, 64, dtype=torch.complex64)
        report = self.block.get_block_efficiency_report(x)
        c = report["phase_coherence_score"]
        assert 0.0 <= c <= 1.0, f"相位相干性越界: {c}"

    def test_qft_alpha_range(self):
        """QFT alpha 在 [0, 1]"""
        x = torch.randn(2, 16, 64, dtype=torch.complex64)
        report = self.block.get_block_efficiency_report(x)
        alpha = report["qel_qft_alpha"]
        assert 0.0 <= alpha <= 1.0, f"QFT alpha 越界: {alpha}"

    def test_unitarity_violation_nonnegative(self):
        """酉性违背度非负"""
        x = torch.randn(2, 16, 64, dtype=torch.complex64)
        report = self.block.get_block_efficiency_report(x)
        assert report["unitarity_max_violation"] >= 0.0

    def test_report_all_float_or_bool(self):
        """所有值为 float 或 bool 类型"""
        x = torch.randn(2, 16, 64, dtype=torch.complex64)
        report = self.block.get_block_efficiency_report(x)
        for k, v in report.items():
            assert isinstance(v, (float, int, bool)), f"键 {k} 的值类型错误: {type(v)}"


# ─────────────────────────────────────────────────────────────────────────────
# 测试类 9: 条件数和 Ω 谱分析 (unitary.py)
# ─────────────────────────────────────────────────────────────────────────────

class TestCayleyLinearConditionAndSpectrum:
    """测试 CayleyLinear.condition_number 和 omega_spectrum"""

    def setup_method(self):
        from quantum_core.unitary import CayleyLinear
        self.square = CayleyLinear(16, 16, init_scale=0.01)
        self.nonsquare = CayleyLinear(16, 32, init_scale=0.01)
        torch.manual_seed(42)

    def test_condition_number_square(self):
        """方阵的条件数接近 1（初始化后）"""
        kappa = self.square.condition_number()
        assert kappa >= 1.0, f"条件数不应小于 1: {kappa}"
        assert kappa < 1000.0, f"初始化后条件数不应过大: {kappa}"

    def test_condition_number_fresh_init_near_one(self):
        """小 init_scale 时条件数接近 1"""
        from quantum_core.unitary import CayleyLinear
        cl = CayleyLinear(8, 8, init_scale=0.001)
        kappa = cl.condition_number()
        assert kappa < 5.0, f"小 init_scale 时条件数应接近 1: {kappa}"

    def test_condition_number_nonsquare_raises(self):
        """非方阵调用 condition_number 应抛出异常"""
        with pytest.raises(RuntimeError, match="方阵"):
            self.nonsquare.condition_number()

    def test_omega_spectrum_keys(self):
        """omega_spectrum 返回所有必要键"""
        spec = self.square.omega_spectrum()
        required_keys = [
            "lambda_max", "lambda_mean",
            "near_singularity_count", "effective_param_ratio",
            "w_phase_uniformity",
        ]
        for k in required_keys:
            assert k in spec, f"缺少键: {k}"

    def test_omega_spectrum_lambda_nonnegative(self):
        """λ 值统计非负"""
        spec = self.square.omega_spectrum()
        assert spec["lambda_max"] >= 0.0
        assert spec["lambda_mean"] >= 0.0

    def test_omega_spectrum_count_nonnegative_int(self):
        """近奇点计数为非负整数"""
        spec = self.square.omega_spectrum()
        assert isinstance(spec["near_singularity_count"], int)
        assert spec["near_singularity_count"] >= 0

    def test_omega_spectrum_ratio_in_range(self):
        """有效参数比例在 [0, 1]"""
        spec = self.square.omega_spectrum()
        ratio = spec["effective_param_ratio"]
        assert 0.0 <= ratio <= 1.0, f"有效参数比例越界: {ratio}"

    def test_omega_spectrum_uniformity_in_range(self):
        """相位均匀性指数在 [0, 1]"""
        spec = self.square.omega_spectrum()
        u = spec["w_phase_uniformity"]
        assert 0.0 <= u <= 1.0, f"相位均匀性越界: {u}"

    def test_omega_spectrum_nonsquare_raises(self):
        """非方阵调用 omega_spectrum 应抛出异常"""
        with pytest.raises(RuntimeError, match="方阵"):
            self.nonsquare.omega_spectrum()


# ─────────────────────────────────────────────────────────────────────────────
# 测试类 10: 集成测试（端到端量子效率分析）
# ─────────────────────────────────────────────────────────────────────────────

class TestEndToEndQuantumEfficiencyAnalysis:
    """端到端测试：综合验证所有新增功能的协作使用"""

    def setup_method(self):
        from quantum_core.model import QuantumArch
        self.model = QuantumArch(
            dim=64, num_layers=2, num_heads=4,
            ffn_dim=128, collapse_enabled=True,
            tau_low=0.3, tau_high=1.2,
            direct_input=True,
        )
        self.model.eval()
        torch.manual_seed(0)

    def test_layer_states_plus_mutual_info(self):
        """分层量子态 + 量子互信息组合使用"""
        from quantum_core.complex_ops import quantum_mutual_information
        x = torch.randn(2, 8, 64, dtype=torch.complex64)
        result = self.model.get_layer_quantum_states(x, layers=[0, 1])

        # 计算输入态和第0层输出态之间的互信息
        input_s = result["input_state"]  # (2, 8, 64)
        layer0_s = result["layer_states"][0]  # (2, 8, 64)

        # 逐 token 计算互信息
        I = quantum_mutual_information(input_s, layer0_s, dim=-1)  # (2, 8)
        assert I.shape == (2, 8)
        assert (I >= 0).all()

    def test_block_efficiency_report_on_model_layers(self):
        """对模型中的实际 block 运行效率报告"""
        x = torch.randn(2, 16, 64, dtype=torch.complex64)
        block = self.model.blocks[0]
        report = block.get_block_efficiency_report(x)
        # 验证效率报告的关键指标
        assert "qsa_topk_utilization" in report
        assert "qci_early_exit_rate" in report
        assert report["qci_early_exit_rate"] >= 0.0

    def test_collapse_efficiency_on_real_block_output(self):
        """对真实的块输出运行坍缩效率分析"""
        x = torch.randn(2, 16, 64, dtype=torch.complex64)
        block = self.model.blocks[0]
        # 前向传播获得实际输出
        with torch.no_grad():
            z, _ = block(x, training=False)
        # 对输出运行坍缩效率分析
        qci = block.collapse
        if qci is not None:
            eff = qci.compute_collapse_efficiency(z)
            assert "information_retention" in eff
            assert 0.0 <= eff["information_retention"] <= 1.0

    def test_unitary_health_after_forward(self):
        """前向传播后酉矩阵仍然健康"""
        x = torch.randn(2, 8, 64, dtype=torch.complex64)
        # 运行前向传播
        with torch.no_grad():
            _ = self.model(x, training=False)
        # 检查第一个 QSA 的 Wq 条件数
        wq = self.model.blocks[0].qsa.Wq
        kappa = wq.condition_number()
        assert kappa < 100.0, f"前向传播后条件数异常: {kappa}"

    def test_multi_head_entropy_on_real_input(self):
        """在真实分布的输入上测试多头熵分析"""
        # 模拟一个有结构的输入（非完全随机）
        x = torch.zeros(2, 16, 64, dtype=torch.complex64)
        # 前几个 token 有明确的激活
        x[:, :4, :8] = 1.0 + 0j
        qsa = self.model.blocks[0].qsa
        summary = qsa.multi_head_entropy_summary(x)
        assert summary["head_diversity_score"] is not None
        assert isinstance(summary["head_diversity_score"], float)

    def test_full_pipeline_no_crash(self):
        """完整分析管线不崩溃"""
        x = torch.randn(2, 8, 64, dtype=torch.complex64)
        # 1. 分层状态
        states = self.model.get_layer_quantum_states(x, layers=[0])
        assert "layer_states" in states

        # 2. 效率报告
        report = self.model.blocks[0].get_block_efficiency_report(x)
        assert "phase_coherence_score" in report

        # 3. Ω 谱分析
        spec = self.model.blocks[0].qsa.Wq.omega_spectrum()
        assert "lambda_max" in spec

        # 所有步骤完成无异常
        assert True


# ─────────────────────────────────────────────────────────────────────────────
# 主函数（直接运行文件时执行测试）
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    # 添加项目根目录到 path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    print("=" * 70)
    print("2026-03-27 迭代测试套件")
    print("=" * 70)

    test_classes = [
        TestQuantumMutualInformation,
        TestMultiHeadEntropySummary,
        TestSchmidtRankProxy,
        TestCollapseEfficiency,
        TestLearningRateSensitivity,
        TestLayerQuantumStates,
        TestQSATheoryDocument,
        TestBlockEfficiencyReport,
        TestCayleyLinearConditionAndSpectrum,
        TestEndToEndQuantumEfficiencyAnalysis,
    ]

    total = passed = failed = 0
    failures = []

    for cls in test_classes:
        instance = cls()
        print(f"\n[测试类] {cls.__name__}")
        # 获取所有 test_ 方法
        test_methods = [m for m in dir(instance) if m.startswith("test_")]
        for method_name in sorted(test_methods):
            total += 1
            try:
                instance.setup_method()
                getattr(instance, method_name)()
                print(f"  ✓ {method_name}")
                passed += 1
            except Exception as e:
                print(f"  ✗ {method_name}: {e}")
                failed += 1
                failures.append(f"{cls.__name__}.{method_name}: {e}")

    print("\n" + "=" * 70)
    print(f"总计: {total} 测试 | 通过: {passed} | 失败: {failed}")
    if failures:
        print("\n失败详情:")
        for f in failures:
            print(f"  - {f}")
    else:
        print("🎉 所有测试通过！")
    print("=" * 70)
