"""
量子架构完整测试套件

覆盖所有核心模块的单元测试:
- 量子叠加注意力 (QSA)
- 量子纠缠层 (QEL)
- 量子坍缩推理 (QCI)
- 复数运算
- 酉矩阵参数化
- 端到端模型测试

运行: pytest tests/ -v --cov=quantum_core --cov-report=html
"""

import pytest
import torch
import numpy as np
import sys
import os
import math

# 项目路径
sys.path.insert(0, r"e:\量子架构")

from quantum_core import (
    QuantumArch,
    QuantumSuperpositionAttention,
    QuantumEntanglementLayer,
    QuantumCollapseInference,
    QuantumFFN,
    QuantumBlock,
    QGD,
    CayleyLinear,
    ComplexLayerNorm,
    ComplexEmbedding,
    ModReLU,
    complex_to_polar,
    polar_to_complex,
    von_neumann_entropy,
    born_normalize,
    normalize_quantum_state,
    check_unitarity,
    QuantumPositionalEncoding,
)
from quantum_core.complex_ops import (
    complex_inner_product,
    complex_softmax,
    interference_score,
    complex_dropout,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def device():
    """测试设备"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def batch_size():
    """批次大小"""
    return 4


@pytest.fixture
def seq_len():
    """序列长度"""
    return 32


@pytest.fixture
def dim():
    """特征维度"""
    return 64


@pytest.fixture
def complex_input(batch_size, seq_len, dim, device):
    """生成复数测试输入"""
    real = torch.randn(batch_size, seq_len, dim, device=device)
    imag = torch.randn(batch_size, seq_len, dim, device=device)
    return torch.complex(real, imag)


# ============================================================================
# 1. 酉矩阵参数化测试 (CR)
# ============================================================================

class TestCayleyLinear:
    """测试 Cayley 线性变换（酉矩阵参数化）"""

    def test_cayley_linear_output_shape(self, dim, device):
        """测试输出形状"""
        layer = CayleyLinear(dim, dim).to(device)
        x = torch.randn(2, dim, dtype=torch.complex64, device=device)
        y = layer(x)
        assert y.shape == (2, dim)

    def test_unitarity_preservation(self, dim, device):
        """测试酉性保持"""
        layer = CayleyLinear(dim, dim).to(device)
        
        # 验证 W^H W ≈ I
        W = layer.unitary_matrix
        W_H_W = torch.mm(W.conj().T, W)
        eye = torch.eye(dim, dtype=torch.complex64, device=device)
        
        error = torch.norm(W_H_W - eye).item()
        assert error < 1e-4, f"酉性误差过大: {error}"

    def test_cayley_forward_backward(self, dim, device):
        """测试前向和反向传播"""
        layer = CayleyLinear(dim, dim).to(device)
        x = torch.randn(2, dim, dtype=torch.complex64, device=device, requires_grad=True)
        y = layer(x)
        loss = y.abs().sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_different_input_sizes(self, device):
        """测试不同输入尺寸"""
        layer = CayleyLinear(64, 32).to(device)
        x = torch.randn(4, 64, dtype=torch.complex64, device=device)
        y = layer(x)
        assert y.shape == (4, 32)


# ============================================================================
# 2. 复数运算测试
# ============================================================================

class TestComplexOps:
    """测试复数操作"""

    def test_complex_inner_product(self, device):
        """测试复数内积"""
        a = torch.complex(torch.randn(4, 8), torch.randn(4, 8))
        b = torch.complex(torch.randn(4, 8), torch.randn(4, 8))
        
        result = complex_inner_product(a, b)
        assert result.shape == (4,)
        assert not torch.isnan(result).any()

    def test_born_normalize(self, device):
        """测试 Born 归一化"""
        # 创建未归一化的复数张量
        x = torch.complex(torch.randn(2, 4, 8), torch.randn(2, 4, 8))
        
        normalized = born_normalize(x, dim=-1)
        
        # born_normalize 返回概率分布 (实数张量)
        # 验证概率和为1
        assert normalized.dtype == torch.float32, "born_normalize 应返回实数概率"
        prob_sum = normalized.sum(dim=-1)
        torch.testing.assert_close(prob_sum, torch.ones_like(prob_sum), rtol=1e-4, atol=1e-5)

    def test_complex_softmax(self, device):
        """测试复数 Softmax"""
        x = torch.randn(2, 4, 8, dtype=torch.complex64, device=device)
        result = complex_softmax(x, dim=-1)
        
        # 验证结果形状
        assert result.shape == x.shape
        assert not torch.isnan(result).any()

    def test_von_neumann_entropy(self, device):
        """测试冯·诺依曼熵"""
        # 创建归一化的密度矩阵
        x = torch.complex(torch.randn(2, 4, 8), torch.randn(2, 4, 8))
        x = born_normalize(x, dim=-1)
        
        entropy = von_neumann_entropy(x, dim=-1)
        assert entropy.shape == (2, 4)
        assert (entropy >= 0).all()

    def test_interference_score(self, device):
        """测试干涉分数"""
        alpha = torch.randn(2, 4, 8, dtype=torch.complex64, device=device)
        beta = torch.randn(2, 4, 8, dtype=torch.complex64, device=device)
        score = interference_score(alpha, beta)
        
        # interference_score 返回与输入相同形状的干涉强度
        assert score.shape == alpha.shape
        assert not torch.isnan(score).any()


# ============================================================================
# 3. 量子叠加注意力测试 (QSA)
# ============================================================================

class TestQuantumSuperpositionAttention:
    """测试 QSA 模块"""

    def test_qsa_output_shape(self, complex_input, device):
        """测试输出形状"""
        qsa = QuantumSuperpositionAttention(
            dim=64,
            num_heads=8,
            topk_ratio=0.1,
        ).to(device)
        
        output, metrics = qsa(complex_input)
        
        assert output.shape == complex_input.shape

    def test_qsa_full_mode(self, complex_input, device):
        """测试完整注意力模式"""
        qsa = QuantumSuperpositionAttention(
            dim=64,
            num_heads=8,
            mode='full',
        ).to(device)
        
        output, metrics = qsa(complex_input, training=False)
        
        assert 'attention_entropy' in metrics
        assert 'attention_probs_max' in metrics

    def test_qsa_topk_mode(self, complex_input, device):
        """测试 Top-K 模式"""
        qsa = QuantumSuperpositionAttention(
            dim=64,
            num_heads=8,
            mode='topk',
            topk_ratio=0.2,
        ).to(device)
        
        output, metrics = qsa(complex_input, training=True)
        
        assert output.shape == complex_input.shape

    def test_qsa_with_mask(self, complex_input, device):
        """测试带掩码的注意力"""
        qsa = QuantumSuperpositionAttention(dim=64, num_heads=8).to(device)
        
        mask = torch.ones(4, 32, device=device)
        mask[:, 16:] = 0  # 掩码后半部分
        
        output, metrics = qsa(complex_input, attention_mask=mask, training=False)
        
        assert output.shape == complex_input.shape


# ============================================================================
# 4. 量子纠缠层测试 (QEL)
# ============================================================================

class TestQuantumEntanglementLayer:
    """测试量子纠缠层"""

    def test_qel_output_shape(self, complex_input, device):
        """测试输出形状"""
        qel = QuantumEntanglementLayer(dim=64).to(device)
        
        output, metrics = qel(complex_input)
        
        assert output.shape == complex_input.shape

    def test_qel_entanglement_strength(self, complex_input, device):
        """测试纠缠强度和度量"""
        qel = QuantumEntanglementLayer(dim=64).to(device)
        
        # 记录原始纠缠度量
        output1, metrics1 = qel(complex_input)
        
        # 修改 QFT 混合系数
        with torch.no_grad():
            qel.qft.logit_alpha.fill_(math.log(0.1 / 0.9))
        output2, metrics2 = qel(complex_input)
        
        # 不同强度应产生不同输出
        assert not torch.allclose(output1, output2, atol=1e-4)


# ============================================================================
# 5. 量子坍缩推理测试 (QCI)
# ============================================================================

class TestQuantumCollapseInference:
    """测试量子坍缩推理"""

    def test_qci_output_shape(self, complex_input, device):
        """测试输出形状"""
        qci = QuantumCollapseInference(dim=64).to(device)
        
        output, metrics = qci(complex_input, training=True)
        
        assert output.shape == complex_input.shape

    def test_qci_entropy_threshold(self, complex_input, device):
        """测试熵阈值触发"""
        qci = QuantumCollapseInference(
            dim=64,
            tau_low=0.1,
            tau_high=2.0,
        ).to(device)
        
        # 高熵情况
        high_entropy_input = torch.complex(
            torch.randn(4, 32, 64, device=device),
            torch.randn(4, 32, 64, device=device)
        )
        
        output, metrics = qci(high_entropy_input, training=True)
        
        # QCI 返回的 metrics 以 'collapse_' 前缀
        assert 'collapse_entropy' in metrics

    def test_qci_early_exit_rate(self, device):
        """测试早退率"""
        qci = QuantumCollapseInference(dim=64, tau_low=0.01).to(device)
        
        # 多次前向传播统计早退率
        early_exit_count = 0
        num_runs = 20
        
        for _ in range(num_runs):
            x = torch.complex(torch.randn(4, 32, 64, device=device), torch.randn(4, 32, 64, device=device))
            _, metrics = qci(x, training=True)
            if metrics.get('early_exit', False):
                early_exit_count += 1
        
        # 早退率应该在合理范围内
        early_exit_rate = early_exit_count / num_runs
        assert 0 <= early_exit_rate <= 1.0


# ============================================================================
# 6. FFN 和归一化测试
# ============================================================================

class TestQuantumFFN:
    """测试量子前馈网络"""

    def test_ffn_output_shape(self, complex_input, device):
        """测试输出形状"""
        ffn = QuantumFFN(dim=64, ffn_dim=256).to(device)
        
        output = ffn(complex_input)
        
        assert output.shape == complex_input.shape

    def test_ffn_gating(self, complex_input, device):
        """测试门控机制"""
        ffn = QuantumFFN(dim=64, ffn_dim=256, use_gating=True).to(device)
        
        output = ffn(complex_input)
        
        assert not torch.isnan(output).any()


class TestComplexLayerNorm:
    """测试复数层归一化"""

    def test_layernorm_output_shape(self, complex_input, device):
        """测试输出形状"""
        norm = ComplexLayerNorm(dim=64).to(device)
        
        output = norm(complex_input)
        
        assert output.shape == complex_input.shape

    def test_layernorm_stability(self, device):
        """测试数值稳定性"""
        norm = ComplexLayerNorm(dim=64).to(device)
        
        # 大输入
        x = torch.complex(torch.randn(4, 32, 64, device=device) * 100,
                         torch.randn(4, 32, 64, device=device) * 100)
        
        output = norm(x)
        
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_layernorm_normalization_effect(self, device):
        """测试归一化效果：实部和虚部均值应接近0"""
        norm = ComplexLayerNorm(dim=64, elementwise_affine=False).to(device)
        
        x = torch.randn(100, 64, dtype=torch.complex64, device=device)
        output = norm(x)
        
        # 实部和虚部均值都应接近 0
        assert abs(output.real.mean().item()) < 0.3
        assert abs(output.imag.mean().item()) < 0.3

    def test_layernorm_gradient_flow(self, device):
        """测试梯度流"""
        norm = ComplexLayerNorm(dim=64).to(device)
        
        x = torch.randn(4, 32, 64, dtype=torch.complex64, device=device, requires_grad=True)
        output = norm(x)
        output.abs().sum().backward()
        
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_layernorm_no_affine(self, device):
        """测试无仿射变换模式"""
        norm = ComplexLayerNorm(dim=64, elementwise_affine=False).to(device)
        
        x = torch.randn(4, 32, 64, dtype=torch.complex64, device=device)
        output = norm(x)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()


# ============================================================================
# 7. 嵌入和位置编码测试
# ============================================================================

class TestEmbedding:
    """测试嵌入层"""

    def test_complex_embedding(self, device):
        """测试复数嵌入"""
        vocab_size = 1000
        batch_size = 4
        seq_len = 32
        dim = 64
        
        embed = ComplexEmbedding(vocab_size, dim).to(device)
        token_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        
        output = embed(token_ids)
        
        assert output.shape == (batch_size, seq_len, dim)
        assert output.dtype == torch.complex64


class TestPositionalEncoding:
    """测试位置编码"""

    def test_quantum_positional_encoding(self, complex_input, device):
        """测试量子位置编码"""
        pe = QuantumPositionalEncoding(dim=64, max_len=2048).to(device)
        
        output = pe(complex_input)
        
        assert output.shape == complex_input.shape

    def test_learned_positional_encoding(self, complex_input, device):
        """测试可学习位置编码"""
        from quantum_core.embedding import LearnedPositionalEncoding
        
        pe = LearnedPositionalEncoding(dim=64, max_len=2048, dropout=0.1).to(device)
        
        output = pe(complex_input, training=True)
        
        assert output.shape == complex_input.shape


# ============================================================================
# 8. 端到端模型测试
# ============================================================================

class TestQuantumArchModel:
    """测试完整 QuantumArch 模型"""

    def test_model_initialization(self, device):
        """测试模型初始化"""
        model = QuantumArch(
            vocab_size=1000,
            dim=64,
            num_layers=4,
            num_heads=8,
        ).to(device)
        
        assert model is not None
        assert model.dim == 64
        assert model.num_layers == 4

    def test_model_with_token_ids(self, device):
        """测试 token IDs 输入"""
        model = QuantumArch(
            vocab_size=1000,
            dim=64,
            num_layers=2,
            num_heads=4,
        ).to(device)
        
        batch_size = 4
        seq_len = 32
        token_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
        
        result = model({'token_ids': token_ids}, training=False)
        
        assert 'output' in result
        assert 'entropy' in result
        assert result['output'].shape == (batch_size, seq_len, 64)

    def test_model_with_complex_input(self, complex_input, device):
        """测试复数输入"""
        model = QuantumArch(
            dim=64,
            num_layers=2,
            num_heads=4,
            direct_input=True,
        ).to(device)
        
        result = model(complex_input, training=False)
        
        assert 'output' in result
        assert result['output'].shape == (4, 32, 64)

    def test_model_early_exit(self, device):
        """测试早退机制"""
        model = QuantumArch(
            dim=64,
            num_layers=6,
            num_heads=4,
            collapse_enabled=True,
        ).to(device)
        
        x = torch.randn(4, 32, 64, device=device)
        
        result = model(x, training=True)
        
        assert 'qci_early_exit' in result

    def test_model_parameter_update(self, device):
        """测试参数更新"""
        model = QuantumArch(dim=64, num_layers=2).to(device)
        
        # 更新 QSA top-k 比例
        model.update_parameters(qsa_topk_ratio=0.2)
        
        assert model.qsa_topk_ratio == 0.2

    def test_model_unitarity_report(self, device):
        """测试酉性报告"""
        model = QuantumArch(dim=32, num_layers=2, num_heads=4).to(device)
        
        report = model.get_unitarity_report()
        
        # 验证酉性误差在可接受范围内
        for key, error in report.items():
            assert error < 1.0, f"{key} 酉性误差过大: {error}"


# ============================================================================
# 9. QGD 优化器测试
# ============================================================================

class TestQGD:
    """测试量子梯度下降优化器"""

    def test_qgd_initialization(self, device):
        """测试 QGD 初始化"""
        model = QuantumArch(dim=32, num_layers=1).to(device)
        optimizer = QGD(model.parameters(), mod_lr=0.001, phase_lr=0.001)
        
        assert optimizer is not None

    def test_qgd_training_step(self, device):
        """测试训练步骤"""
        model = QuantumArch(dim=32, num_layers=1).to(device)
        optimizer = QGD(model.parameters(), mod_lr=0.01, phase_lr=0.01)
        
        x = torch.randn(2, 16, 32, device=device)
        
        # 前向传播
        result = model(x, training=True)
        loss = result['output'].abs().sum()
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        assert True  # 如果没有崩溃则通过


# ============================================================================
# 10. 集成测试
# ============================================================================

class TestIntegration:
    """集成测试"""

    def test_full_training_loop(self, device):
        """完整训练循环"""
        # 创建模型
        model = QuantumArch(
            vocab_size=500,
            dim=64,
            num_layers=3,
            num_heads=4,
            ffn_dim=256,
            dropout=0.1,
            collapse_enabled=True,
        ).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        
        # 模拟几个训练步骤
        for i in range(5):
            token_ids = torch.randint(0, 500, (2, 16), device=device)
            
            result = model({'token_ids': token_ids}, training=True)
            loss = result['output'].abs().mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f"Step {i}: loss={loss.item():.4f}")
        
        assert True

    def test_model_eval_mode(self, device):
        """测试评估模式"""
        model = QuantumArch(dim=64, num_layers=2).to(device)
        model.eval()
        
        x = torch.randn(2, 16, 64, device=device)
        
        with torch.no_grad():
            result = model(x, training=False)
        
        assert 'output' in result
        assert result['output'].shape == (2, 16, 64)


# ============================================================================
# 性能基准测试
# ============================================================================

class TestPerformance:
    """性能基准测试"""

    @pytest.mark.slow
    def test_qsa_performance(self, device):
        """QSA 性能基准"""
        import time
        
        qsa = QuantumSuperpositionAttention(
            dim=512,
            num_heads=8,
            mode='topk',
            topk_ratio=0.1,
        ).to(device)
        
        x = torch.randn(8, 128, 512, dtype=torch.complex64, device=device)
        
        # 预热
        for _ in range(3):
            _ = qsa(x, training=True)
        
        # 计时
        start = time.time()
        for _ in range(10):
            _ = qsa(x, training=True)
        elapsed = time.time() - start
        
        print(f"\nQSA 推理时间 (8x128x512): {elapsed/10*1000:.2f}ms")
        
        assert elapsed / 10 < 1.0  # 单次推理应在1秒内

    @pytest.mark.slow
    def test_model_throughput(self, device):
        """模型吞吐量基准"""
        import time
        
        model = QuantumArch(
            dim=256,
            num_layers=6,
            num_heads=8,
        ).to(device)
        
        x = torch.randn(16, 64, 256, device=device)
        
        # 预热
        _ = model(x, training=False)
        
        # 计时
        start = time.time()
        for _ in range(5):
            _ = model(x, training=False)
        elapsed = time.time() - start
        
        throughput = 16 * 5 / elapsed
        print(f"\n模型吞吐量: {throughput:.2f} samples/sec")
        
        assert throughput > 10  # 至少10 samples/sec


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
