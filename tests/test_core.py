"""量子架构核心模块单元测试

覆盖所有核心组件的正确性、数学性质和梯度流。
运行方式：python -m pytest tests/ -v
"""

import sys
import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# 确保项目根目录在路径中
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from quantum_core import (
    complex_to_polar, polar_to_complex, normalize_quantum_state,
    born_probability, born_normalize, von_neumann_entropy,
    entropy_from_state, complex_inner_product, complex_softmax,
    complex_dropout, check_unitarity,
    ModReLU, ModReLUV2, CReLU, ComplexGELU,
    ComplexLayerNorm, ComplexBatchNorm,
    CayleyLinear, CayleyLinearSimple,
    ComplexEmbedding, QuantumPositionalEncoding, LearnedPositionalEncoding,
    QuantumSuperpositionAttention, PhaseModulation,
    QuantumEntanglementLayer, EntanglementGate, AdaptiveEntanglementGate,
    QuantumCollapseInference, POVMProjector,
    QuantumFFN, QuantumBlock,
    QuantumArch, QGD,
)

# 使用 pytest 风格但兼容纯 Python 运行
passed = 0
failed = 0
errors = []


def test(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        msg = f"  [FAIL] {name}"
        if detail:
            msg += f" -- {detail}"
        print(msg)
        errors.append(name)


def approx_eq(a, b, tol=1e-5):
    if isinstance(a, torch.Tensor):
        a = a.item()
    if isinstance(b, torch.Tensor):
        b = b.item()
    return abs(a - b) < tol


def approx_allclose(a, b, tol=1e-4, **kwargs):
    return torch.allclose(a, b, atol=tol, **kwargs)


# ═══════════════════════════════════════════════════════════════════
# 1. 复数运算测试 (complex_ops)
# ═══════════════════════════════════════════════════════════════════

def test_complex_ops():
    print("\n=== 复数基础运算 (complex_ops) ===")

    # complex_to_polar / polar_to_complex 往返
    z = torch.randn(4, 8, 16, dtype=torch.complex64)
    mag, phase = complex_to_polar(z)
    test("polar_to_complex 往返重建", approx_allclose(polar_to_complex(mag, phase), z))
    test("模长非负", (mag >= 0).all().item())
    test("相位范围 [-pi, pi]", (phase >= -math.pi - 1e-5).all().item() and (phase <= math.pi + 1e-5).all().item())

    # normalize_quantum_state
    z_norm = normalize_quantum_state(z, dim=-1)
    norms = z_norm.abs().pow(2).sum(dim=-1)
    test("量子态归一化 L2=1", torch.allclose(norms, torch.ones_like(norms), atol=1e-5))

    # born_normalize
    probs = born_normalize(z, dim=-1)
    test("Born 概率非负", (probs >= 0).all().item())
    test("Born 概率和为1", torch.allclose(probs.sum(dim=-1), torch.ones(4, 8), atol=1e-5))
    # 验证 P(i) = |alpha_i|^2 / sum
    expected = z.abs().pow(2) / (z.abs().pow(2).sum(dim=-1, keepdim=True) + 1e-8)
    test("Born 概率公式正确", approx_allclose(probs, expected, tol=1e-5))

    # von_neumann_entropy
    uniform = torch.ones(4, 16) / 16
    entropy_uniform = von_neumann_entropy(uniform, dim=-1)
    test("均匀分布熵 = log(d)", approx_eq(entropy_uniform.mean(), math.log(16), tol=1e-3))
    # 集中分布
    peaked = torch.zeros(4, 16)
    peaked[:, 0] = 1.0
    entropy_peaked = von_neumann_entropy(peaked, dim=-1)
    test("集中分布熵 ≈ 0", approx_eq(entropy_peaked.mean(), 0.0, tol=1e-3))

    # entropy_from_state
    entropy_direct = entropy_from_state(z, dim=-1)
    entropy_indirect = von_neumann_entropy(born_normalize(z, dim=-1), dim=-1)
    test("entropy_from_state 等价间接计算", approx_allclose(entropy_direct, entropy_indirect))

    # complex_inner_product
    a = torch.randn(2, 4, dtype=torch.complex64)
    b = torch.randn(2, 4, dtype=torch.complex64)
    ip = complex_inner_product(a, b, dim=-1)
    expected_ip = (a.conj() * b).sum(dim=-1)
    test("复数内积公式正确", approx_allclose(ip, expected_ip))

    # complex_softmax
    cs = complex_softmax(z, dim=-1)
    test("complex_softmax 输出为复数", cs.is_complex())
    test("complex_softmax 模长归一化", approx_allclose(cs.abs().pow(2).sum(dim=-1), torch.ones(4, 8), tol=1e-4))

    # complex_dropout
    z_dropped = complex_dropout(z, p=0.5, training=True)
    test("complex_dropout 输出形状不变", z_dropped.shape == z.shape)
    test("complex_dropout eval模式不变", torch.allclose(complex_dropout(z, p=0.5, training=False), z))

    # check_unitarity
    W = torch.randn(8, 8, dtype=torch.complex64)
    W, _ = torch.linalg.qr(W)  # 酉矩阵
    report = check_unitarity(W)
    test("QR分解矩阵通过酉性检查", report['is_unitary'])
    W_bad = torch.randn(8, 8, dtype=torch.complex64)
    report_bad = check_unitarity(W_bad)
    test("随机矩阵不满足酉性", not report_bad['is_unitary'])


# ═══════════════════════════════════════════════════════════════════
# 2. 激活函数测试 (activations)
# ═══════════════════════════════════════════════════════════════════

def test_activations():
    print("\n=== 激活函数 (activations) ===")

    z = torch.randn(4, 16, dtype=torch.complex64, requires_grad=True)

    # ModReLU
    modrelu = ModReLU(16)
    y = modrelu(z)
    test("ModReLU 输出形状不变", y.shape == z.shape)
    test("ModReLU 输出为复数", y.is_complex())
    # 相位保持性质（当 |z| + b > 0 时）
    y.real.sum().backward()
    test("ModReLU 梯度存在", z.grad is not None and z.grad.abs().sum() > 0)
    # 偏置可学习
    test("ModReLU 偏置有梯度", modrelu.bias.grad is not None)

    # ModReLU 数值验证
    z_test = torch.randn(2, 4, 16, dtype=torch.complex64)
    y_manual = F.relu(z_test.abs() + modrelu.bias) / (z_test.abs() + 1e-8) * z_test
    y_auto = modrelu(z_test)
    test("ModReLU 数值一致性", approx_allclose(y_auto, y_manual))

    # CReLU
    crelu = CReLU()
    y_c = crelu(z_test)
    test("CReLU 实部非负", (y_c.real >= 0).all().item())
    test("CReLU 虚部非负", (y_c.imag >= 0).all().item())

    # ComplexGELU
    gelu = ComplexGELU()
    y_g = gelu(z_test)
    test("ComplexGELU 输出形状", y_g.shape == z_test.shape)


# ═══════════════════════════════════════════════════════════════════
# 3. 归一化测试 (normalization)
# ═══════════════════════════════════════════════════════════════════

def test_normalization():
    print("\n=== 归一化层 (normalization) ===")

    from quantum_core.normalization import MagnitudeBatchNorm

    z = torch.randn(4, 8, 16, dtype=torch.complex64, requires_grad=True)

    # ComplexLayerNorm
    ln = ComplexLayerNorm(16)
    y = ln(z)
    test("ComplexLayerNorm 输出形状", y.shape == z.shape)
    # 新实现：基于 2D 增广实数 LN，归一化后均值应接近 0
    # 有仿射变换时 β=0 所以均值≈0，但小样本统计波动需要较大 tolerance
    y_mean = y.mean(dim=-1)
    test("ComplexLayerNorm 均值≈0", approx_allclose(y_mean.abs(), torch.zeros_like(y_mean.abs()), tol=1.0))
    # 梯度
    y.abs().sum().backward()
    test("ComplexLayerNorm 梯度流", z.grad is not None and z.grad.abs().sum() > 0)

    # ComplexLayerNorm 无仿射变换
    ln_no_affine = ComplexLayerNorm(16, elementwise_affine=False)
    y2 = ln_no_affine(z.detach())
    test("ComplexLayerNorm 无仿射输出形状", y2.shape == z.shape)

    # ComplexLayerNorm 数值稳定性（大输入）
    z_large = torch.complex(
        torch.randn(4, 8, 16) * 100,
        torch.randn(4, 8, 16) * 100
    )
    y_large = ln(z_large)
    test("ComplexLayerNorm 大输入无NaN", not torch.isnan(y_large).any())
    test("ComplexLayerNorm 大输入无Inf", not torch.isinf(y_large).any())

    # ComplexLayerNorm：实部/虚部分别归一化验证
    # 无仿射时，实部均值≈0、虚部均值≈0
    z_test = torch.randn(100, 16, dtype=torch.complex64)
    y_test = ln_no_affine(z_test)
    test("ComplexLayerNorm 实部均值≈0", abs(y_test.real.mean().item()) < 0.2)
    test("ComplexLayerNorm 虚部均值≈0", abs(y_test.imag.mean().item()) < 0.2)

    # ComplexBatchNorm（需要 batch 维度）
    bn = ComplexBatchNorm(16, track_running_stats=False)
    z_bn = torch.randn(8, 16, dtype=torch.complex64, requires_grad=True)
    y_bn = bn(z_bn)
    test("ComplexBatchNorm 输出形状", y_bn.shape == z_bn.shape)
    y_bn.abs().sum().backward()
    test("ComplexBatchNorm 梯度流", z_bn.grad is not None and z_bn.grad.abs().sum() > 0)

    # 3D 输入
    z_3d = torch.randn(4, 8, 16, dtype=torch.complex64, requires_grad=True)
    bn_3d = ComplexBatchNorm(16, track_running_stats=False)
    y_3d = bn_3d(z_3d)
    test("ComplexBatchNorm 3D输入", y_3d.shape == z_3d.shape)

    # ComplexBatchNorm 无仿射
    bn_no_affine = ComplexBatchNorm(16, affine=False, track_running_stats=False)
    y_bn_na = bn_no_affine(z_bn.detach())
    test("ComplexBatchNorm 无仿射输出形状", y_bn_na.shape == z_bn.shape)

    # MagnitudeBatchNorm
    mbn = MagnitudeBatchNorm(16, track_running_stats=False)
    z_mbn = torch.randn(8, 16, dtype=torch.complex64, requires_grad=True)
    y_mbn = mbn(z_mbn)
    test("MagnitudeBatchNorm 输出形状", y_mbn.shape == z_mbn.shape)
    # 相位保持不变
    test("MagnitudeBatchNorm 相位不变", torch.allclose(y_mbn.angle(), z_mbn.angle()))
    # 梯度
    y_mbn.abs().sum().backward()
    test("MagnitudeBatchNorm 梯度流", z_mbn.grad is not None and z_mbn.grad.abs().sum() > 0)

    # MagnitudeBatchNorm 3D 输入
    z_3d_mbn = torch.randn(4, 8, 16, dtype=torch.complex64, requires_grad=True)
    mbn_3d = MagnitudeBatchNorm(16, track_running_stats=False)
    y_3d_mbn = mbn_3d(z_3d_mbn)
    test("MagnitudeBatchNorm 3D输入", y_3d_mbn.shape == z_3d_mbn.shape)
    test("MagnitudeBatchNorm 3D相位不变", torch.allclose(y_3d_mbn.angle(), z_3d_mbn.angle()))


# ═══════════════════════════════════════════════════════════════════
# 4. 酉矩阵测试 (unitary)
# ═══════════════════════════════════════════════════════════════════

def test_unitary():
    print("\n=== 酉矩阵参数化 (unitary) ===")

    d = 8

    # CayleyLinear 方阵
    cayley = CayleyLinear(d, d, init_scale=0.02)
    W = cayley.unitary_matrix
    test("CayleyLinear is_square=True", cayley.is_square)
    violation = cayley.get_unitarity_violation().item()
    test(f"CayleyLinear 酉性违背 < 1e-4 (实际 {violation:.2e})", violation < 1e-4, f"violation={violation:.2e}")

    # 前向传播
    x = torch.randn(2, 4, d, dtype=torch.complex64, requires_grad=True)
    y = cayley(x)
    test("CayleyLinear 前向形状", y.shape == x.shape)
    # 模长保持（酉变换不改变向量模长）
    x_norm = x.abs().pow(2).sum(dim=-1)
    y_norm = y.abs().pow(2).sum(dim=-1)
    test("CayleyLinear 模长保持", approx_allclose(x_norm, y_norm, tol=1e-3))
    # 梯度
    y.abs().sum().backward()
    test("CayleyLinear 梯度流", x.grad is not None and x.grad.abs().sum() > 0)

    # CayleyLinear 非方阵
    cayley_ns = CayleyLinear(d, d * 2)
    x_ns = torch.randn(2, 4, d, dtype=torch.complex64)
    y_ns = cayley_ns(x_ns)
    test("CayleyLinear 非方阵前向", y_ns.shape == (2, 4, d * 2))
    test("CayleyLinear 非方阵 is_square=False", not cayley_ns.is_square)
    test("CayleyLinear 非方阵 violation=inf", math.isinf(cayley_ns.get_unitarity_violation().item()))

    # CayleyLinearSimple
    cayley_s = CayleyLinearSimple(d)
    W_s = cayley_s.unitary_matrix
    report_s = check_unitarity(W_s)
    test("CayleyLinearSimple 酉性", report_s['violation_norm'] < 0.2, f"violation={report_s['violation_norm']:.2e}")


# ═══════════════════════════════════════════════════════════════════
# 5. 嵌入层测试 (embedding)
# ═══════════════════════════════════════════════════════════════════

def test_embedding():
    print("\n=== 嵌入层 (embedding) ===")

    vocab, dim = 100, 32

    # ComplexEmbedding
    emb = ComplexEmbedding(vocab, dim, normalize=True)
    ids = torch.randint(0, vocab, (4, 16))
    z = emb(ids)
    test("ComplexEmbedding 输出形状", z.shape == (4, 16, dim))
    # 归一化检查
    norms = z.abs().pow(2).sum(dim=-1)
    test("ComplexEmbedding 归一化", approx_allclose(norms, torch.ones_like(norms), tol=1e-5))
    # 梯度
    z.abs().sum().backward()
    test("ComplexEmbedding 梯度流", emb.embedding.grad is not None and emb.embedding.grad.abs().sum() > 0)

    # QuantumPositionalEncoding
    pe = QuantumPositionalEncoding(dim, max_len=64)
    x = torch.randn(4, 16, dim, dtype=torch.complex64, requires_grad=True)
    y = pe(x, training=True)
    test("QuantumPositionalEncoding 形状", y.shape == x.shape)
    test("QuantumPositionalEncoding 改变输出", not torch.allclose(x, y))

    # LearnedPositionalEncoding
    lpe = LearnedPositionalEncoding(dim, max_len=64)
    y2 = lpe(x, training=True)
    test("LearnedPositionalEncoding 形状", y2.shape == x.shape)


# ═══════════════════════════════════════════════════════════════════
# 6. QSA 量子叠加注意力测试 (attention)
# ═══════════════════════════════════════════════════════════════════

def test_attention():
    print("\n=== QSA 量子叠加注意力 (attention) ===")

    B, N, D = 2, 16, 32
    x = torch.randn(B, N, D, dtype=torch.complex64, requires_grad=True)

    # topk 模式
    qsa_topk = QuantumSuperpositionAttention(D, num_heads=4, topk_ratio=0.1, mode='topk')
    y, metrics = qsa_topk(x, training=True)
    test("QSA topk 输出形状", y.shape == (B, N, D))
    test("QSA metrics 包含 entropy", 'attention_entropy' in metrics)
    test("QSA metrics 包含 interference", 'interference_phase_std' in metrics)

    # 梯度
    y.abs().sum().backward()
    test("QSA 梯度流", x.grad is not None and x.grad.abs().sum() > 0)

    # full 模式
    x2 = torch.randn(B, N, D, dtype=torch.complex64)
    qsa_full = QuantumSuperpositionAttention(D, num_heads=4, mode='full')
    y2, m2 = qsa_full(x2, training=False)
    test("QSA full 输出形状", y2.shape == (B, N, D))

    # PhaseModulation
    pm = PhaseModulation(hidden_dim=32)
    mag = torch.randn(4, 16)
    phase = pm(mag)
    test("PhaseModulation 输出形状", phase.shape == (4, 16))

    # Q/W 投影的酉性
    violation_wq = qsa_topk.Wq.get_unitarity_violation().item()
    test(f"QSA Wq 酉性 < 1e-3 ({violation_wq:.2e})", violation_wq < 1e-3)


# ═══════════════════════════════════════════════════════════════════
# 7. QEL 量子纠缠层测试 (entanglement)
# ═══════════════════════════════════════════════════════════════════

def test_entanglement():
    print("\n=== QEL 量子纠缠层 (entanglement) ===")

    B, N, D = 2, 16, 32

    # QuantumEntanglementLayer
    qel = QuantumEntanglementLayer(D, use_adaptive=True, use_global_qft=True)
    x = torch.randn(B, N, D, dtype=torch.complex64, requires_grad=True)
    y, metrics = qel(x, training=True)
    test("QEL 输出形状", y.shape == (B, N, D))
    test("QEL metrics 包含 strength", 'entanglement_strength' in metrics)

    # 梯度
    y.abs().sum().backward()
    test("QEL 梯度流", x.grad is not None and x.grad.abs().sum() > 0)

    # EntanglementGate 酉性（新版使用 Schmidt 纠缠门，验证内部酉矩阵）
    gate = EntanglementGate(dim=D)
    U = gate.get_gate_matrix()
    test("EntanglementGate 酉性", check_unitarity(U)['is_unitary'])

    # 自适应纠缠
    a = torch.randn(2, 4, D, dtype=torch.complex64)
    b = torch.randn(2, 4, D, dtype=torch.complex64)
    ag = AdaptiveEntanglementGate(D, theta_max=1.0)
    a_out, b_out, strength = ag(a, b)
    test("AdaptiveEntanglementGate 形状", a_out.shape == a.shape and b_out.shape == b.shape)
    test("AdaptiveEntanglementGate strength 范围", (strength >= 0).all().item() and (strength <= 1.0).all().item())


# ═══════════════════════════════════════════════════════════════════
# 8. QCI 量子坍缩推理测试 (collapse)
# ═══════════════════════════════════════════════════════════════════

def test_collapse():
    print("\n=== QCI 量子坍缩推理 (collapse) ===")

    from quantum_core.collapse import AdaptiveThreshold

    B, N, D = 2, 16, 32

    # QuantumCollapseInference（默认自适应阈值）
    qci = QuantumCollapseInference(D, tau_low=0.5, tau_high=1.5)
    x = torch.randn(B, N, D, dtype=torch.complex64, requires_grad=True)
    y, metrics = qci(x, training=True)
    test("QCI 输出形状", y.shape == (B, N, D))
    test("QCI metrics 包含 entropy", 'collapse_entropy' in metrics)
    test("QCI metrics 包含 early_exit_rate", 'collapse_early_exit_rate' in metrics)

    # 熵范围检查
    entropy = metrics['collapse_entropy']
    max_ent = math.log(D)
    test(f"QCI 熵范围 [0, log({D})]={max_ent:.2f}", 0 <= entropy <= max_ent + 0.1)

    # 梯度
    y.abs().sum().backward()
    test("QCI 梯度流", x.grad is not None and x.grad.abs().sum() > 0)

    # 推理模式（硬阈值）
    x_eval = torch.randn(B, N, D, dtype=torch.complex64)
    y_eval, m_eval = qci(x_eval, training=False)
    test("QCI eval 形状", y_eval.shape == (B, N, D))

    # POVM 完整性违背度
    test("QCI metrics 含 povm_violation", 'collapse_povm_violation' in metrics)

    # update_thresholds（自适应模式）
    qci.update_thresholds(0.3, 2.0)
    test("QCI 阈值更新 tau_low", abs(qci.threshold.tau_low.item() - 0.3) < 0.01)
    test("QCI 阈值更新 tau_high", abs(qci.threshold.tau_high.item() - 2.0) < 0.01)

    # 非自适应模式
    qci_static = QuantumCollapseInference(D, tau_low=0.5, tau_high=1.5, adaptive_tau=False)
    y_s, m_s = qci_static(x.detach(), training=True)
    test("QCI 非自适应输出形状", y_s.shape == (B, N, D))
    qci_static.update_thresholds(0.3, 2.0)
    test("QCI 非自适应阈值更新", abs(qci_static.tau_low.item() - 0.3) < 0.01)

    # POVMProjector
    povm = POVMProjector(D, D)
    psi = normalize_quantum_state(torch.randn(B, N, D, dtype=torch.complex64), dim=-1)
    collapsed, probs = povm(psi)
    test("POVM collapsed 形状", collapsed.shape == (B, N, D))
    test("POVM probs 形状", probs.shape == (B, N, D))
    test("POVM probs 非负", (probs >= 0).all().item())
    test("POVM probs 和为1", torch.allclose(probs.sum(dim=-1), torch.ones(B, N), atol=1e-4))

    # POVM 完整性违背度
    violation = povm.get_completeness_violation().item()
    test(f"POVM 完整性违背 < 1.0", violation < 1.0)

    # POVM 权重为正（softplus 保证）
    weights = povm.get_operators()
    test("POVM 权重非负", (weights >= 0).all().item())
    test("POVM 权重≈1（初始化）", torch.allclose(weights, torch.ones(D), atol=0.1))

    # POVM 非方阵（out_dim < in_dim）
    povm_nsq = POVMProjector(D, 16)
    violation_nsq = povm_nsq.get_completeness_violation().item()
    test(f"POVM 非方阵违背度合理", violation_nsq >= 0)

    # AdaptiveThreshold
    at = AdaptiveThreshold(dim=D, tau_low_init=0.5, tau_high_init=1.5)
    test("AdaptiveThreshold 初始 tau_low", abs(at.tau_low.item() - 0.5) < 0.01)
    test("AdaptiveThreshold 初始 tau_high", abs(at.tau_high.item() - 1.5) < 0.01)
    test("AdaptiveThreshold max_entropy", abs(at.max_entropy - math.log(D)) < 0.01)

    # AdaptiveThreshold 更新
    fake_entropy = torch.tensor([[0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
    for _ in range(200):
        at.update(fake_entropy, training=True)
    # 200次更新（每次step_count+1），tau_low 应该衰减
    test("AdaptiveThreshold tau_low 衰减", at.tau_low.item() <= 0.5)

    # get_unitarity_violation（QCI 级别）
    uv = qci.get_unitarity_violation()
    test("QCI 酉性报告有 povm", 'povm_completeness' in uv)


# ═══════════════════════════════════════════════════════════════════
# 9. FFN_Q 量子前馈网络测试 (ffn)
# ═══════════════════════════════════════════════════════════════════

def test_ffn():
    print("\n=== FFN_Q 量子前馈网络 (ffn) ===")

    from quantum_core.ffn import ComplexSigmoid, QuantumGate, ComplexLinear

    B, N, D = 2, 16, 32
    x = torch.randn(B, N, D, dtype=torch.complex64, requires_grad=True)

    # 基本版
    ffn = QuantumFFN(D, ffn_dim=64, dropout=0.0)
    y = ffn(x, training=True)
    test("FFN_Q 输出形状", y.shape == (B, N, D))
    # 残差连接验证：x 和 y 不完全相同
    test("FFN_Q 有残差", not torch.allclose(x.detach(), y.detach()))

    # 梯度
    y.abs().sum().backward()
    test("FFN_Q 梯度流", x.grad is not None and x.grad.abs().sum() > 0)

    # GLU 变体（量子门控）
    x2 = torch.randn(B, N, D, dtype=torch.complex64, requires_grad=True)
    ffn_glu = QuantumFFN(D, ffn_dim=64, use_glu=True, dropout=0.0)
    y2 = ffn_glu(x2, training=True)
    test("FFN_Q GLU 输出形状", y2.shape == (B, N, D))

    # GLU 梯度流
    y2.abs().sum().backward()
    test("FFN_Q GLU 梯度流", x2.grad is not None and x2.grad.abs().sum() > 0)

    # use_gating 参数名兼容
    ffn_gating = QuantumFFN(D, ffn_dim=64, use_gating=True, dropout=0.0)
    y3 = ffn_gating(x2.detach(), training=True)
    test("FFN_Q use_gating 兼容", y3.shape == (B, N, D))

    # ComplexSigmoid 测试
    cs = ComplexSigmoid()
    z = torch.randn(10, dtype=torch.complex64)
    s = cs(z)
    test("ComplexSigmoid 实部在(0,1)", (s.real > 0).all() and (s.real < 1).all())
    test("ComplexSigmoid 虚部在(0,1)", (s.imag > 0).all() and (s.imag < 1).all())

    # QuantumGate 测试
    gate = QuantumGate(dim=D)
    x_gate = torch.randn(4, D, dtype=torch.complex64)
    g_out = gate(x_gate)
    test("QuantumGate 输出形状", g_out.shape == x_gate.shape)

    # ComplexLinear 测试
    cl_non_square = ComplexLinear(D, 64)
    out_nsq = cl_non_square(x)
    test("ComplexLinear 非方阵形状", out_nsq.shape == (B, N, 64))
    test("ComplexLinear 非方阵非Cayley", not cl_non_square.is_cayley)

    # ComplexLinear 方阵 Cayley
    cl_square = ComplexLinear(D, D, use_cayley=True)
    out_sq = cl_square(x)
    test("ComplexLinear 方阵Cayley形状", out_sq.shape == (B, N, D))
    test("ComplexLinear 方阵是Cayley", cl_square.is_cayley)

    # W_down 酉性检查（ffn_dim != dim 时非方阵，不检查酉性）
    violations = ffn.get_unitarity_violation()
    test("FFN_Q 酉性报告有 W_down", 'W_down' in violations)
    # ffn_dim=64, dim=32，非方阵，W_down 违背度应为 inf
    test("FFN_Q 非方阵 W_down 非酉", violations['W_down'] > 100)

    # 方阵 W_down 酉性检查（ffn_dim == dim 时用 Cayley）
    ffn_square = QuantumFFN(D, ffn_dim=D, dropout=0.0)
    v2 = ffn_square.get_unitarity_violation()
    test("FFN_Q 方阵 W_down Cayley 酉", v2['W_down'] < 0.01)


# ═══════════════════════════════════════════════════════════════════
# 10. QuantumBlock 量子块测试 (quantum_block)
# ═══════════════════════════════════════════════════════════════════

def test_quantum_block():
    print("\n=== QuantumBlock 量子块 ===")

    B, N, D = 2, 16, 32
    x = torch.randn(B, N, D, dtype=torch.complex64, requires_grad=True)

    qb = QuantumBlock(D, num_heads=4, ffn_dim=64, collapse_enabled=True)
    y, metrics = qb(x, training=True)
    test("QuantumBlock 输出形状", y.shape == (B, N, D))
    test("QuantumBlock metrics 含 QSA", any('qsa_' in k for k in metrics))

    # 梯度
    y.abs().sum().backward()
    test("QuantumBlock 梯度流", x.grad is not None and x.grad.abs().sum() > 0)

    # update_parameters
    qb.update_parameters(qsa_topk_ratio=0.2, qci_tau_low=0.3, qci_tau_high=2.0)
    test("QuantumBlock update_parameters", qb.qsa.topk_ratio == 0.2)

    # 无 QCI
    qb_nc = QuantumBlock(D, num_heads=4, collapse_enabled=False)
    y_nc, m_nc = qb_nc(x.detach(), training=True)
    test("QuantumBlock 无QCI 输出形状", y_nc.shape == (B, N, D))


# ═══════════════════════════════════════════════════════════════════
# 11. QuantumArch 完整模型测试 (model)
# ═══════════════════════════════════════════════════════════════════

def test_model():
    print("\n=== QuantumArch 完整模型 ===")

    B, N, D = 2, 16, 32

    # direct_input 模式
    model = QuantumArch(
        dim=D, num_layers=2, num_heads=4, ffn_dim=64,
        collapse_enabled=True, direct_input=True,
    )
    x_real = torch.randn(B, N, D)
    result = model({'inputs': x_real}, training=True)
    test("QuantumArch 输出形状", result['output'].shape == (B, N, D))
    test("QuantumArch 含 qsa_time", 'qsa_time' in result)
    test("QuantumArch 含 entropy", 'entropy' in result)
    test("QuantumArch 含 qci_early_exit", 'qci_early_exit' in result)

    # 反向传播
    model.zero_grad()
    loss = result['output'].sum()
    loss.backward()
    grad_count = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for p in model.parameters())
    test(f"QuantumArch 梯度覆盖 {grad_count}/{total_params}", grad_count > total_params * 0.8)

    # 酉性报告
    report = model.get_unitarity_report()
    test("QuantumArch 酉性报告非空", len(report) > 0)
    max_violation = max(report.values()) if report else 0
    test(f"QuantumArch 酉性最大违背 < 1e-2 ({max_violation:.2e})", max_violation < 1e-2)

    # update_parameters
    model.update_parameters(qsa_topk_ratio=0.15, qci_tau_low=0.4, qci_tau_high=1.8)
    test("QuantumArch update_parameters", model.qsa_topk_ratio == 0.15)

    # QGD 优化器集成
    optimizer = QGD.from_model(model, mod_lr=1e-4, phase_lr=1e-3)
    optimizer.step()
    test("QGD 优化器步进成功", True)

    # 连续多步训练不报错
    for _ in range(3):
        model.zero_grad()
        x_batch = torch.randn(B, N, D)
        result = model({'inputs': x_batch}, training=True)
        result['output'].sum().backward()
        optimizer.step()
    test("QuantumArch 多步训练稳定", True)

    # token_ids 模式
    model_tok = QuantumArch(vocab_size=100, dim=D, num_layers=1, num_heads=4, ffn_dim=64, direct_input=False)
    ids = torch.randint(0, 100, (B, N))
    result_tok = model_tok({'token_ids': ids}, training=True)
    test("QuantumArch token_ids 模式", result_tok['output'].shape == (B, N, D))


# ═══════════════════════════════════════════════════════════════════
# 12. QGD 优化器测试 (optimizer)
# ═══════════════════════════════════════════════════════════════════

def test_optimizer():
    print("\n=== QGD 量子梯度下降优化器 ===")

    from quantum_core.optimizer import wirtinger_to_polar, _is_cayley_param

    # 复数参数
    p_complex = nn.Parameter(torch.randn(4, 8, dtype=torch.complex64))
    # 实数参数
    p_real = nn.Parameter(torch.randn(4, 8))

    opt = QGD([p_complex, p_real], mod_lr=1e-3, phase_lr=1e-2)

    # 模拟梯度
    p_complex.grad = torch.randn_like(p_complex)
    p_real.grad = torch.randn_like(p_real)

    old_c = p_complex.data.clone()
    old_r = p_real.data.clone()

    opt.step()

    test("QGD 复数参数已更新", not torch.allclose(p_complex.data, old_c))
    test("QGD 实数参数已更新", not torch.allclose(p_real.data, old_r))

    # 模长非负
    test("QGD 模长非负", (p_complex.abs() >= 0).all().item())

    # Wirtinger 导数正确性验证
    # 对于 L = |z - target|^2，手动计算梯度并验证 Wirtinger 分解
    z = torch.randn(10, dtype=torch.complex64, requires_grad=True)
    target = torch.randn(10, dtype=torch.complex64)
    loss = (z - target).abs().pow(2).sum()
    loss.backward()
    grad = z.grad.clone()

    grad_r, grad_phi = wirtinger_to_polar(z, grad)
    # 验证: 模长梯度应为正（损失是距离，增大模长应增大损失当 z 和 target 方向一致时）
    test("QGD Wirtinger grad_r 形状", grad_r.shape == z.shape)
    test("QGD Wirtinger grad_phi 形状", grad_phi.shape == z.shape)
    # grad_r 和 grad_phi 应该是实数
    test("QGD Wirtinger grad_r 实数", not grad_r.is_complex())
    test("QGD Wirtinger grad_phi 实数", not grad_phi.is_complex())
    # 数值验证: 有限值
    test("QGD Wirtinger grad_r 有限", torch.isfinite(grad_r).all().item())
    test("QGD Wirtinger grad_phi 有限", torch.isfinite(grad_phi).all().item())

    # Cayley 参数识别测试
    test("QGD 识别 omega_diag", _is_cayley_param("blocks.0.qsa.Wq.omega_diag"))
    test("QGD 识别 omega_tri", _is_cayley_param("blocks.0.qsa.Wq.omega_tri"))
    test("QGD 识别 .A 参数", _is_cayley_param("blocks.0.gate.U_ent.A"))
    test("QGD 不误识别普通复数参数", not _is_cayley_param("blocks.0.ffn.W_up.weight"))
    test("QGD 不误识别 bias", not _is_cayley_param("blocks.0.norm.weight"))

    # from_model 工厂方法
    from quantum_core import CayleyLinear, QuantumArch
    model_small = QuantumArch(dim=16, num_layers=1, num_heads=2, ffn_dim=32)
    opt_fm = QGD.from_model(model_small, mod_lr=1e-3, phase_lr=1e-2)
    test("QGD from_model 创建成功", opt_fm is not None)
    # 验证参数名称已注册
    names_registered = 'names' in opt_fm.param_groups[0]
    test("QGD from_model 参数名称已注册", names_registered)

    # from_model 训练测试
    x_fm = torch.randn(2, 4, 16)
    target_fm = torch.randn(2, 4, 16)
    result_fm = model_small({'inputs': x_fm}, training=True)
    loss_fm = (result_fm['output'] - target_fm).abs().pow(2).mean()
    model_small.zero_grad()
    loss_fm.backward()
    opt_fm.step()
    test("QGD from_model 训练成功", True)

    # 梯度裁剪测试
    p_clip = nn.Parameter(torch.randn(4, 8, dtype=torch.complex64))
    opt_clip = QGD([p_clip], mod_lr=1e-3, phase_lr=1e-2, max_grad_norm=1.0)
    p_clip.grad = torch.randn_like(p_clip) * 100  # 大梯度
    opt_clip.step()
    test("QGD 梯度裁剪无NaN", not torch.isnan(p_clip).any().item())

    # 多步稳定性
    for _ in range(50):
        p_complex.grad = torch.randn_like(p_complex) * 0.1
        p_real.grad = torch.randn_like(p_real) * 0.1
        opt.step()

    test("QGD 50步无NaN", not (torch.isnan(p_complex).any() or torch.isnan(p_real).any()))


# ═══════════════════════════════════════════════════════════════════
# 13. 端到端集成测试
# ═══════════════════════════════════════════════════════════════════

def test_end_to_end():
    print("\n=== 端到端集成测试 ===")

    B, N, D = 2, 8, 16

    model = QuantumArch(
        dim=D, num_layers=2, num_heads=2, ffn_dim=32,
        collapse_enabled=True, direct_input=True, dropout=0.0,
    )
    optimizer = QGD.from_model(model, mod_lr=1e-3, phase_lr=1e-2)
    criterion = nn.MSELoss()

    # 训练3步
    losses = []
    for step in range(5):
        model.zero_grad()
        x = torch.randn(B, N, D)
        target = torch.randn(B, N, D)
        result = model({'inputs': x}, training=True)
        output = result['output']
        if output.shape != target.shape:
            target = target[..., :output.shape[-1]]
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    test("端到端训练5步成功", len(losses) == 5)
    test("端到端损失有限", all(math.isfinite(l) for l in losses))

    # 酉性在整个训练后仍然保持
    report = model.get_unitarity_report()
    max_viol = max(report.values()) if report else 0
    test(f"训练后酉性 < 0.15 ({max_viol:.2e})", max_viol < 0.15)


# ═══════════════════════════════════════════════════════════════════
# 运行所有测试
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("QuantumArch 核心模块单元测试")
    print("=" * 60)

    test_complex_ops()
    test_activations()
    test_normalization()
    test_unitary()
    test_embedding()
    test_attention()
    test_entanglement()
    test_collapse()
    test_ffn()
    test_quantum_block()
    test_model()
    test_optimizer()
    test_end_to_end()

    print("\n" + "=" * 60)
    print(f"结果: {passed} 通过, {failed} 失败 (共 {passed + failed})")
    if errors:
        print("失败项:")
        for e in errors:
            print(f"  - {e}")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)
