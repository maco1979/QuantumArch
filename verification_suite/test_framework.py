"""
技术验证测试框架
用于验证量子架构工程化挑战的解决方案
"""

import torch
import torch.nn as nn
import numpy as np
import time
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """测试结果数据类"""
    test_name: str
    passed: bool
    metric_value: float
    expected_value: float
    tolerance: float
    details: str = ""


class ComplexArithmeticValidator:
    """复数运算验证器"""

    def __init__(self):
        self.results: List[TestResult] = []

    def test_complex_multiplication_speed(self) -> TestResult:
        """测试复数矩阵乘法速度"""
        batch, seq, dim = 64, 256, 512

        # 实数版本
        x_real = torch.randn(batch, seq, dim, dtype=torch.float32).cuda()
        y_real = torch.randn(batch, seq, dim, dtype=torch.float32).cuda()

        # 复数版本
        x_complex = torch.randn(batch, seq, dim, dtype=torch.cfloat).cuda()
        y_complex = torch.randn(batch, seq, dim, dtype=torch.cfloat).cuda()

        # 预热
        for _ in range(10):
            _ = torch.matmul(x_real, y_real.transpose(-2, -1))
            _ = torch.matmul(x_complex, y_complex.transpose(-2, -1))
        torch.cuda.synchronize()

        # 测量实数
        start = time.time()
        for _ in range(100):
            _ = torch.matmul(x_real, y_real.transpose(-2, -1))
        torch.cuda.synchronize()
        real_time = (time.time() - start) / 100

        # 测量复数
        start = time.time()
        for _ in range(100):
            _ = torch.matmul(x_complex, y_complex.transpose(-2, -1))
        torch.cuda.synchronize()
        complex_time = (time.time() - start) / 100

        ratio = complex_time / real_time

        logger.info(f"复数/实数矩阵乘法时间比: {ratio:.2f}×")

        # 验收标准：复数开销 < 3×
        result = TestResult(
            test_name="complex_multiplication_speed",
            passed=ratio < 3.0,
            metric_value=ratio,
            expected_value=3.0,
            tolerance=0.0,
            details=f"实数={real_time*1000:.2f}ms, 复数={complex_time*1000:.2f}ms"
        )

        self.results.append(result)
        return result

    def test_modrelu_gradient(self) -> TestResult:
        """测试ModReLU梯度正确性"""
        def modrelu(z, bias=0.0):
            """ModReLU激活函数"""
            abs_z = torch.abs(z)
            scale = torch.relu(abs_z + bias) / (abs_z + 1e-8)
            return z * scale

        z = torch.tensor([1.0 + 2.0j, 3.0 + 4.0j], requires_grad=True)

        # 前向传播
        output = modrelu(z)
        loss = output.abs().sum()

        # 反向传播
        loss.backward()

        # 验证梯度存在且数值合理
        has_gradient = z.grad is not None
        gradient_finite = torch.isfinite(z.grad).all()

        logger.info(f"ModReLU梯度: {z.grad}")
        logger.info(f"梯度存在: {has_gradient}, 梯度有限: {gradient_finite}")

        result = TestResult(
            test_name="modrelu_gradient",
            passed=has_gradient and gradient_finite,
            metric_value=1.0 if (has_gradient and gradient_finite) else 0.0,
            expected_value=1.0,
            tolerance=0.0,
            details=f"梯度={z.grad if has_gradient else 'None'}"
        )

        self.results.append(result)
        return result

    def test_unitary_constraint(self) -> TestResult:
        """测试酉约束是否保持"""
        def cayley_transform(omega):
            """Cayley变换：将厄米矩阵转为酉矩阵"""
            half_i_omega = 0.5j * omega
            I = torch.eye(omega.shape[0], dtype=torch.complex128)
            return torch.linalg.solve(I + half_i_omega, I - half_i_omega)

        # 创建随机厄米矩阵
        d = 16
        A = torch.randn(d, d, dtype=torch.complex128)
        omega = (A + A.conj().T) / 2  # 厄米矩阵

        # 转换为酉矩阵
        U = cayley_transform(omega)

        # 验证酉性：U^† U = I
        U_dagger_U = U.conj().T @ U
        violation = torch.norm(U_dagger_U - torch.eye(d, dtype=torch.complex128)).item()

        logger.info(f"酉约束违背度: {violation:.2e}")

        # 验收标准：违背度 < 1e-6
        result = TestResult(
            test_name="unitary_constraint",
            passed=violation < 1e-6,
            metric_value=violation,
            expected_value=1e-6,
            tolerance=0.0,
            details=f"||U†U - I|| = {violation:.2e}"
        )

        self.results.append(result)
        return result

    def test_gradient_flow_conservation(self) -> TestResult:
        """测试梯度流守恒"""
        # 模拟简单的酉变换网络
        class SimpleUnitaryNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.W = nn.Parameter(torch.randn(64, 64, dtype=torch.cfloat))

            def forward(self, x):
                # 投影到酉空间
                omega = (self.W + self.W.conj().T) / 2
                half_i_omega = 0.5j * omega
                I = torch.eye(64, dtype=torch.complex128)
                U = torch.linalg.solve(I + half_i_omega, I - half_i_omega)
                return U @ x

        model = SimpleUnitaryNet()
        x = torch.randn(32, 64, dtype=torch.cfloat)
        y = torch.randn(32, 64, dtype=torch.cfloat)

        # 计算损失
        output = model(x)
        loss = (output - y).abs().sum()

        # 反向传播
        loss.backward()

        # 检查输入梯度范数 <= 输出梯度范数
        if x.grad is not None:
            input_grad_norm = x.grad.norm().item()
            # 输出梯度（通过损失链式法则）
            output_grad_norm = (output - y).norm().item()

            is_conserved = input_grad_norm <= output_grad_norm * (1.0 + 0.1)  # 允许10%误差

            logger.info(f"输入梯度范数: {input_grad_norm:.4f}")
            logger.info(f"输出梯度范数: {output_grad_norm:.4f}")
            logger.info(f"梯度流守恒: {is_conserved}")

            result = TestResult(
                test_name="gradient_flow_conservation",
                passed=is_conserved,
                metric_value=input_grad_norm,
                expected_value=output_grad_norm,
                tolerance=0.1 * output_grad_norm,
                details=f"input={input_grad_norm:.4f}, output={output_grad_norm:.4f}"
            )

            self.results.append(result)
            return result
        else:
            logger.error("输入梯度为None")
            result = TestResult(
                test_name="gradient_flow_conservation",
                passed=False,
                metric_value=0.0,
                expected_value=1.0,
                tolerance=0.0,
                details="输入梯度为None"
            )
            self.results.append(result)
            return result


class HardwareUtilizationValidator:
    """硬件利用验证器"""

    def __init__(self):
        self.results: List[TestResult] = []

    def test_gpu_utilization(self) -> TestResult:
        """测试GPU利用率"""
        if not torch.cuda.is_available():
            logger.warning("CUDA不可用，跳过GPU利用率测试")
            result = TestResult(
                test_name="gpu_utilization",
                passed=False,
                metric_value=0.0,
                expected_value=60.0,
                tolerance=0.0,
                details="CUDA不可用"
            )
            self.results.append(result)
            return result

        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)

            # 执行计算任务
            x = torch.randn(1024, 1024, 1024, dtype=torch.cfloat).cuda()
            for _ in range(10):
                _ = torch.matmul(x, x.transpose(-2, -1))

            # 获取GPU利用率
            info = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = info.gpu

            logger.info(f"GPU利用率: {gpu_util}%")

            # 验收标准：利用率 > 60%
            result = TestResult(
                test_name="gpu_utilization",
                passed=gpu_util > 60,
                metric_value=float(gpu_util),
                expected_value=60.0,
                tolerance=0.0,
                details=f"GPU利用率={gpu_util}%"
            )

            self.results.append(result)
            return result

        except ImportError:
            logger.warning("pynvml未安装，无法获取GPU利用率")
            result = TestResult(
                test_name="gpu_utilization",
                passed=False,
                metric_value=0.0,
                expected_value=60.0,
                tolerance=0.0,
                details="pynvml未安装"
            )
            self.results.append(result)
            return result

    def test_memory_efficiency(self) -> TestResult:
        """测试内存效率"""
        if not torch.cuda.is_available():
            result = TestResult(
                test_name="memory_efficiency",
                passed=False,
                metric_value=0.0,
                expected_value=1.0,
                tolerance=0.0,
                details="CUDA不可用"
            )
            self.results.append(result)
            return result

        # 清空缓存
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # 创建复数张量
        batch, seq, dim = 32, 128, 512
        z = torch.randn(batch, seq, dim, dtype=torch.cfloat).cuda()

        # 计算预期内存
        expected_bytes = batch * seq * dim * 2 * 4  # 2个浮点数(实+虚) * 4字节

        # 获取实际内存
        allocated = torch.cuda.memory_allocated()

        # 计算效率（实际/预期）
        efficiency = expected_bytes / allocated

        logger.info(f"预期内存: {expected_bytes/1024/1024:.2f}MB")
        logger.info(f"实际内存: {allocated/1024/1024:.2f}MB")
        logger.info(f"内存效率: {efficiency:.2%}")

        # 验收标准：效率 > 50%（考虑开销）
        result = TestResult(
            test_name="memory_efficiency",
            passed=efficiency > 0.5,
            metric_value=efficiency,
            expected_value=0.5,
            tolerance=0.0,
            details=f"效率={efficiency:.2%}"
        )

        self.results.append(result)
        return result


class TrainingStabilityValidator:
    """训练稳定性验证器"""

    def __init__(self):
        self.results: List[TestResult] = []

    def test_loss_convergence(self) -> TestResult:
        """测试损失收敛性"""
        # 模拟简单的优化过程
        x = torch.randn(100, 10, dtype=torch.cfloat, requires_grad=True)
        y = torch.randn(100, 10, dtype=torch.cfloat)

        optimizer = torch.optim.Adam([x], lr=1e-3)

        loss_history = []
        for step in range(100):
            optimizer.zero_grad()

            # 简单损失：MSE
            loss = torch.mean((x - y).abs() ** 2)
            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())

        # 检查单调递减（允许小波动）
        is_decreasing = all(
            loss_history[i] >= loss_history[i+1] - 0.01
            for i in range(len(loss_history) - 1)
        )

        # 最终损失显著低于初始损失
        initial_loss = loss_history[0]
        final_loss = loss_history[-1]
        improvement = (initial_loss - final_loss) / initial_loss

        logger.info(f"初始损失: {initial_loss:.4f}")
        logger.info(f"最终损失: {final_loss:.4f}")
        logger.info(f"改进率: {improvement:.2%}")
        logger.info(f"单调递减: {is_decreasing}")

        # 验收标准：改进率 > 50%
        result = TestResult(
            test_name="loss_convergence",
            passed=improvement > 0.5 and is_decreasing,
            metric_value=improvement,
            expected_value=0.5,
            tolerance=0.0,
            details=f"改进率={improvement:.2%}, 单调={is_decreasing}"
        )

        self.results.append(result)
        return result

    def test_gradient_explosion_detection(self) -> TestResult:
        """测试梯度爆炸检测"""
        # 模拟梯度爆炸
        x = torch.randn(10, 10, dtype=torch.cfloat, requires_grad=True)
        y = torch.randn(10, 10, dtype=torch.cfloat)

        # 使用过大的学习率触发梯度爆炸
        optimizer = torch.optim.Adam([x], lr=1.0)

        detected = False
        for step in range(10):
            optimizer.zero_grad()

            loss = torch.mean((x - y).abs() ** 2)
            loss.backward()

            # 检查梯度范数
            if x.grad is not None:
                grad_norm = x.grad.norm().item()
                if grad_norm > 10.0:  # 梯度爆炸阈值
                    detected = True
                    logger.warning(f"梯度爆炸检测: {grad_norm:.2f}")
                    break

            optimizer.step()

        # 验收标准：能够检测到梯度爆炸
        result = TestResult(
            test_name="gradient_explosion_detection",
            passed=detected,
            metric_value=1.0 if detected else 0.0,
            expected_value=1.0,
            tolerance=0.0,
            details=f"检测到={detected}"
        )

        self.results.append(result)
        return result


class TheoryValidationValidator:
    """理论验证器"""

    def __init__(self):
        self.results: List[TestResult] = []

    def test_born_rule(self) -> TestResult:
        """测试Born法则实现"""
        # 量子态
        psi = torch.tensor([0.6 + 0.2j, 0.5 + 0.4j, 0.3 + 0.6j])
        psi = psi / psi.norm()  # 归一化

        # Born概率：|α_i|²
        probabilities = torch.abs(psi) ** 2

        # 验证概率和为1
        sum_prob = probabilities.sum().item()

        logger.info(f"概率和: {sum_prob:.6f}")
        logger.info(f"概率分布: {probabilities}")

        # 验收标准：概率和 = 1 ± 1e-6
        result = TestResult(
            test_name="born_rule",
            passed=abs(sum_prob - 1.0) < 1e-6,
            metric_value=sum_prob,
            expected_value=1.0,
            tolerance=1e-6,
            details=f"概率和={sum_prob:.6f}"
        )

        self.results.append(result)
        return result

    def test_interference_effect(self) -> TestResult:
        """测试干涉效应"""
        # 两个波函数
        psi1 = torch.tensor([1.0 + 0.0j, 0.0 + 0.0j])
        psi2 = torch.tensor([0.0 + 1.0j, 0.0 + 0.0j])

        # 线性叠加
        psi_sum = psi1 + psi2

        # 概率（应该显示干涉）
        prob_sum = torch.abs(psi_sum) ** 2
        prob_1 = torch.abs(psi1) ** 2
        prob_2 = torch.abs(psi2) ** 2

        # 干涉项
        interference = (prob_sum - (prob_1 + prob_2)).abs().sum().item()

        logger.info(f"干涉效应强度: {interference:.6f}")

        # 验收标准：干涉效应显著（> 0.5）
        result = TestResult(
            test_name="interference_effect",
            passed=interference > 0.5,
            metric_value=interference,
            expected_value=0.5,
            tolerance=0.0,
            details=f"干涉强度={interference:.6f}"
        )

        self.results.append(result)
        return result


class VerificationSuite:
    """完整验证套件"""

    def __init__(self):
        self.complex_validator = ComplexArithmeticValidator()
        self.hardware_validator = HardwareUtilizationValidator()
        self.training_validator = TrainingStabilityValidator()
        self.theory_validator = TheoryValidationValidator()

    def run_all_tests(self) -> Dict[str, Any]:
        """运行所有测试"""
        logger.info("=" * 80)
        logger.info("开始量子架构技术验证测试")
        logger.info("=" * 80)

        # 复数运算测试
        logger.info("\n【复数运算验证】")
        self.complex_validator.test_complex_multiplication_speed()
        self.complex_validator.test_modrelu_gradient()
        self.complex_validator.test_unitary_constraint()
        self.complex_validator.test_gradient_flow_conservation()

        # 硬件利用测试
        logger.info("\n【硬件利用验证】")
        self.hardware_validator.test_gpu_utilization()
        self.hardware_validator.test_memory_efficiency()

        # 训练稳定性测试
        logger.info("\n【训练稳定性验证】")
        self.training_validator.test_loss_convergence()
        self.training_validator.test_gradient_explosion_detection()

        # 理论验证测试
        logger.info("\n【理论验证】")
        self.theory_validator.test_born_rule()
        self.theory_validator.test_interference_effect()

        # 生成报告
        return self.generate_report()

    def generate_report(self) -> Dict[str, Any]:
        """生成测试报告"""
        all_results = (
            self.complex_validator.results +
            self.hardware_validator.results +
            self.training_validator.results +
            self.theory_validator.results
        )

        passed = sum(1 for r in all_results if r.passed)
        total = len(all_results)
        pass_rate = passed / total

        logger.info("\n" + "=" * 80)
        logger.info("测试报告摘要")
        logger.info("=" * 80)
        logger.info(f"总测试数: {total}")
        logger.info(f"通过数: {passed}")
        logger.info(f"失败数: {total - passed}")
        logger.info(f"通过率: {pass_rate:.2%}")
        logger.info("=" * 80)

        # 详细结果
        logger.info("\n详细结果:")
        for result in all_results:
            status = "✓ 通过" if result.passed else "✗ 失败"
            logger.info(f"  {result.test_name}: {status}")
            logger.info(f"    {result.details}")
            if not result.passed:
                logger.info(f"    期望: {result.expected_value}, 实际: {result.metric_value}")

        return {
            'total': total,
            'passed': passed,
            'failed': total - passed,
            'pass_rate': pass_rate,
            'results': all_results
        }


def main():
    """主函数"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 运行验证套件
    suite = VerificationSuite()
    report = suite.run_all_tests()

    # 保存报告
    output_dir = Path('./verification_results')
    output_dir.mkdir(exist_ok=True)

    import json
    with open(output_dir / 'test_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=lambda x: float(x) if isinstance(x, (np.integer, np.floating)) else x.__dict__)

    logger.info(f"\n测试报告已保存: {output_dir / 'test_report.json'}")

    # 返回退出码
    exit_code = 0 if report['pass_rate'] >= 0.8 else 1
    return exit_code


if __name__ == '__main__':
    exit(main())
