"""
QGD 量子梯度下降优化器

在复数希尔伯特空间中优化，核心特性：
1. 模长/相位分离更新：使用不同学习率
2. 酉约束保持：通过 Cayley 参数化自动满足
3. 相位动量：独立的相位空间动量

基于 PyTorch 优化器接口实现，可与标准训练管线无缝集成。
"""

import torch
from torch.optim import Optimizer
from typing import Dict, List, Optional, Tuple
import math


class QGD(Optimizer):
    """量子梯度下降优化器。

    对于复数参数 z = |z| * exp(i*φ)：
    - 模长更新：|z|_{t+1} = |z|_t - η_mod * grad_mod
    - 相位更新：φ_{t+1} = φ_t - η_phase * grad_phase

    对于 Cayley 参数化的酉矩阵：
    - 直接优化 Ω 参数，酉性由 Cayley 变换自动保持

    Args:
        params: 可迭代参数
        mod_lr: 模长学习率
        phase_lr: 相位学习率（通常 > mod_lr）
        betas: Adam 风格的动量参数
        eps: 数值稳定 epsilon
        weight_decay: 权重衰减
    """

    def __init__(
        self,
        params,
        mod_lr: float = 1e-4,
        phase_lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        defaults = dict(
            mod_lr=mod_lr,
            phase_lr=phase_lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """执行一步优化更新。"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            mod_lr = group['mod_lr']
            phase_lr = group['phase_lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                # 初始化状态
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    # 模长 Adam 状态
                    state['mod_exp_avg'] = torch.zeros_like(p, dtype=torch.float32)
                    state['mod_exp_avg_sq'] = torch.zeros_like(p, dtype=torch.float32)
                    # 相位 Adam 状态
                    state['phase_exp_avg'] = torch.zeros_like(p, dtype=torch.float32)
                    state['phase_exp_avg_sq'] = torch.zeros_like(p, dtype=torch.float32)

                state['step'] += 1
                t = state['step']

                is_complex = p.dtype in (torch.complex64, torch.complex128)

                if is_complex:
                    # ── 复数参数：模长/相位分离更新 ──
                    magnitude = p.abs().float()
                    phase = p.angle().float()

                    # 梯度分解为模长和相位分量
                    # dL/d|z| ≈ Re(conj(z) * grad) / (|z| + eps)
                    # dL/dφ   ≈ -Im(conj(z) * grad) / (|z| + eps) * |z|
                    safe_mag = magnitude + eps
                    conj_z_grad = (p.conj() * grad)

                    grad_mod = conj_z_grad.real / safe_mag
                    grad_phase = -conj_z_grad.imag  # 相位梯度

                    # 权重衰减
                    if weight_decay > 0:
                        grad_mod = grad_mod + weight_decay * magnitude

                    # 模长 Adam 更新
                    state['mod_exp_avg'].mul_(beta1).add_(grad_mod, alpha=1 - beta1)
                    state['mod_exp_avg_sq'].mul_(beta2).addcmul_(grad_mod, grad_mod, value=1 - beta2)

                    mod_bias_corr1 = 1 - beta1 ** t
                    mod_bias_corr2 = 1 - beta2 ** t
                    mod_denom = (state['mod_exp_avg_sq'].sqrt() / math.sqrt(mod_bias_corr2)) + eps
                    mod_step = (state['mod_exp_avg'] / mod_bias_corr1) / mod_denom

                    # 相位 Adam 更新
                    state['phase_exp_avg'].mul_(beta1).add_(grad_phase, alpha=1 - beta1)
                    state['phase_exp_avg_sq'].mul_(beta2).addcmul_(grad_phase, grad_phase, value=1 - beta2)

                    phase_bias_corr1 = 1 - beta1 ** t
                    phase_bias_corr2 = 1 - beta2 ** t
                    phase_denom = (state['phase_exp_avg_sq'].sqrt() / math.sqrt(phase_bias_corr2)) + eps
                    phase_step = (state['phase_exp_avg'] / phase_bias_corr1) / phase_denom

                    # 应用更新
                    new_mag = magnitude - mod_lr * mod_step
                    new_phase = phase - phase_lr * phase_step

                    # 合成复数参数
                    p.copy_(torch.polar(new_mag.clamp(min=0), new_phase))

                else:
                    # ── 实数参数：标准 Adam 更新 ──
                    grad_f = grad.float()

                    if weight_decay > 0:
                        grad_f = grad_f + weight_decay * p

                    state['mod_exp_avg'].mul_(beta1).add_(grad_f, alpha=1 - beta1)
                    state['mod_exp_avg_sq'].mul_(beta2).addcmul_(grad_f, grad_f, value=1 - beta2)

                    bias_corr1 = 1 - beta1 ** t
                    bias_corr2 = 1 - beta2 ** t
                    denom = (state['mod_exp_avg_sq'].sqrt() / math.sqrt(bias_corr2)) + eps
                    step = (state['mod_exp_avg'] / bias_corr1) / denom

                    p.copy_(p.float() - mod_lr * step)

        return loss
