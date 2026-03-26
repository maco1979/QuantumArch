"""
QGD 量子梯度下降优化器 (Quantum Gradient Descent)

在复数希尔伯特空间中优化，核心特性：
1. Wirtinger 导数：正确的复数梯度分解（模长 + 相位）
2. 模长/相位分离更新：使用不同学习率
3. Cayley 参数特殊处理：自动识别 Cayley 参数，使用标准实数 Adam
4. 相位动量：独立的相位空间动量
5. 梯度裁剪：防止复数梯度爆炸

理论基础：
    复数参数 z = r·e^{iφ} 的损失函数 L(z, z̄) 使用 Wirtinger 导数：
        ∂L/∂z̄ = (∂L/∂Re(z) + i·∂L/∂Im(z)) / 2

    分解到极坐标：
        ∂L/∂r = Re(z̄ · ∂L/∂z̄) / r
        ∂L/∂φ = Re(-i·z̄ · ∂L/∂z̄) / r = Im(z̄ · ∂L/∂z̄) / r

    对于 Cayley 参数化的酉矩阵 W = Cayley(Ω)：
        不直接优化 W，而是优化 Ω 参数（实数/复数参数）
        酉性由 Cayley 变换自动保持，无需额外约束
"""

import torch
from torch.optim import Optimizer
from typing import Dict, List, Optional, Tuple, Set
import math
import re


# ──────────────────────────────────────────────
# 参数名模式匹配（Cayley 参数自动识别）
# ──────────────────────────────────────────────

# Cayley 参数名称模式
_CAYLEY_PATTERNS = [
    r"omega_diag",  # CayleyLinear 的对角虚部参数（实数）
    r"omega_tri",  # CayleyLinear 的上三角参数（复数）
    r"\.A$",  # CayleyLinearSimple 的一般矩阵参数（复数）
    r"_cayley_",  # 通用 Cayley 相关参数
]

# QGD 内部酉耦合参数（需要 Cayley 处理）
_UNITARY_PARAM_PATTERNS = [
    r"phase$",  # UnitaryCoupling diagonal 模式的相位参数
    r"mix$",  # UnitaryCoupling diagonal 模式的混合参数
]


def _is_cayley_param(name: str) -> bool:
    """判断参数名是否属于 Cayley 参数化模块的底层参数。

    Cayley 参数（如 omega_diag, omega_tri, .A）应该用标准 Adam 更新，
    而非 Wirtinger 极坐标分解（因为它们是 Cayley 变换的输入参数，不是酉矩阵本身）。

    Args:
        name: 参数全名（如 'blocks.0.qsa.Wq.omega_diag'）
    Returns:
        是否为 Cayley 底层参数
    """
    for pattern in _CAYLEY_PATTERNS:
        if re.search(pattern, name):
            return True
    return False


def _is_unitary_param(name: str) -> bool:
    """判断参数是否属于酉变换模块。

    Args:
        name: 参数全名
    Returns:
        是否为酉变换参数
    """
    return _is_cayley_param(name) or any(re.search(p, name) for p in _UNITARY_PARAM_PATTERNS)


# ──────────────────────────────────────────────
# Wirtinger 导数工具
# ──────────────────────────────────────────────


def wirtinger_to_polar(
    z: torch.Tensor,
    grad: torch.Tensor,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """将 Wirtinger 导数（共轭梯度）分解为极坐标分量。

    对于 z = r·e^{iφ}，损失 L 对 (r, φ) 的梯度：

        ∂L/∂r = Re(z̄ · g) / r
        ∂L/∂φ = Im(z̄ · g) / r

    其中 g = ∂L/∂z̄ 是 Wirtinger 导数（PyTorch 中 p.grad 的值）。

    数学推导：
        z = r·e^{iφ}, z̄ = r·e^{-iφ}
        ∂L/∂r = Re(∂L/∂z · ∂z/∂r + ∂L/∂z̄ · ∂z̄/∂r)
               = Re(∂L/∂z · e^{iφ} + ∂L/∂z̄ · e^{-iφ})
               = 2·Re(z̄ · ∂L/∂z̄) / r  （对于实值 L，∂L/∂z = conj(∂L/∂z̄)）

        ∂L/∂φ = Re(-i·z̄ · ∂L/∂z̄) / r = Im(z̄ · ∂L/∂z̄) / r

    Args:
        z: 复数参数 (shape)
        grad: Wirtinger 导数 g = ∂L/∂z̄ (shape)
        eps: 数值稳定 epsilon
    Returns:
        (grad_r, grad_phi): 模长梯度和相位梯度，均为实数 (shape)
    """
    r = z.abs().float().clamp(min=eps)
    # z̄ · g 的实部和虚部
    conj_z_grad = z.conj() * grad
    grad_r = conj_z_grad.real / r
    grad_phi = conj_z_grad.imag / r
    return grad_r, grad_phi


# ──────────────────────────────────────────────
# 主优化器
# ──────────────────────────────────────────────


class QGD(Optimizer):
    """量子梯度下降优化器。

    对不同类型的参数使用不同的更新策略：

    1. **普通复数参数**（Wirtinger 模式）：
       - 将 Wirtinger 导数分解为模长梯度和相位梯度
       - 使用 Adam 风格的模长/相位分离更新
       - 模长和相位可以有不同的学习率

    2. **Cayley 参数**（标准 Adam 模式）：
       - 自动识别 omega_diag, omega_tri, .A 等 Cayley 底层参数
       - 使用标准 Adam 更新（不做 Wirtinger 分解）
       - 酉性由 Cayley 变换自动保持

    3. **实数参数**（标准 Adam 模式）：
       - 使用标准实数 Adam 更新
       - 与复数参数相同的动量配置

    Args:
        params: 可迭代参数（或参数组）
        mod_lr: 模长学习率（普通复数参数）
        phase_lr: 相位学习率（普通复数参数，通常 > mod_lr）
        cayley_lr: Cayley 参数学习率（默认 = mod_lr）
        real_lr: 实数参数学习率（默认 = mod_lr）
        betas: Adam 风格的动量参数 (beta1, beta2)
        eps: 数值稳定 epsilon
        weight_decay: 权重衰减（对复数使用 |z|² 正则化）
        max_grad_norm: 梯度裁剪阈值（None 禁用）
        amsgrad: 是否使用 AMSGrad 变体
    """

    def __init__(
        self,
        params,
        mod_lr: float = 1e-4,
        phase_lr: float = 1e-3,
        cayley_lr: Optional[float] = None,
        real_lr: Optional[float] = None,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        max_grad_norm: Optional[float] = None,
        amsgrad: bool = False,
    ):
        if cayley_lr is None:
            cayley_lr = mod_lr
        if real_lr is None:
            real_lr = mod_lr

        defaults = dict(
            mod_lr=mod_lr,
            phase_lr=phase_lr,
            cayley_lr=cayley_lr,
            real_lr=real_lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            amsgrad=amsgrad,
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
            mod_lr = group["mod_lr"]
            phase_lr = group["phase_lr"]
            cayley_lr = group["cayley_lr"]
            real_lr = group["real_lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            max_grad_norm = group["max_grad_norm"]
            amsgrad = group["amsgrad"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                # 梯度裁剪
                if max_grad_norm is not None:
                    # 对复数梯度使用 view_as_real 计算 norm
                    if grad.is_complex():
                        grad_norm = torch.view_as_real(grad).float().norm()
                    else:
                        grad_norm = grad.float().norm()
                    if grad_norm > max_grad_norm:
                        scale = max_grad_norm / grad_norm
                        grad = grad * scale

                # 初始化状态
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    # Adam 状态（模长/相位/标准 各一套）
                    state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
                    if amsgrad:
                        state["max_exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
                    # 第二套 Adam 状态（相位专用）
                    state["phase_exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                    state["phase_exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
                    if amsgrad:
                        state["phase_max_exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)

                state["step"] += 1
                t = state["step"]

                is_complex = p.dtype in (torch.complex64, torch.complex128)
                is_cayley = _is_cayley_param(self._get_param_name(p))

                if is_cayley or not is_complex:
                    # ── Cayley 参数 / 实数参数：标准 Adam 更新 ──
                    self._adam_step(
                        p,
                        grad,
                        state,
                        t,
                        lr=cayley_lr if is_cayley else real_lr,
                        beta1=beta1,
                        beta2=beta2,
                        eps=eps,
                        weight_decay=weight_decay if not is_complex else 0.0,
                        amsgrad=amsgrad,
                    )
                else:
                    # ── 普通复数参数：Wirtinger 极坐标分离更新 ──
                    self._wirtinger_step(
                        p,
                        grad,
                        state,
                        t,
                        mod_lr=mod_lr,
                        phase_lr=phase_lr,
                        beta1=beta1,
                        beta2=beta2,
                        eps=eps,
                        weight_decay=weight_decay,
                        amsgrad=amsgrad,
                    )

        return loss

    def _wirtinger_step(
        self,
        p: torch.Tensor,
        grad: torch.Tensor,
        state: Dict,
        t: int,
        mod_lr: float,
        phase_lr: float,
        beta1: float,
        beta2: float,
        eps: float,
        weight_decay: float,
        amsgrad: bool,
    ):
        """Wirtinger 极坐标分离更新。

        对复数参数 z = r·e^{iφ}：
        1. 计算 Wirtinger 导数的极坐标分量
        2. 模长用 Adam 更新
        3. 相位用独立 Adam 更新
        4. 合成新的复数参数
        """
        # 1. 极坐标分解
        magnitude = p.abs().float()
        phase = p.angle().float()

        grad_r, grad_phi = wirtinger_to_polar(p, grad, eps=eps)

        # 2. 权重衰减（L2 正则化 on |z|²）
        if weight_decay > 0:
            grad_r = grad_r + weight_decay * magnitude

        # 3. 模长 Adam 更新
        state["exp_avg"].mul_(beta1).add_(grad_r, alpha=1 - beta1)
        state["exp_avg_sq"].mul_(beta2).addcmul_(grad_r, grad_r, value=1 - beta2)

        if amsgrad:
            # 维护最大值
            torch.max(
                state["max_exp_avg_sq"],
                state["exp_avg_sq"] / (1 - beta2**t),
                out=state["max_exp_avg_sq"],
            )
            mod_denom = state["max_exp_avg_sq"].sqrt() + eps
        else:
            bias_corr2 = 1 - beta2**t
            mod_denom = (state["exp_avg_sq"].sqrt() / math.sqrt(bias_corr2)) + eps

        bias_corr1 = 1 - beta1**t
        mod_step = (state["exp_avg"] / bias_corr1) / mod_denom

        # 4. 相位 Adam 更新（独立的动量和方差估计）
        state["phase_exp_avg"].mul_(beta1).add_(grad_phi, alpha=1 - beta1)
        state["phase_exp_avg_sq"].mul_(beta2).addcmul_(grad_phi, grad_phi, value=1 - beta2)

        if amsgrad:
            torch.max(
                state["phase_max_exp_avg_sq"],
                state["phase_exp_avg_sq"] / (1 - beta2**t),
                out=state["phase_max_exp_avg_sq"],
            )
            phase_denom = state["phase_max_exp_avg_sq"].sqrt() + eps
        else:
            bias_corr2_phase = 1 - beta2**t
            phase_denom = (state["phase_exp_avg_sq"].sqrt() / math.sqrt(bias_corr2_phase)) + eps

        bias_corr1_phase = 1 - beta1**t
        phase_step = (state["phase_exp_avg"] / bias_corr1_phase) / phase_denom

        # 5. 应用更新
        new_mag = magnitude - mod_lr * mod_step
        new_phase = phase - phase_lr * phase_step

        # 模长非负约束
        new_mag = new_mag.clamp(min=0)

        # 6. 合成复数参数
        p.copy_(torch.polar(new_mag, new_phase))

    def _adam_step(
        self,
        p: torch.Tensor,
        grad: torch.Tensor,
        state: Dict,
        t: int,
        lr: float,
        beta1: float,
        beta2: float,
        eps: float,
        weight_decay: float,
        amsgrad: bool,
    ):
        """标准 Adam 更新（用于 Cayley 参数和实数参数）。

        对于复数参数（如 Cayley 的 omega_tri）：
        使用 view_as_real 将复数视为 [real, imag] 两通道实数，
        然后逐元素做 Adam。这等价于对实部和虚部独立做 Adam。
        """
        is_complex = p.is_complex()

        if is_complex:
            # 复数参数：view_as_real 后做实数 Adam
            p_real = torch.view_as_real(p.data)  # (..., 2)
            grad_real = torch.view_as_real(grad)  # (..., 2)
            # Adam 状态也需要 view_as_real
            state_exp_avg = state["exp_avg"]
            state_exp_avg_sq = state["exp_avg_sq"]

            # 如果状态维度不匹配（首次处理复数），需要重建
            if state_exp_avg.shape != p_real.shape:
                state["exp_avg"] = torch.zeros_like(p_real, dtype=torch.float32)
                state["exp_avg_sq"] = torch.zeros_like(p_real, dtype=torch.float32)
                if amsgrad:
                    state["max_exp_avg_sq"] = torch.zeros_like(p_real, dtype=torch.float32)
                state_exp_avg = state["exp_avg"]
                state_exp_avg_sq = state["exp_avg_sq"]

            grad_f = grad_real.float()

            if weight_decay > 0:
                grad_f = grad_f + weight_decay * p_real.float()

            state_exp_avg.mul_(beta1).add_(grad_f, alpha=1 - beta1)
            state_exp_avg_sq.mul_(beta2).addcmul_(grad_f, grad_f, value=1 - beta2)

            if amsgrad:
                torch.max(
                    state["max_exp_avg_sq"],
                    state_exp_avg_sq / (1 - beta2**t),
                    out=state["max_exp_avg_sq"],
                )
                denom = state["max_exp_avg_sq"].sqrt() + eps
            else:
                bias_corr2 = 1 - beta2**t
                denom = (state_exp_avg_sq.sqrt() / math.sqrt(bias_corr2)) + eps

            bias_corr1 = 1 - beta1**t
            step = (state_exp_avg / bias_corr1) / denom

            p_real.data.add_(-lr * step)
        else:
            # 实数参数：标准 Adam
            grad_f = grad.float()

            if weight_decay > 0:
                grad_f = grad_f + weight_decay * p.float()

            state["exp_avg"].mul_(beta1).add_(grad_f, alpha=1 - beta1)
            state["exp_avg_sq"].mul_(beta2).addcmul_(grad_f, grad_f, value=1 - beta2)

            if amsgrad:
                torch.max(
                    state["max_exp_avg_sq"],
                    state["exp_avg_sq"] / (1 - beta2**t),
                    out=state["max_exp_avg_sq"],
                )
                denom = state["max_exp_avg_sq"].sqrt() + eps
            else:
                bias_corr2 = 1 - beta2**t
                denom = (state["exp_avg_sq"].sqrt() / math.sqrt(bias_corr2)) + eps

            bias_corr1 = 1 - beta1**t
            step = (state["exp_avg"] / bias_corr1) / denom

            p.copy_(p.float() - lr * step)

    def _get_param_name(self, p: torch.Tensor) -> str:
        """获取参数在模型中的名称。

        通过遍历 param_groups 查找参数对应的名称。
        如果无法确定名称，返回空字符串。
        """
        # 尝试从 state 中缓存的名字获取
        if "_name" in self.state.get(p, {}):
            return self.state[p]["_name"]

        # 在参数组中查找
        for group in self.param_groups:
            if "names" in group:
                for name, param in zip(group["names"], group["params"]):
                    if param is p:
                        self.state[p]["_name"] = name
                        return name

        return ""

    def add_param_group(self, param_group: dict):
        """添加参数组，支持 'names' 字段用于 Cayley 参数识别。

        Args:
            param_group: 参数组字典，可包含：
                - 'params': 参数列表
                - 'names': 对应的参数名列表（可选）
                - 其他优化器参数
        """
        super().add_param_group(param_group)

    def get_stats(self) -> dict:
        """返回当前步数及各参数组的梯度统计信息。

        用于调试和优化系统监控，无需额外计算开销。

        Returns:
            dict 包含:
                - 'global_step': 全局最大步数
                - 'complex_params': 普通复数参数数量
                - 'cayley_params': Cayley 参数数量
                - 'real_params': 实数参数数量
                - 'grad_norm_complex': 复数参数的梯度 L2 范数（最近一步）
                - 'grad_norm_real': 实数参数的梯度 L2 范数（最近一步）
                - 'mod_lr': 当前模长学习率（第一个参数组）
                - 'phase_lr': 当前相位学习率（第一个参数组）
        """
        global_step = 0
        n_complex = n_cayley = n_real = 0
        grad_norm_sq_complex = 0.0
        grad_norm_sq_real = 0.0

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state.get(p, {})
                step = state.get("step", 0)
                global_step = max(global_step, step)

                is_complex = p.dtype in (torch.complex64, torch.complex128)
                is_cayley = _is_cayley_param(self._get_param_name(p))

                if is_cayley:
                    n_cayley += 1
                elif is_complex:
                    n_complex += 1
                    grad_norm_sq_complex += torch.view_as_real(p.grad).float().pow(2).sum().item()
                else:
                    n_real += 1
                    grad_norm_sq_real += p.grad.float().pow(2).sum().item()

        # 读取第一个参数组的当前学习率
        first_group = self.param_groups[0] if self.param_groups else {}

        return {
            "global_step": global_step,
            "complex_params": n_complex,
            "cayley_params": n_cayley,
            "real_params": n_real,
            "grad_norm_complex": float(grad_norm_sq_complex**0.5),
            "grad_norm_real": float(grad_norm_sq_real**0.5),
            "mod_lr": first_group.get("mod_lr", 0.0),
            "phase_lr": first_group.get("phase_lr", 0.0),
        }

    def grad_norm_ema(self, alpha: float = 0.99) -> dict:
        """计算梯度范数的指数移动平均（EMA），用于稳定性监控。

        EMA 平滑可以过滤单步梯度波动，揭示训练趋势：
        - EMA 持续增大 → 训练可能不稳定，考虑降低学习率
        - EMA 趋近于零 → 训练收敛，可考虑降低学习率或停止

        使用方法：
            >>> stats = optimizer.grad_norm_ema()
            >>> print(f"复数梯度 EMA: {stats['ema_complex']:.4f}")

        Args:
            alpha: EMA 衰减系数（0.99 表示 100 步窗口）
        Returns:
            dict with 'ema_complex', 'ema_real', 'ema_total'
        """
        # 从内部 state 获取或初始化 EMA 缓存
        if not hasattr(self, "_grad_ema"):
            self._grad_ema = {"ema_complex": 0.0, "ema_real": 0.0, "count": 0}

        stats = self.get_stats()
        norm_c = stats["grad_norm_complex"]
        norm_r = stats["grad_norm_real"]

        # EMA 更新（偏差校正版本）
        self._grad_ema["count"] += 1
        n = self._grad_ema["count"]
        effective_alpha = alpha if n > 1 else 0.0  # 第一步无历史，直接赋值

        self._grad_ema["ema_complex"] = (
            effective_alpha * self._grad_ema["ema_complex"] + (1 - effective_alpha) * norm_c
        )
        self._grad_ema["ema_real"] = (
            effective_alpha * self._grad_ema["ema_real"] + (1 - effective_alpha) * norm_r
        )

        # 偏差校正
        bias_corr = 1.0 - alpha**n
        ema_c = self._grad_ema["ema_complex"] / max(bias_corr, 1e-8)
        ema_r = self._grad_ema["ema_real"] / max(bias_corr, 1e-8)

        return {
            "ema_complex": ema_c,
            "ema_real": ema_r,
            "ema_total": (ema_c**2 + ema_r**2) ** 0.5,
            "step_count": n,
        }

    @classmethod
    def from_model(
        cls,
        model: torch.nn.Module,
        mod_lr: float = 1e-4,
        phase_lr: float = 1e-3,
        cayley_lr: Optional[float] = None,
        real_lr: Optional[float] = None,
        **kwargs,
    ) -> "QGD":
        """从模型创建优化器，自动识别参数类型。

        自动将模型的命名参数分组，识别 Cayley 参数和普通复数参数。

        Args:
            model: PyTorch 模型
            mod_lr: 模长学习率
            phase_lr: 相位学习率
            cayley_lr: Cayley 参数学习率
            real_lr: 实数参数学习率
            **kwargs: 其他优化器参数
        Returns:
            QGD 优化器实例
        """
        if cayley_lr is None:
            cayley_lr = mod_lr
        if real_lr is None:
            real_lr = mod_lr

        # 收集所有参数及其名称
        named_params = list(model.named_parameters())
        all_names = [name for name, _ in named_params]
        all_params = [param for _, param in named_params]

        optimizer = cls(
            all_params,
            mod_lr=mod_lr,
            phase_lr=phase_lr,
            cayley_lr=cayley_lr,
            real_lr=real_lr,
            **kwargs,
        )

        # 注册参数名称用于 Cayley 识别
        if all_names:
            optimizer.param_groups[0]["names"] = all_names

        return optimizer

    def per_group_lr_state(self) -> List[Dict]:
        """返回每个参数组的学习率状态摘要。

        当使用 ``add_param_group`` 配置多参数组时（例如按层分组，
        Embedding 层使用小学习率，深层使用大学习率），此方法提供
        每个参数组的当前 mod_lr / phase_lr 及统计信息，方便
        独立调度器对不同参数组分别降学习率。

        典型用法：
            >>> for i, group_state in enumerate(optimizer.per_group_lr_state()):
            ...     print(f"Group {i}: mod_lr={group_state['mod_lr']:.2e}, "
            ...           f"params={group_state['num_params']}")

        Returns:
            list of dict，每个 dict 包含：
                - ``group_idx``: 参数组索引
                - ``mod_lr``: 当前模长学习率
                - ``phase_lr``: 当前相位学习率
                - ``cayley_lr``: 当前 Cayley 参数学习率
                - ``real_lr``: 当前实数参数学习率
                - ``num_params``: 参数组内参数数量
                - ``num_complex``: 复数参数数量
                - ``num_cayley``: Cayley 参数数量
                - ``num_real``: 实数参数数量
                - ``has_grad``: 是否有任何参数具有梯度（上步是否被更新）
        """
        result = []
        for idx, group in enumerate(self.param_groups):
            n_complex = n_cayley = n_real = 0
            has_grad = False

            for p in group["params"]:
                is_complex = p.dtype in (torch.complex64, torch.complex128)
                is_cayley = _is_cayley_param(self._get_param_name(p))
                if p.grad is not None:
                    has_grad = True
                if is_cayley:
                    n_cayley += 1
                elif is_complex:
                    n_complex += 1
                else:
                    n_real += 1

            result.append({
                "group_idx": idx,
                "mod_lr": group.get("mod_lr", 0.0),
                "phase_lr": group.get("phase_lr", 0.0),
                "cayley_lr": group.get("cayley_lr", 0.0),
                "real_lr": group.get("real_lr", 0.0),
                "num_params": len(group["params"]),
                "num_complex": n_complex,
                "num_cayley": n_cayley,
                "num_real": n_real,
                "has_grad": has_grad,
            })

        return result



# ──────────────────────────────────────────────
# 学习率调度器：Warmup + 余弦退火
# ──────────────────────────────────────────────


class WarmupCosineScheduler:
    """量子梯度下降的 Warmup + 余弦退火学习率调度器。

    专为 QGD 设计：分别对模长学习率（mod_lr）和相位学习率（phase_lr）
    执行独立的 warmup + 余弦退火，因为二者在训练初期需要不同的预热策略。

    调度策略：
    - Warmup 阶段（step ∈ [0, warmup_steps)）：
        lr(t) = lr_max * t / warmup_steps    (线性预热)
    - 余弦退火阶段（step ∈ [warmup_steps, total_steps]）：
        lr(t) = lr_min + 0.5*(lr_max - lr_min)*(1 + cos(π*(t-warmup_steps)/(total_steps-warmup_steps)))

    为何复数优化需要专门的调度：
    - 模长学习率（mod_lr）过大会导致模长坍缩到 0（无法恢复）
    - 相位学习率（phase_lr）过大会导致相位噪声累积，损害干涉质量
    - 两者的 warmup 步数可独立配置，适应不同的收敛速度

    Args:
        optimizer: QGD 优化器实例
        warmup_steps: 预热步数（建议 total_steps 的 5-10%）
        total_steps: 总训练步数
        mod_lr_max: 模长学习率峰值（默认从 optimizer.defaults 读取）
        phase_lr_max: 相位学习率峰值（默认从 optimizer.defaults 读取）
        mod_lr_min: 模长学习率最小值（默认 1e-6）
        phase_lr_min: 相位学习率最小值（默认 mod_lr_min * 10）
        last_step: 上一步的 step（用于恢复训练，-1 表示从头开始）
    """

    def __init__(
        self,
        optimizer: "QGD",
        warmup_steps: int,
        total_steps: int,
        mod_lr_max: Optional[float] = None,
        phase_lr_max: Optional[float] = None,
        mod_lr_min: float = 1e-6,
        phase_lr_min: Optional[float] = None,
        last_step: int = -1,
    ):
        if not isinstance(optimizer, QGD):
            raise TypeError(f"WarmupCosineScheduler 只支持 QGD 优化器，收到: {type(optimizer)}")
        if warmup_steps >= total_steps:
            raise ValueError(
                f"warmup_steps({warmup_steps}) 必须小于 total_steps({total_steps})"
            )

        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

        # 从优化器 defaults 读取峰值学习率
        self.mod_lr_max = mod_lr_max or optimizer.defaults["mod_lr"]
        self.phase_lr_max = phase_lr_max or optimizer.defaults["phase_lr"]

        self.mod_lr_min = mod_lr_min
        # 相位 lr 的衰减下限通常比模长大 10 倍（相位更需要保持探索性）
        self.phase_lr_min = phase_lr_min or (mod_lr_min * 10)

        self._step = last_step + 1  # 内部 step 计数（从 0 开始）

    @property
    def last_step(self) -> int:
        """当前已执行的 step 数（从 0 开始）"""
        return self._step

    def _cosine_lr(self, t: int, lr_max: float, lr_min: float) -> float:
        """计算余弦退火 lr 值。

        Args:
            t: 余弦退火阶段的相对步数（t=0 → lr_max，t=T → lr_min）
            lr_max: 峰值学习率
            lr_min: 最小学习率
        Returns:
            当前步的学习率
        """
        T = self.total_steps - self.warmup_steps
        if T <= 0:
            return lr_min
        ratio = t / T
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * ratio))
        return lr_min + (lr_max - lr_min) * cosine_decay

    def get_lr(self) -> dict:
        """计算当前 step 对应的学习率（不修改优化器状态）。

        Returns:
            dict with 'mod_lr', 'phase_lr', 'step'
        """
        t = self._step
        w = self.warmup_steps

        if t < w:
            # 线性 warmup
            ratio = max(t, 1) / max(w, 1)
            mod_lr = self.mod_lr_max * ratio
            phase_lr = self.phase_lr_max * ratio
        else:
            # 余弦退火
            t_cosine = t - w
            mod_lr = self._cosine_lr(t_cosine, self.mod_lr_max, self.mod_lr_min)
            phase_lr = self._cosine_lr(t_cosine, self.phase_lr_max, self.phase_lr_min)

        return {"mod_lr": mod_lr, "phase_lr": phase_lr, "step": t}

    def step(self):
        """更新优化器的学习率并递增 step 计数。

        调用时机：每次 optimizer.step() 之后调用一次。

        Example:
            >>> for batch in dataloader:
            ...     loss = model(batch)
            ...     optimizer.zero_grad()
            ...     loss.backward()
            ...     optimizer.step()
            ...     scheduler.step()   # ← 在 optimizer.step() 后调用
        """
        lr_dict = self.get_lr()
        for group in self.optimizer.param_groups:
            group["mod_lr"] = lr_dict["mod_lr"]
            group["phase_lr"] = lr_dict["phase_lr"]
        self._step += 1

    def state_dict(self) -> dict:
        """返回调度器状态（用于保存 checkpoint）。"""
        return {
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "mod_lr_max": self.mod_lr_max,
            "phase_lr_max": self.phase_lr_max,
            "mod_lr_min": self.mod_lr_min,
            "phase_lr_min": self.phase_lr_min,
            "_step": self._step,
        }

    def load_state_dict(self, state: dict):
        """从 checkpoint 恢复调度器状态。"""
        self.warmup_steps = state["warmup_steps"]
        self.total_steps = state["total_steps"]
        self.mod_lr_max = state["mod_lr_max"]
        self.phase_lr_max = state["phase_lr_max"]
        self.mod_lr_min = state["mod_lr_min"]
        self.phase_lr_min = state["phase_lr_min"]
        self._step = state["_step"]

    def __repr__(self) -> str:
        lr = self.get_lr()
        return (
            f"WarmupCosineScheduler("
            f"step={self._step}/{self.total_steps}, "
            f"warmup={self.warmup_steps}, "
            f"mod_lr={lr['mod_lr']:.2e}, "
            f"phase_lr={lr['phase_lr']:.2e})"
        )


# ──────────────────────────────────────────────
# 向后兼容
# ──────────────────────────────────────────────

# 保留旧版 QGD 类名（QGD 本身就是新版）
__all__ = ["QGD", "WarmupCosineScheduler"]
