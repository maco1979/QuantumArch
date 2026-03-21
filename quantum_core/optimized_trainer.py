"""
量子神经网络训练器 - 优化版本

优化内容：
1. 增大学习率 (1e-4 -> 1e-3)
2. 增大batch size (2 -> 16)
3. 添加学习率调度器 (Cosine Annealing + Warmup)
4. 混合精度训练支持
5. 梯度裁剪
6. 早停机制
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from typing import Dict, Optional, Callable
import time
from dataclasses import dataclass, field


@dataclass
class TrainingConfig:
    """优化后的训练配置"""
    # 基础参数
    learning_rate: float = 1e-3       # 从1e-4提升到1e-3
    weight_decay: float = 0.01
    batch_size: int = 16               # 从2提升到16
    num_epochs: int = 100
    gradient_clip: float = 1.0         # 梯度裁剪
    
    # 学习率调度
    warmup_steps: int = 500            # 预热步数
    min_lr: float = 1e-6               # 最小学习率
    use_cosine: bool = True            # 使用余弦退火
    use_onecycle: bool = False         # 或使用OneCycleLR
    
    # 早停
    early_stopping_patience: int = 30   # 早停耐心值
    early_stopping_threshold: float = 1e-4
    
    # 混合精度
    use_amp: bool = True               # 启用自动混合精度
    amp_dtype: torch.dtype = torch.float16
    
    # 设备
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 日志
    log_interval: int = 10
    eval_interval: int = 100


class QuantumTrainer:
    """量子神经网络优化训练器"""
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[TrainingConfig] = None,
    ):
        self.config = config or TrainingConfig()
        self.model = model.to(self.config.device)
        
        # 优化器 - 使用AdamW
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        
        # 学习率调度器
        self.scheduler = self._create_scheduler()
        
        # 混合精度训练器
        if self.config.use_amp and torch.cuda.is_available():
            self.scaler = GradScaler()
        else:
            self.scaler = None
        
        # 训练状态
        self.global_step = 0
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.history = []
        
    def _create_scheduler(self):
        """创建学习率调度器"""
        if self.config.use_onecycle:
            return OneCycleLR(
                self.optimizer,
                max_lr=self.config.learning_rate,
                total_steps=self.config.num_epochs,
                pct_start=0.1,
                anneal_strategy='cos',
            )
        elif self.config.use_cosine:
            return CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=10,                    # 第一个周期
                T_mult=2,                   # 周期倍数
                eta_min=self.config.min_lr,
            )
        else:
            return None
    
    def train_step(self, batch: Dict) -> Dict:
        """单步训练"""
        self.model.train()
        
        # 获取数据并移动到设备
        inputs = batch['input_ids'].to(self.config.device)
        labels = batch.get('labels', inputs).to(self.config.device)
        
        # 梯度清零
        self.optimizer.zero_grad()
        
        # 混合精度前向传播
        if self.scaler is not None:
            with autocast(dtype=self.config.amp_dtype):
                outputs = self.model(inputs)
                if isinstance(outputs, tuple):
                    loss = nn.functional.cross_entropy(
                        outputs[0].view(-1, outputs[0].size(-1)),
                        labels.view(-1)
                    )
                else:
                    loss = nn.functional.cross_entropy(
                        outputs.view(-1, outputs.size(-1)),
                        labels.view(-1)
                    )
            
            # 混合精度反向传播
            self.scaler.scale(loss).backward()
            
            # 梯度裁剪
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.gradient_clip
            )
            
            # 参数更新
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # 标准训练（CPU或无AMP）
            outputs = self.model(inputs)
            if isinstance(outputs, tuple):
                loss = nn.functional.cross_entropy(
                    outputs[0].view(-1, outputs[0].size(-1)),
                    labels.view(-1)
                )
            else:
                loss = nn.functional.cross_entropy(
                    outputs.view(-1, outputs.size(-1)),
                    labels.view(-1)
                )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip
            )
            self.optimizer.step()
        
        # 学习率调度
        if self.scheduler is not None:
            self.scheduler.step()
        
        self.global_step += 1
        
        return {
            'loss': loss.item(),
            'lr': self.optimizer.param_groups[0]['lr'],
            'step': self.global_step,
        }
    
    def evaluate(self, eval_batch: Dict) -> Dict:
        """评估模型"""
        self.model.eval()
        
        with torch.no_grad():
            inputs = eval_batch['input_ids'].to(self.config.device)
            labels = eval_batch.get('labels', inputs).to(self.config.device)
            
            outputs = self.model(inputs)
            if isinstance(outputs, tuple):
                loss = nn.functional.cross_entropy(
                    outputs[0].view(-1, outputs[0].size(-1)),
                    labels.view(-1)
                )
            else:
                loss = nn.functional.cross_entropy(
                    outputs.view(-1, outputs.size(-1)),
                    labels.view(-1)
                )
        
        return {'eval_loss': loss.item()}
    
    def should_early_stop(self, loss: float) -> bool:
        """检查是否应该早停"""
        if loss < self.best_loss - self.config.early_stopping_threshold:
            self.best_loss = loss
            self.patience_counter = 0
            return False
        
        self.patience_counter += 1
        return self.patience_counter >= self.config.early_stopping_patience
    
    def train(
        self,
        train_dataloader,
        eval_dataloader=None,
        callback: Optional[Callable] = None,
    ):
        """完整训练循环"""
        print(f"\n{'='*60}")
        print(f"开始训练 (优化配置)")
        print(f"{'='*60}")
        print(f"设备: {self.config.device}")
        print(f"Batch Size: {self.config.batch_size}")
        print(f"学习率: {self.config.learning_rate}")
        print(f"权重衰减: {self.config.weight_decay}")
        print(f"梯度裁剪: {self.config.gradient_clip}")
        print(f"混合精度: {self.scaler is not None}")
        print(f"早停耐心: {self.config.early_stopping_patience}")
        print(f"{'='*60}\n")
        
        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            epoch_start = time.time()
            
            for batch_idx, batch in enumerate(train_dataloader):
                metrics = self.train_step(batch)
                epoch_loss += metrics['loss']
                
                # 日志输出
                if self.global_step % self.config.log_interval == 0:
                    print(f"Epoch {epoch+1} | Step {self.global_step} | "
                          f"Loss: {metrics['loss']:.4f} | "
                          f"LR: {metrics['lr']:.2e}")
                
                # 评估
                if eval_dataloader and self.global_step % self.config.eval_interval == 0:
                    eval_batch = next(iter(eval_dataloader))
                    eval_metrics = self.evaluate(eval_batch)
                    print(f"  Eval Loss: {eval_metrics['eval_loss']:.4f}")
                    
                    # 早停检查
                    if self.should_early_stop(eval_metrics['eval_loss']):
                        print(f"\n早停触发于 Step {self.global_step}")
                        return self.history
            
            # Epoch统计
            avg_loss = epoch_loss / len(train_dataloader)
            epoch_time = time.time() - epoch_start
            
            self.history.append({
                'epoch': epoch + 1,
                'loss': avg_loss,
                'lr': self.optimizer.param_groups[0]['lr'],
                'time': epoch_time,
            })
            
            print(f"Epoch {epoch+1} 完成 | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Time: {epoch_time:.2f}s | "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # 回调
            if callback:
                callback(epoch, avg_loss, self.model)
        
        return self.history


def create_optimized_trainer(model: nn.Module, device: str = "auto") -> QuantumTrainer:
    """创建优化训练器的工厂函数"""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    config = TrainingConfig(
        learning_rate=1e-3,       # 10倍学习率
        batch_size=16,            # 8倍batch
        use_amp=torch.cuda.is_available(),
        device=device,
        warmup_steps=100,
        use_cosine=True,
        early_stopping_patience=20,
    )
    
    return QuantumTrainer(model, config)
