"""
SSL Trainer - 自监督训练器
负责：训练流程（train/validate）、AMP混合精度、Warmup+Cosine调度、日志与模型保存
"""
import os
import torch
import torch.nn.utils
import torch.optim as optim
from tqdm import tqdm
from torch.cuda.amp import GradScaler
from transformers import get_cosine_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter

from logging_utils import setup_logger

logger = setup_logger()


class SSLTrainer:
    def __init__(self, model, dataloaders, config, modality):
        config = self._fix_config_types(config)

        self.model = model
        self.train_loader = dataloaders['train']
        self.val_loader = dataloaders['val']
        self.config = config
        self.modality = modality

        # 设备
        self.device = config['train']['device']
        if self.device.startswith('cuda') and not torch.cuda.is_available():
            logger.warning("⚠️ CUDA not available, using CPU")
            self.device = 'cpu'
            self.config['train']['device'] = 'cpu'
        self.model.to(self.device)

        # 优化器
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config['train']['lr'],
            weight_decay=config['train']['weight_decay']
        )

        # Warmup + Cosine LR schedule
        total_steps = len(self.train_loader) * config['train']['epochs']
        warmup_steps = len(self.train_loader) * config['train'].get('warmup_epochs', 3)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        self.loss_weights = config['train']['loss_weights']

        # 输出路径
        self.output_dir = config['train']['output_dir']
        os.makedirs(os.path.join(self.output_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'logs'), exist_ok=True)

        # TensorBoard
        self.writer = SummaryWriter(os.path.join(self.output_dir, 'tensorboard'))

        # AMP GradScaler
        self.scaler = GradScaler()

        logger.info(f"✅ SSLTrainer initialized - Device: {self.device}, Loss weights: {self.loss_weights}")

    def _fix_config_types(self, config):
        """确保配置中的数据类型正确"""
        train_cfg = config['train']
        # 数值转换
        for key, expected_type in [
            ('lr', float), ('weight_decay', float), ('epochs', int),
            ('mask_ratio', float), ('num_chunks', int)
        ]:
            if key in train_cfg and isinstance(train_cfg[key], str):
                train_cfg[key] = expected_type(train_cfg[key])

        # 损失权重转换
        if 'loss_weights' in train_cfg:
            for k, v in train_cfg['loss_weights'].items():
                if isinstance(v, str):
                    train_cfg['loss_weights'][k] = float(v)
        return config

    def train_epoch(self, epoch):
        self.model.train()
        total_loss, valid_batches = 0.0, 0
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1} [Train]')

        for batch in pbar:
            if batch is None:
                continue
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            self.optimizer.zero_grad()

            with torch.amp.autocast(device_type='cuda', enabled=self.device.startswith('cuda')):
                results = self.model(**batch)
                losses = results['losses']

                total_loss_val = sum(
                    losses[k] * float(self.loss_weights[k])
                    for k in losses if k in self.loss_weights and isinstance(losses[k], torch.Tensor)
                )

            self.scaler.scale(total_loss_val).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()  # 每batch更新

            total_loss += total_loss_val.item()
            valid_batches += 1

        avg_loss = total_loss / max(valid_batches, 1)
        logger.info(f"📊 Epoch {epoch+1} Train Loss: {avg_loss:.4f}")
        return avg_loss

    def validate(self, epoch):
        self.model.eval()
        total_loss, valid_batches = 0.0, 0
        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1} [Val]')

        with torch.no_grad():
            for batch in pbar:
                if batch is None:
                    continue
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                with torch.amp.autocast(device_type='cuda', enabled=self.device.startswith('cuda')):
                    losses = self.model(**batch)['losses']
                    total_loss_val = sum(
                        losses[k] * float(self.loss_weights[k])
                        for k in losses if k in self.loss_weights and isinstance(losses[k], torch.Tensor)
                    )
                total_loss += total_loss_val.item()
                valid_batches += 1

        avg_loss = total_loss / max(valid_batches, 1)
        logger.info(f"📊 Epoch {epoch} Val Loss: {avg_loss:.4f}")
        return avg_loss

    def train(self):
        logger.info(f"🚀 Starting training for {self.config['train']['epochs']} epochs...")

        for epoch in range(self.config['train']['epochs']):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)

            # 写入TensorBoard
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Val', val_loss, epoch)

            # 每隔N个epoch保存一次
            if epoch % 10 == 0 or epoch == self.config['train']['epochs'] - 1:
                self.save_checkpoint(epoch)

        # 保存最终模型encoder
        self.save_encoder()
        logger.info("✅ Training completed!")

    def save_checkpoint(self, epoch):
        checkpoint_path = os.path.join(self.output_dir, 'models', f"checkpoint_{self.modality}_{epoch}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'modality': self.modality
        }, checkpoint_path)
        logger.info(f"💾 Checkpoint saved: {checkpoint_path}")

    def save_encoder(self):
        encoder_path = os.path.join(self.output_dir, 'models', f"time_encoder_{self.modality}.pt")
        torch.save(self.model.time_encoder.state_dict(), encoder_path)
        logger.info(f"💾 Time encoder saved: {encoder_path}")
