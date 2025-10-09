"""
训练主入口
"""
import yaml
import torch
import argparse
from pathlib import Path

from data_pipeline import create_dataloaders
from ssl_model import create_emg_encoder, create_imu_encoder, create_ssl_model
from train_ssl import SSLTrainer
from logging_utils import setup_logger

logger = setup_logger()


def load_config(config_path: Path):
    """加载并检查配置文件"""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    for key in ('data', 'train', 'model'):
        if key not in config:
            raise KeyError(f"Config missing required section: '{key}'")
    return config


def main(config_path: str, modality='emg', resume=False, resume_path=None, debug=False, output_dir=None):
    try:
        # 1. 加载配置
        config_path = Path(config_path)
        config = load_config(config_path)

        if output_dir:
            config['train']['output_dir'] = str(output_dir)

        logger.info(f" Loaded config from {config_path}")
        logger.info(f" Modality: {modality}")

        # 2. 创建数据流水线
        dataloaders = create_dataloaders(config, modality)

        # 3. 创建模型
        encoder = create_emg_encoder(config) if modality == 'emg' else create_imu_encoder(config)
        model = create_ssl_model(encoder, config, modality)

        # 4. 创建训练器
        trainer = SSLTrainer(model, dataloaders, config, modality)
        if debug:
            logger.info(" Debug mode enabled – verbose output")

        # 5. 恢复训练
        if resume:
            if resume_path:
                ckpt_file = Path(resume_path)
            else:
                ckpt_file = Path(config['train']['output_dir']) / 'models' / f"checkpoint_{modality}_latest.pt"

            if ckpt_file.exists():
                logger.info(f" Resuming from checkpoint: {ckpt_file}")
                checkpoint = torch.load(ckpt_file, map_location=trainer.device)
                trainer.model.load_state_dict(checkpoint['model_state_dict'])
                trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'scheduler_state_dict' in checkpoint:
                    trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            else:
                logger.warning(f"No checkpoint found at {ckpt_file}, starting fresh.")

        # 6. 启动训练
        trainer.train()

        logger.info(f" Training completed for {modality}. Encoder saved to {config['train']['output_dir']}/models/time_encoder_{modality}.pt")
        return 0

    except Exception as e:
        logger.error(f" Training failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train SSL Model")
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--modality', choices=['emg', 'imu'], default='emg')
    parser.add_argument('--resume', action='store_true', help='Resume training from latest checkpoint')
    parser.add_argument('--resume_path', type=str, help='Path to specific checkpoint file')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode in trainer/model')
    parser.add_argument('--output_dir', type=str, help='Override output_dir in config')
    args = parser.parse_args()

    exit_code = main(args.config, args.modality, args.resume, args.resume_path, args.debug, args.output_dir)
    exit(exit_code)