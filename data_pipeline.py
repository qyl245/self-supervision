import os
import torch
from torch.utils.data import DataLoader

from datasets import KneePAD, MovePort
from sliding_window import SlidingWindowDataset
from logging_utils import setup_logger
from augmentations import create_ssl_transforms

logger = setup_logger()

class TransformSubsetDataset(torch.utils.data.Dataset):
    """
    用于对数据集子集应用 transform
    """
    def __init__(self, dataset, indices, transform=None):
        self.base_dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):
        sample = self.base_dataset[self.indices[idx]]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.indices)


def create_dataloaders(config, modality):
    """
    创建自监督学习(SSL)训练的DataLoader。

    Args:
        config (dict): 全局配置字典
        modality (str): 模态类型 ("emg" 或 "imu")

    Returns:
        dict: {'train': train_loader, 'val': val_loader}
    """

    data_cfg = config['data']
    train_cfg = config['train']

    # ===== Step 1: 加载基础数据集 =====
    logger.info("📂 Loading base datasets...")

    knee_pad = KneePAD(root_dir=data_cfg['knee_pad_root'])
    move_port = MovePort(root_dir=data_cfg['move_port_root'])
    base_datasets = [knee_pad, move_port]

    # ===== Step 2: 创建滑动窗口数据集 =====
    logger.info("📊 Creating sliding window dataset...")
    train_dataset = SlidingWindowDataset(
        base_datasets=base_datasets,
        window_sec=data_cfg['window_sec'],
        step_sec=data_cfg['step_sec'],
        target_sr=data_cfg['target_sr'],
        cache_dir=data_cfg['cache_dir'],
        enable_filtering=data_cfg['enable_filtering'],
        config=config,
        num_jobs=data_cfg.get('num_jobs', 4),
        force_rebuild=data_cfg.get('force_rebuild', False),
        modality=modality
    )

    # ===== Step 3: 数据集划分 =====
    seed = config.get("seed", 42)
    torch.manual_seed(seed)

    indices = torch.randperm(len(train_dataset))
    train_size = int(0.8 * len(train_dataset))

    logger.info(f"📊 Total samples: {len(train_dataset)}")
    logger.info(f"📊 Train samples: {train_size}")
    logger.info(f"📊 Val samples: {len(train_dataset) - train_size}")

    # ===== Step 4: 创建SSL数据增强 Transform =====
    ssl_transform = create_ssl_transforms(
        mask_ratio=train_cfg['mask_ratio'],
        num_views=train_cfg['num_views'],
        num_chunks=train_cfg['num_chunks'],
        modality=modality
    )

    # ===== Step 5: 构造子集数据集 =====
    train_ds = TransformSubsetDataset(train_dataset, indices[:train_size].tolist(), ssl_transform)
    val_ds = TransformSubsetDataset(train_dataset, indices[train_size:].tolist(), ssl_transform)

    # ===== Step 6: DataLoader 性能优化参数 =====
    worker_count = min(data_cfg.get('num_workers', 4), os.cpu_count() or 1)
    prefetch_factor = data_cfg.get('prefetch_factor', 4)

    logger.info(f"🔧 DataLoader workers: {worker_count}, prefetch_factor: {prefetch_factor}")

    # ===== Step 7: 创建 DataLoader =====
    train_loader = DataLoader(
        train_ds,
        batch_size=data_cfg['batch_size'],
        shuffle=True,
        num_workers=worker_count,
        pin_memory=False,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
        drop_last=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=data_cfg['batch_size'],
        shuffle=False,
        num_workers=worker_count,
        pin_memory=False,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
        drop_last=False
    )

    logger.info(f"✅ DataLoaders created - Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    logger.info(f"🔧 SSL Transform config - mask_ratio: {train_cfg['mask_ratio']}, num_views: {train_cfg['num_views']}, num_chunks: {train_cfg['num_chunks']}")

    return {'train': train_loader, 'val': val_loader}
