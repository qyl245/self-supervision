"""
数据增强工具函数
包含信号噪声、翻转、时间分块、随机打乱等，以及SSL任务组合transform
"""
import torch
import random


def add_noise(x: torch.Tensor, noise_level=0.1):
    """为信号添加高斯噪声，兼容 2D (channels, T) 和 3D (sensors, axes, T)"""
    orig_shape = x.shape
    if x.ndim == 3:
        C, A, T = x.shape
        x = x.view(C * A, T)
    elif x.ndim != 2:
        raise ValueError(f"Unexpected signal shape {x.shape}")

    noise = torch.randn_like(x) * noise_level
    x_noisy = x + noise
    return x_noisy.view(orig_shape) if len(orig_shape) == 3 else x_noisy


def time_flip(x: torch.Tensor):
    """在时间维度反转信号，兼容 2D 和 3D"""
    orig_shape = x.shape
    if x.ndim == 3:
        C, A, T = x.shape
        x = x.view(C * A, T)
    elif x.ndim != 2:
        raise ValueError(f"Unexpected signal shape {x.shape}")

    x_flipped = torch.flip(x, dims=[-1])
    return x_flipped.view(orig_shape) if len(orig_shape) == 3 else x_flipped


def split_into_chunks(x: torch.Tensor, num_chunks=3):
    """
    将信号在时间维度分成 num_chunks 份
    兼容 2D 和 3D
    返回: (chunks_tensor, num_chunks, chunk_size, T)
    """
    orig_shape = x.shape
    if x.ndim == 3:
        C, A, T = x.shape
        x = x.view(C * A, T)
    elif x.ndim == 2:
        C, T = x.shape
    else:
        raise ValueError(f"Unexpected signal shape {x.shape}")

    chunk_size = T // num_chunks
    chunks = torch.stack([x[..., i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks)], dim=0)
    return chunks, num_chunks, chunk_size, T


def random_shuffle(chunks_tuple):
    """对 split_into_chunks 返回的 chunks 进行随机打乱，保持返回结构一致"""
    chunks, num_chunks, chunk_size, T = chunks_tuple
    indices = list(range(num_chunks))
    random.shuffle(indices)
    shuffled_chunks = chunks[indices, ...]
    return shuffled_chunks, num_chunks, chunk_size, T


def create_ssl_transforms(mask_ratio, num_views, num_chunks, modality='emg'):
    """
    创建自监督SSL数据增强transform
    返回: function(sample) -> augmented_sample
    """
    def transform(sample):
        # 获取信号
        if modality == 'emg':
            signal = sample['emg']
        elif modality == 'imu':
            imu = sample['imu']
            num_sensors, num_axes, T = imu.shape
            signal = imu.view(num_sensors * num_axes, T)
        else:
            raise ValueError(f"Unsupported modality: {modality}")

        T = signal.shape[-1]

        # === MAE Mask ===
        mask = torch.rand(T) < mask_ratio
        mae_input = signal.clone()
        mae_input[..., mask] = 0

        # === Contrastive Views ===
        view1 = add_noise(signal, 0.1)
        view2 = time_flip(signal)
        contrastive_views = torch.stack([view1, view2], dim=0)

        # === Order Prediction ===
        chunks_info = split_into_chunks(signal, num_chunks)
        is_shuffled = torch.rand(1) < 0.5
        if is_shuffled:
            order_sequence = random_shuffle(chunks_info)[0]
            order_label = 0
        else:
            order_sequence = chunks_info[0]
            order_label = 1

        # 返回增强后的sample
        return {
            'mae_input': mae_input,
            'original_signal': signal,
            'mask': mask,
            'contrastive_views': contrastive_views,
            'order_sequence': order_sequence,
            'order_label': torch.tensor(order_label, dtype=torch.long),
            **sample
        }
    return transform
