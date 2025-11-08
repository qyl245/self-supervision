import os
import torch
from torch.utils.data import DataLoader, Dataset
import random
from sliding_window import SlidingWindowDataset as SlidingWindowDatasetPhase1
from sklearn.preprocessing import LabelEncoder
from logging_utils import setup_logger

logger = setup_logger()

class SlidingWindowDatasetPhase2(SlidingWindowDatasetPhase1):
    """
    Phase2 独立版 SlidingWindowDataset：
    用于跨模态任务，强制双模态存在
    """
    def __init__(self, *args, **kwargs):
        # 禁用单模态过滤逻辑
        kwargs['enable_filtering'] = False
        # 固定成多模态模式
        kwargs['modality'] = None
        super().__init__(*args, **kwargs)

        # 在初始化之后过滤 preprocessed_trials
        self.preprocessed_trials = self._filter_dual_modality(self.preprocessed_trials)
        # 重建 window_index，因为 trial 数量变化了
        self.window_index = self._build_window_index_from_cache()

    def _filter_dual_modality(self, trials):
        """只保留既有 emg_path 又有 imu_path 的 trial"""
        filtered = []
        for t in trials:
            if t.get('emg_path') and t.get('imu_path'):
                filtered.append(t)
        return filtered

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        # Phase2必须双模态
        if sample.get('emg') is None or sample.get('imu') is None:
            raise ValueError(f"[Phase2Dataset] 索引 {idx} 缺失模态")
        return sample



def build_label_encoder(dataset):
    """
    根据整个 dataset 的 activity_name 构建 LabelEncoder
    """
    all_labels = [sample['metadata']['activity_name'] for sample in dataset]
    le = LabelEncoder()
    le.fit(all_labels)
    return le


# ====== 数据增强函数 ======
def augment_emg(emg_tensor):
    """
    emg_tensor: (C, T) 单样本 或 (B, C, T) 批量
    """
    x = emg_tensor.clone()

    # 1. 高斯噪声扰动
    if random.random() > 0.5:
        x += torch.randn_like(x) * 0.05

    # 2. 时间掩码
    if random.random() > 0.5:
        # 针对每个通道做随机掩码
        mask = torch.rand_like(x[..., :1]) > 0.2
        x = x * mask

    # 3. 通道丢弃
    if random.random() > 0.3:
        drop_idx = random.randint(0, x.shape[-2] - 1)  # C维
        if x.dim() == 2:       # (C, T)
            x[drop_idx] = 0
        elif x.dim() == 3:     # (B, C, T)
            x[:, drop_idx] = 0

    # 4. 时间缩放 (修复线性模式4D错误)
    if random.random() > 0.3:
        scale = random.uniform(0.8, 1.2)
        T = x.shape[-1]
        new_T = max(1, int(T * scale))

        if x.dim() == 2:  # (C, T) 单样本
            x = torch.nn.functional.interpolate(
                x.unsqueeze(0), size=new_T, mode='linear', align_corners=False
            ).squeeze(0)
        elif x.dim() == 3:  # (B, C, T) 批量
            out_batch = []
            for sample in x:
                sample_scaled = torch.nn.functional.interpolate(
                    sample.unsqueeze(0), size=new_T, mode='linear', align_corners=False
                ).squeeze(0)
                out_batch.append(sample_scaled)
            x = torch.stack(out_batch, dim=0)

        # 保持原长度T
        if new_T < T:
            pad_len = T - new_T
            pad_shape = list(x.shape[:-1]) + [pad_len]
            x = torch.cat([x, torch.zeros(*pad_shape, device=x.device)], dim=-1)
        elif new_T > T:
            x = x[..., :T]

    # 5. 频域扰动
    if random.random() > 0.3:
        freq = torch.fft.rfft(x, dim=-1)
        noise = (torch.randn_like(freq) + 1j * torch.randn_like(freq)) * 0.02
        freq = freq + noise
        x = torch.fft.irfft(freq, n=x.shape[-1], dim=-1)

    return x


def augment_imu(imu_tensor):
    """
    imu_tensor: (S, C, T) 不带 batch, 或 (B, S, C, T)
    """
    x = imu_tensor.clone()

    # 1. 高斯噪声扰动
    if random.random() > 0.5:
        x += torch.randn_like(x) * 0.05

    # 2. 时间掩码
    if random.random() > 0.5:
        mask = torch.rand_like(x[..., :1]) > 0.2
        x = x * mask

    # 3. 传感器丢弃
    if random.random() > 0.3:
        drop_sensor = random.randint(0, x.shape[0] - 1)
        x[drop_sensor] = 0

    # 4. 时间缩放 (修复4D输入问题)
    if random.random() > 0.3:
        scale = random.uniform(0.85, 1.15)
        T = x.shape[-1]
        new_T = max(1, int(T * scale))

        if x.dim() == 3:  # (S, C, T)
            out = []
            for sensor in x:
                # sensor: (C, T) -> 插值要求(N, C, L)
                sensor_scaled = torch.nn.functional.interpolate(
                    sensor.unsqueeze(0), size=new_T, mode='linear', align_corners=False
                ).squeeze(0)
                out.append(sensor_scaled)
            x = torch.stack(out, dim=0)
        elif x.dim() == 4:  # (B, S, C, T)
            out_batch = []
            for sample in x:
                out_sensors = []
                for sensor in sample:
                    sensor_scaled = torch.nn.functional.interpolate(
                        sensor.unsqueeze(0), size=new_T, mode='linear', align_corners=False
                    ).squeeze(0)
                    out_sensors.append(sensor_scaled)
                out_batch.append(torch.stack(out_sensors, dim=0))
            x = torch.stack(out_batch, dim=0)

        # 尺寸对齐回原T
        if new_T < T:
            pad_len = T - new_T
            pad_shape = list(x.shape[:-1]) + [pad_len]
            x = torch.cat([x, torch.zeros(*pad_shape, device=x.device)], dim=-1)
        elif new_T > T:
            x = x[..., :T]

    # 5. 频域扰动
    if random.random() > 0.3:
        freq = torch.fft.rfft(x, dim=-1)
        noise = (torch.randn_like(freq) + 1j * torch.randn_like(freq)) * 0.02
        freq = freq + noise
        x = torch.fft.irfft(freq, n=x.shape[-1], dim=-1)

    return x


# ====== Phase 2 Transform ======
def phase2_transform(sample):
    """确保双模态数据在增强前存在"""
    if sample['emg'] is None or sample['imu'] is None:
        raise ValueError("[phase2_transform] 输入样本缺模态！")

    emg = augment_emg(sample['emg'])
    imu = augment_imu(sample['imu'])

    meta = sample['metadata']
    meta['trial_id'] = str(meta['trial_id'])
    meta['window_index'] = int(meta['window_index'])

    return {
        'emg': emg,
        'imu': imu,
        'metadata': meta
    }


class Phase2Dataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        return phase2_transform(sample)


def make_collate_fn(le):
    def collate_fn(batch):
        batch_out = {
            'emg': torch.stack([item['emg'] for item in batch]),
            'imu': torch.stack([item['imu'] for item in batch]),
            'metadata': [item['metadata'] for item in batch],
            'labels': torch.tensor(
                le.transform([item['metadata']['activity_name'] for item in batch]),
                dtype=torch.long
            )
        }
        return batch_out
    return collate_fn


def build_phase2_loaders(config, base_datasets):
    data_cfg = config['data']

    # SlidingWindowDataset 请用你原来的定义
    full_dataset = SlidingWindowDatasetPhase2(
        base_datasets=base_datasets,
        window_sec=data_cfg['window_sec'],
        step_sec=data_cfg['step_sec'],
        target_sr=data_cfg['target_sr'],
        cache_dir=data_cfg['cache_dir'],
        enable_filtering=data_cfg['enable_filtering'],
        config=config,
        num_jobs=data_cfg.get('num_jobs', 20),
        force_rebuild=False
    )


    # ===== Step 3: 稀有动作过滤（方法1） =====
    min_subj_per_act = data_cfg.get('min_subjects_per_action', None)
    rare_actions = set()
    if min_subj_per_act is not None:
        from collections import defaultdict
        # 统计每个动作的受试者集合
        subjects_per_action = defaultdict(set)
        for win in full_dataset.window_index:
            act = str(win['metadata']['activity_name'])
            sid = str(win['metadata']['subject_id'])
            subjects_per_action[act].add(sid)

        rare_actions = {act for act, subj_set in subjects_per_action.items()
                        if len(subj_set) < min_subj_per_act}
        if rare_actions:
            logger.info(f"[FILTER] Removing rare actions with subjects < {min_subj_per_act}: {sorted(rare_actions)}")
        else:
            logger.info("[FILTER] No rare actions found.")

    # ===== Step 4: 数据集划分 =====
    if data_cfg.get('subject_split_path') and os.path.exists(data_cfg['subject_split_path']):
        split = torch.load(data_cfg['subject_split_path'])
        train_subjects = set(split['train_subjects'])
        val_subjects = set(split['val_subjects'])
        logger.info("📂 Loaded subject split from file, ensuring consistent train/val subjects.")

    else:
        seed = config.get("seed", 42)
        torch.manual_seed(seed)

        # 获取所有 unique subject_id
        all_subjects = list({win['metadata']['subject_id'] for win in full_dataset.window_index})
        logger.info(f"📊 Total unique subjects: {len(all_subjects)}")

        # 随机打乱受试者顺序
        if isinstance(all_subjects[0], str):
            import random
            random.Random(seed).shuffle(all_subjects)
        else:
            all_subjects = torch.tensor(all_subjects)
            all_subjects = all_subjects[torch.randperm(len(all_subjects))].tolist()

        train_subject_count = int(0.8 * len(all_subjects))
        train_subjects = set(all_subjects[:train_subject_count])
        val_subjects = set(all_subjects[train_subject_count:])

        # 保存划分
        torch.save({
            'train_subjects': list(train_subjects),
            'val_subjects': list(val_subjects)
        }, data_cfg.get('subject_split_path'))

    # 根据 subject 过滤 + 稀有动作过滤窗口索引
    train_indices = [
        i for i, win in enumerate(full_dataset.window_index)
        if win['metadata']['subject_id'] in train_subjects
           and (str(win['metadata']['activity_name']) not in rare_actions)
    ]
    val_indices = [
        i for i, win in enumerate(full_dataset.window_index)
        if win['metadata']['subject_id'] in val_subjects
           and (str(win['metadata']['activity_name']) not in rare_actions)
    ]

    logger.info(f"📊 Train samples(after filter): {len(train_indices)}, Val samples(after filter): {len(val_indices)}")

    # ===== 调试日志 - 检查交叉受试者 =====
    overlap_subjects = train_subjects & val_subjects
    if overlap_subjects:
        logger.warning(
            f"⚠️ 数据泄露风险: {len(overlap_subjects)}个受试者在 Train 和 Val 集都有出现: {overlap_subjects}")
    else:
        logger.info("✅ 无受试者交叉，数据划分安全。")



    train_ds = Phase2Dataset(torch.utils.data.Subset(full_dataset, train_indices))
    val_ds = Phase2Dataset(torch.utils.data.Subset(full_dataset, val_indices))


    le = build_label_encoder(val_ds)

    collate_fn_train = make_collate_fn(le)
    collate_fn_val = make_collate_fn(le)

    worker_count = min(data_cfg.get('num_workers', 12), os.cpu_count())
    prefetch_factor = data_cfg.get('prefetch_factor', 4)

    train_loader = DataLoader(
        train_ds,
        batch_size=data_cfg['batch_size'],
        shuffle=True,
        num_workers=worker_count,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
        drop_last=True,
        collate_fn=collate_fn_train
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=data_cfg['batch_size'],
        shuffle=False,
        num_workers=worker_count,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
        drop_last=False,
        collate_fn=collate_fn_val
    )

    return train_loader, val_loader, le
