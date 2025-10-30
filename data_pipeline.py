import os
import torch
from torch.utils.data import DataLoader
from collections import Counter, defaultdict

from datasets import KneePAD, MovePort
from sliding_window import SlidingWindowDataset
from logging_utils import setup_logger
from augmentations import create_ssl_transforms

logger = setup_logger()


class TransformSubsetDataset(torch.utils.data.Dataset):
    """
    用于对数据集子集应用 transform,并附加域标签
    """

    def __init__(self, dataset, indices, transform=None, domain_map=None):
        self.base_dataset = dataset
        self.indices = indices
        self.transform = transform
        self.domain_map = domain_map  # dict: dataset_name -> int

    def __getitem__(self, idx):
        sample = self.base_dataset[self.indices[idx]]
        if self.transform:
            sample = self.transform(sample)

        # 生成 domain_label
        if self.domain_map is not None and 'metadata' in sample:
            meta = sample['metadata']
            dataset_name = meta.get('data', None)  # 需要保证 SlidingWindowDataset 有这个字段
            if dataset_name is not None and dataset_name in self.domain_map:
                sample['domain_label'] = torch.tensor(self.domain_map[dataset_name], dtype=torch.long)
            else:
                sample['domain_label'] = torch.tensor(-1, dtype=torch.long)  # 未知域

        return sample

    def __len__(self):
        return len(self.indices)


def audit_actions(dataloader, name="Train", enabled=True):
    if not enabled:
        return set(), Counter(), defaultdict(set)

    action_counts = Counter()
    subjects_per_action = defaultdict(set)
    total_windows = 0

    for batch in dataloader:
        meta = batch['metadata']
        # 兼容两种结构：meta 是 list(dict) 或 dict里包含批内列表
        if isinstance(meta, list):
            for m in meta:
                act = str(m.get('activity_name', 'UNKNOWN'))
                sid = str(m.get('subject_id', 'UNKNOWN'))
                action_counts[act] += 1
                subjects_per_action[act].add(sid)
                total_windows += 1
        elif isinstance(meta, dict):
            acts = meta.get('activity_name')
            sids = meta.get('subject_id')
            # 把批内每个样本展开
            if isinstance(acts, (list, tuple)) and isinstance(sids, (list, tuple)):
                for act, sid in zip(acts, sids):
                    act = str(act)
                    sid = str(sid)
                    action_counts[act] += 1
                    subjects_per_action[act].add(sid)
                    total_windows += 1
            else:
                act = str(acts)
                sid = str(sids)
                action_counts[act] += 1
                subjects_per_action[act].add(sid)
                total_windows += 1
        else:
            raise TypeError(f"Unexpected metadata type: {type(meta)}")

    actions = set(action_counts.keys())
    print(f"[{name}] windows={total_windows}, unique_actions={len(actions)}")
    # 列出每个动作的窗口数与受试者数
    for act in sorted(actions):
        print(f"  - {act}: windows={action_counts[act]}, subjects={len(subjects_per_action[act])}")

    return actions, action_counts, subjects_per_action


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
    full_dataset = SlidingWindowDataset(
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

    # ===== Step 4: 创建SSL数据增强 Transform =====
    ssl_transform = create_ssl_transforms(
        mask_ratio=train_cfg['mask_ratio'],
        num_views=train_cfg['num_views'],
        num_chunks=train_cfg['num_chunks'],
        modality=modality
    )

    # ===== Step 5: 构造子集数据集 =====

    # 定义 domain 映射规则（这里用数据集来源）
    domain_map = {"KneePAD": 0, "MovePort": 1}

    train_ds = TransformSubsetDataset(full_dataset, train_indices, ssl_transform, domain_map=domain_map)
    val_ds = TransformSubsetDataset(full_dataset, val_indices, ssl_transform, domain_map=domain_map)

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

    # # ===== 可选审计 =====
    # enable_audit = data_cfg.get('enable_audit', True)

    # if enable_audit:
    #     train_actions, train_counts, train_subj_per_act = audit_actions(
    #         train_loader, "Train", enabled=True)
    #     val_actions, val_counts, val_subj_per_act = audit_actions(
    #         val_loader, "Val", enabled=True)

    #     global_actions = train_actions | val_actions
    #     missing_in_train = global_actions - train_actions
    #     missing_in_val = global_actions - val_actions

    #     print("Global actions:", sorted(global_actions))
    #     print("Missing in Train:", sorted(missing_in_train))
    #     print("Missing in Val:", sorted(missing_in_val))
    # else:
    #     logger.info("🚫 Action audit disabled by config.")

    logger.info(f"✅ DataLoaders created - Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    logger.info(
        f"🔧 SSL Transform config - mask_ratio: {train_cfg['mask_ratio']}, num_views: {train_cfg['num_views']}, num_chunks: {train_cfg['num_chunks']}")

    return {'train': train_loader, 'val': val_loader}