"""
滑动窗口数据集与单Trial预处理
"""
import json
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.signal import resample
import joblib

from validators import DataValidator
from filters import SignalFilters
from logging_utils import setup_logger

logger = setup_logger()


def preprocess_single_trial(task, cache_dir, target_sr, enable_filtering):
    """
    预处理单个trial数据：
    1. 加载原始数据
    2. 数据验证与修复
    3. 可选滤波
    4. 重采样
    5. 缓存保存
    """
    try:
        cache_key = task['cache_key']
        emg_cache_path = Path(cache_dir) / f"emg_{cache_key}.npy"
        imu_cache_path = Path(cache_dir) / f"imu_{cache_key}.npy"

        # 检查缓存是否存在
        if emg_cache_path.exists() and imu_cache_path.exists():
            emg_cached = np.load(emg_cache_path, mmap_mode='r')
            return {
                'cache_key': cache_key,
                'emg_path': str(emg_cache_path),
                'imu_path': str(imu_cache_path),
                'length': emg_cached.shape[1],
                'metadata': task['metadata'],
                'dataset_name': task['dataset_name']
            }

        # Step1 加载原始数据
        emg_raw, imu_raw, _ = task['base_ds'].load_raw_data(task['trial_idx'])

        if emg_raw is None or imu_raw is None:
            logger.warning(f"Trial {cache_key}: raw data is None")
            return None

        # Step2 数据验证
        validator = DataValidator()
        emg_raw, emg_valid = validator.validate_and_fix_data(emg_raw, 'emg')
        imu_raw, imu_valid = validator.validate_and_fix_data(imu_raw, 'imu')

        if not (emg_valid and imu_valid):
            logger.warning(f"Trial {cache_key}: validation failed")
            return None

        # Step3 滤波
        if enable_filtering:
            emg_raw = SignalFilters.emg_filter(emg_raw, task['ds_info']['original_emg_rate'])
            imu_raw = SignalFilters.imu_filter(imu_raw, task['ds_info']['original_imu_rate'])

        # Step4 重采样
        num_resampled_emg = int(emg_raw.shape[1] * target_sr / task['ds_info']['original_emg_rate'])
        num_resampled_imu = int(imu_raw.shape[1] * target_sr / task['ds_info']['original_imu_rate'])
        emg_resampled = resample(emg_raw, num_resampled_emg, axis=1)
        imu_resampled = resample(imu_raw, num_resampled_imu, axis=1)

        # 对齐长度
        min_len = min(emg_resampled.shape[1], imu_resampled.shape[1])
        emg_resampled = emg_resampled[:, :min_len]
        imu_resampled = imu_resampled[:, :min_len]

        # Step5 保存缓存
        np.save(emg_cache_path, emg_resampled.astype(np.float32))
        np.save(imu_cache_path, imu_resampled.astype(np.float32))
        logger.info(f"Trial {cache_key}: cached successfully, length={min_len}")

        return {
            'cache_key': cache_key,
            'emg_path': str(emg_cache_path),
            'imu_path': str(imu_cache_path),
            'length': min_len,
            'metadata': task['metadata'],
            'dataset_name': task['dataset_name']
        }

    except Exception as e:
        logger.error(f"Trial {task.get('cache_key', 'unknown')}: {e}", exc_info=True)
        return None


class SlidingWindowDataset(Dataset):
    """
    滑动窗口数据集
    负责从缓存的trial数据中切分固定大小窗口，并对齐EMG和IMU模态
    """

    def __init__(
        self,
        base_datasets,
        window_sec,
        step_sec,
        target_sr,
        config,
        transform=None,
        cache_dir="cache/preprocessed",
        enable_filtering=True,
        num_jobs=15,
        force_rebuild=False,
        modality="emg"
    ):
        self.base_datasets = base_datasets
        self.transform = transform
        self.enable_filtering = enable_filtering
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.config = config
        self.modality = modality.lower()

        self.window_points = int(target_sr * window_sec)
        self.step_points = int(target_sr * step_sec)
        self.target_sr = target_sr

        cache_index_file = self.cache_dir / "preprocessed_index.json"
        if not force_rebuild and cache_index_file.exists():
            logger.info("📁 Loading from existing cache...")
            self.preprocessed_trials = self._load_cache_index()
        else:
            logger.info("🔄 Building new cache...")
            self._preprocess_all_trials(num_jobs)
            self._save_cache_index()

        self.window_index = self._build_window_index_from_cache()
        logger.info(f"✅ Dataset ready: {len(self.preprocessed_trials)} trials, {len(self.window_index)} windows")

    def _load_cache_index(self):
        cache_index_file = self.cache_dir / "preprocessed_index.json"
        with open(cache_index_file, 'r') as f:
            return json.load(f)

    def _save_cache_index(self):
        cache_index_file = self.cache_dir / "preprocessed_index.json"
        with open(cache_index_file, 'w') as f:
            json.dump(self.preprocessed_trials, f, indent=2)

    def _preprocess_all_trials(self, num_jobs):
        tasks = []
        for ds_idx, base_ds in enumerate(self.base_datasets):
            dataset_name = base_ds.__class__.__name__
            ds_info = self.config['dataset_info'][dataset_name]
            for trial_idx in range(len(base_ds)):
                row = base_ds.index_df.iloc[trial_idx]
                file_identifier = f"{row['emg_path']}_{row['imu_path']}"
                file_hash = hashlib.md5(file_identifier.encode()).hexdigest()[:8]
                cache_key = f"{dataset_name}_{ds_idx}_{trial_idx}_{file_hash}"
                tasks.append({
                    'ds_idx': ds_idx,
                    'trial_idx': trial_idx,
                    'dataset_name': dataset_name,
                    'cache_key': cache_key,
                    'base_ds': base_ds,
                    'metadata': {
                        'subject_id': row['subject_id'],
                        'activity_name': row['activity_name'],
                        'activity_condition': row['activity_condition'],
                        'trial_id': row['trial_id']
                    },
                    'ds_info': ds_info
                })

        logger.info(f"📊 Processing {len(tasks)} trials with {num_jobs} workers...")
        results = joblib.Parallel(n_jobs=num_jobs)(
            joblib.delayed(preprocess_single_trial)(task, self.cache_dir, self.target_sr, self.enable_filtering)
            for task in tasks
        )
        self.preprocessed_trials = [r for r in results if r is not None]
        logger.info(f"✅ Preprocessing complete: {len(self.preprocessed_trials)} succeeded, {len(results)-len(self.preprocessed_trials)} failed")

    def _build_window_index_from_cache(self):
        all_windows = []
        for trial_idx, trial_info in enumerate(self.preprocessed_trials):
            length = trial_info['length']
            num_windows = max(0, (length - self.window_points) // self.step_points + 1)
            for i in range(num_windows):
                all_windows.append({
                    'trial_info_idx': trial_idx,
                    'start_point': i * self.step_points,
                    'metadata': trial_info['metadata'],
                    'dataset_name': trial_info['dataset_name'],
                    'window_index_in_trial': i
                })
        return all_windows

    def __len__(self):
        return len(self.window_index)

    def _pad_or_truncate(self, data, target_length):
        if data.shape[1] > target_length:
            return data[:, :target_length]
        elif data.shape[1] < target_length:
            padding = target_length - data.shape[1]
            return np.pad(data, ((0, 0), (0, padding)), 'constant')
        return data

    def __getitem__(self, idx):
        try:
            window_info = self.window_index[idx]
            trial_info = self.preprocessed_trials[window_info['trial_info_idx']]
            emg_full = np.load(trial_info['emg_path'], mmap_mode='r')
            imu_full = np.load(trial_info['imu_path'], mmap_mode='r')

            start = window_info['start_point']
            end = start + self.window_points

            emg_window = self._pad_or_truncate(emg_full[:, start:end], self.window_points)
            imu_window = self._pad_or_truncate(imu_full[:, start:end], self.window_points)

            # EMG补齐 mask
            max_emg_ch = self.config['data']['max_emg_channels']
            padded_emg = np.zeros((max_emg_ch, self.window_points), dtype=np.float32)
            padded_emg[:emg_window.shape[0], :] = emg_window
            emg_mask = np.zeros(max_emg_ch, dtype=bool)
            emg_mask[:emg_window.shape[0]] = True

            # IMU reshape并补齐
            ds_info = self.config['dataset_info'][window_info['dataset_name']]
            num_axes = ds_info['num_imu_axes']
            total_imu_ch = imu_window.shape[0]
            if total_imu_ch % num_axes == 0:
                num_sensors = total_imu_ch // num_axes
                imu_reshaped = imu_window.reshape(num_sensors, num_axes, self.window_points)
            else:
                imu_reshaped = imu_window.reshape(1, total_imu_ch, self.window_points)
                num_axes = total_imu_ch
                num_sensors = 1
            max_imu_sensors = self.config['data']['max_imu_sensors']
            padded_imu = np.zeros((max_imu_sensors, num_axes, self.window_points), dtype=np.float32)
            padded_imu[:num_sensors, :, :] = imu_reshaped
            imu_mask = np.zeros(max_imu_sensors, dtype=bool)
            imu_mask[:num_sensors] = True

            # 标准化（可选）
            if self.config['data'].get('normalize', True):
                for ch in range(padded_emg.shape[0]):
                    mean, std = padded_emg[ch].mean(), padded_emg[ch].std()
                    std = 1.0 if std < 1e-6 else std
                    padded_emg[ch] = (padded_emg[ch] - mean) / std
                for s in range(padded_imu.shape[0]):
                    for a in range(padded_imu.shape[1]):
                        mean, std = padded_imu[s, a].mean(), padded_imu[s, a].std()
                        std = 1.0 if std < 1e-6 else std
                        padded_imu[s, a] = (padded_imu[s, a] - mean) / std

            # 归一化到[-1,1]（可选）
            if self.config['data'].get('scale_to_unit', False):
                padded_emg = np.clip(padded_emg, -3, 3) / 3
                padded_imu = np.clip(padded_imu, -3, 3) / 3

            sample = {
                "emg": torch.from_numpy(padded_emg),
                "emg_mask": torch.from_numpy(emg_mask),
                "imu": torch.from_numpy(padded_imu),
                "imu_mask": torch.from_numpy(imu_mask),
                "metadata": {
                    "data": window_info['dataset_name'],
                    "subject_id": window_info['metadata']["subject_id"],
                    "activity_name": window_info['metadata']['activity_name'],
                    "activity_condition": window_info['metadata']['activity_condition'],
                    "trial_id": window_info['metadata']['trial_id'],
                    "window_index": window_info.get("window_index_in_trial", idx),
                    "anchor_index": idx
                }
            }

            # 根据目标模态裁剪
            if self.modality == "emg":
                sample.pop('imu', None)
                sample.pop('imu_mask', None)
            elif self.modality == "imu":
                sample.pop('emg', None)
                sample.pop('emg_mask', None)

            if self.transform:
                sample = self.transform(sample)

            return sample
        except Exception as e:
            logger.error(f"Window {idx} error: {e}", exc_info=True)
            return None