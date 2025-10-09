"""
原始数据集类
"""
import os
import warnings
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class KneePAD(Dataset):
    """KneePAD 数据集加载器"""

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.index_df = self._build_index()

    def _build_index(self):
        records = []
        class_map = {
            0: ("Squat", "Correct"), 1: ("Squat_WT", "Wrong"),
            2: ("Squat_FL", "Wrong"), 3: ("Extension", "Correct"),
            4: ("Extension_NF", "Wrong"), 5: ("Extension_LL", "Wrong"),
            6: ("Gait", "Correct"), 7: ("Gait_NF", "Wrong"), 8: ("Gait_HA", "Wrong")
        }

        for subj_folder in sorted(os.listdir(self.root_dir)):
            if not subj_folder.startswith("Subject_"):
                continue
            subj_path = os.path.join(self.root_dir, subj_folder)

            for exercise_id in sorted(os.listdir(subj_path)):
                exercise_path = os.path.join(subj_path, exercise_id)
                if not os.path.isdir(exercise_path):
                    continue
                exercise_id = int(exercise_id)
                if exercise_id not in class_map:
                    continue

                exercise_name, activity_condition = class_map[exercise_id]

                for trial_folder in sorted(os.listdir(exercise_path)):
                    trial_path = os.path.join(exercise_path, trial_folder)
                    if not os.path.isdir(trial_path):
                        continue
                    emg_file = os.path.join(trial_path, "emg.npy")
                    imu_file = os.path.join(trial_path, "imu.npy")
                    if not (os.path.exists(emg_file) and os.path.exists(imu_file)):
                        continue
                    records.append({
                        "subject_id": subj_folder,
                        "activity_name": exercise_name,
                        "activity_condition": activity_condition,
                        "trial_id": trial_folder,
                        "emg_path": emg_file,
                        "imu_path": imu_file
                    })
        return pd.DataFrame(records)

    def __len__(self):
        return len(self.index_df)

    def load_raw_data(self, idx):
        row = self.index_df.iloc[idx]
        try:
            emg = np.load(row["emg_path"])
            imu = np.load(row["imu_path"])
            return emg, imu, row
        except Exception as e:
            warnings.warn(f"Error loading files: {e}")
            return None, None, row


class MovePort(Dataset):
    """MovePort 数据集加载器"""

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.index_df = self._build_index()

    def _build_index(self):
        records = []
        condition_map = {
            "treadmill_normal": "Normal",
            "still": "Normal",
            "ground_gait": "Normal",
            "treadmill_dragging": "Abnormal",
            "treadmill_leghigh": "Abnormal",
            "forward": "Abnormal",
            "back": "Abnormal",
            "halfsquat": "Abnormal"
        }
        for subj_folder in sorted(os.listdir(self.root_dir)):
            if not subj_folder.isdigit():
                continue
            subj_path = os.path.join(self.root_dir, subj_folder)
            for activity_name in sorted(os.listdir(subj_path)):
                act_path = os.path.join(subj_path, activity_name)
                if not os.path.isdir(act_path):
                    continue
                activity_condition = condition_map.get(activity_name, "Unknown")
                for file_name in sorted(os.listdir(act_path)):
                    if not file_name.endswith(".csv"):
                        continue
                    parts = file_name.split("_")
                    if len(parts) < 2:
                        continue
                    modality = parts[0]
                    segment_id = parts[1].replace(".csv", "")
                    file_path = os.path.join(act_path, file_name)
                    if modality == "emg":
                        imu_file = os.path.join(act_path, f"imu_{segment_id}.csv")
                        if not os.path.exists(imu_file):
                            continue
                        records.append({
                            "subject_id": subj_folder,
                            "activity_name": activity_name,
                            "activity_condition": activity_condition,
                            "trial_id": segment_id,
                            "emg_path": file_path,
                            "imu_path": imu_file
                        })
        return pd.DataFrame(records)

    def _load_emg_csv_safe(self, file_path):
        try:
            df = pd.read_csv(file_path, header=None)
            if df.empty or df.shape[1] < 2:
                warnings.warn(f"Invalid EMG file: {file_path}")
                return None
            return df.iloc[1:, 1:].to_numpy(dtype=np.float32)
        except Exception as e:
            warnings.warn(f"Error reading EMG: {e}")
            return None

    def _load_imu_csv_safe(self, file_path):
        try:
            df = pd.read_csv(file_path, header=None)
            if df.empty or df.shape[1] < 2:
                warnings.warn(f"Invalid IMU file: {file_path}")
                return None
            channel_names = df.iloc[:, 0].astype(str).tolist()
            data_values = df.iloc[:, 1:].to_numpy(dtype=np.float32)
            mask = [(("_Acc_" in ch) or ("_Gyr_" in ch)) for ch in channel_names]
            if not any(mask):
                return None
            return data_values[mask, :]
        except Exception as e:
            warnings.warn(f"Error reading IMU: {e}")
            return None

    def __len__(self):
        return len(self.index_df)

    def load_raw_data(self, idx):
        row = self.index_df.iloc[idx]
        emg = self._load_emg_csv_safe(row["emg_path"])
        imu = self._load_imu_csv_safe(row["imu_path"])
        return emg, imu, row