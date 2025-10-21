"""
数据验证与修复
"""
import warnings
import numpy as np


class DataValidator:
    """用于验证和修复 EMG/IMU 数据"""

    def __init__(self, min_length=100, variance_threshold=1e-10):
        self.min_length = min_length
        self.variance_threshold = variance_threshold

    def validate_and_fix_data(self, data, data_type="emg"):
        """
        验证并修复数据
        Args:
            data: np.ndarray (channels, time_points)
            data_type: "emg" 或 "imu"
        Returns:
            (processed_data, is_valid)
        """
        if data is None or data.size == 0:
            warnings.warn(f"Empty {data_type} data detected")
            return None, False

        # 保证二维
        data = self._ensure_2d(data)
        if data.shape[1] < self.min_length:
            warnings.warn(f"Too short {data_type}: {data.shape[1]} < {self.min_length}")
            return None, False

        data = self._handle_missing_values(data, data_type)

        if self._is_invalid_signal(data):
            warnings.warn(f"Invalid {data_type} signal detected (constant or zero)")
            return None, False

        return data, True

    def _ensure_2d(self, data):
        if data.ndim == 1:
            return data.reshape(1, -1)
        elif data.ndim > 2:
            warnings.warn(f"Unexpected data dimensions: {data.shape}")
            return np.atleast_2d(data.squeeze())
        return data

    def _handle_missing_values(self, data, data_type):
        if np.isnan(data).any():
            warnings.warn(f"NaN values found in {data_type} data, interpolating...")
            data = self._interpolate_nan(data)
        if np.isinf(data).any():
            warnings.warn(f"Infinite values found in {data_type} data, clipping...")
            data = np.clip(data, -1e6, 1e6)
        return data

    def _interpolate_nan(self, data):
        for ch in range(data.shape[0]):
            mask = ~np.isnan(data[ch])
            if mask.sum() > 1:
                idx = np.arange(data.shape[1])
                data[ch] = np.interp(idx, idx[mask], data[ch][mask])
            else:
                data[ch] = 0.0
        return data

    def _is_invalid_signal(self, data):
        return any(np.var(data[ch]) < self.variance_threshold for ch in range(data.shape[0]))