"""
信号滤波器
"""
from scipy.signal import butter, filtfilt


class SignalFilters:
    """EMG和IMU滤波工具"""

    @staticmethod
    def emg_filter(data, fs, low_cut=20, high_cut=450, notch_freq=50, notch_quality=30):
        filtered = data.copy()
        for ch in range(data.shape[0]):
            sig = data[ch]
            if notch_freq > 0:
                b_notch, a_notch = SignalFilters._design_notch_filter(notch_freq, fs, notch_quality)
                sig = filtfilt(b_notch, a_notch, sig)
            b_band, a_band = SignalFilters._design_bandpass_filter(low_cut, high_cut, fs)
            sig = filtfilt(b_band, a_band, sig)
            filtered[ch] = sig
        return filtered

    @staticmethod
    def imu_filter(data, fs, low_cut=0.1, high_cut=20, filter_type='bandpass'):
        filtered = data.copy()
        for ch in range(data.shape[0]):
            sig = data[ch]
            if filter_type == 'bandpass':
                b, a = SignalFilters._design_bandpass_filter(low_cut, high_cut, fs)
            elif filter_type == 'lowpass':
                b, a = SignalFilters._design_lowpass_filter(high_cut, fs)
            elif filter_type == 'highpass':
                b, a = SignalFilters._design_highpass_filter(low_cut, fs)
            else:
                continue
            sig = filtfilt(b, a, sig)
            filtered[ch] = sig
        return filtered

    @staticmethod
    def _design_bandpass_filter(low, high, fs, order=4):
        nyq = fs / 2
        low = max(low / nyq, 0.001)
        high = min(high / nyq, 0.999)
        return butter(order, [low, high], btype='band')

    @staticmethod
    def _design_lowpass_filter(high, fs, order=4):
        nyq = fs / 2
        high = min(high / nyq, 0.999)
        return butter(order, high, btype='low')

    @staticmethod
    def _design_highpass_filter(low, fs, order=4):
        nyq = fs / 2
        low = max(low / nyq, 0.001)
        return butter(order, low, btype='high')

    @staticmethod
    def _design_notch_filter(freq, fs, quality=30):
        nyq = fs / 2
        freq_norm = freq / nyq
        bw = freq / quality
        bw_norm = bw / nyq
        return butter(2, [freq_norm - bw_norm / 2, freq_norm + bw_norm / 2], btype='bandstop')
