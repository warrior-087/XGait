# preprocess.py
import numpy as np


def preprocess_doppler(arr: np.ndarray, freq_down=1, time_down=1, normalize=True, merge_ncnt=True):
    if arr.ndim != 3:
        raise ValueError("expect arr shape (ncnt, freq, time)")

    # 1. Mean downsampling
    # Use reshape + mean to perform block-averaging downsampling
    ncnt, f, t = arr.shape

    if freq_down > 1:
        # Truncate the remainder that is not divisible
        f_new = f // freq_down
        arr = arr[:, :f_new * freq_down, :]
        arr = arr.reshape(ncnt, f_new, freq_down, t).mean(axis=2)

    if time_down > 1:
        ncnt, f_new, t = arr.shape
        t_new = t // time_down
        arr = arr[:, :, :t_new * time_down]
        arr = arr.reshape(ncnt, f_new, t_new, time_down).mean(axis=3)

    # 2. Convert to dB scale (log-scale)
    # Assume the input is a power/energy spectrum; avoid log(0)
    arr = np.abs(arr)  # Ensure positive values
    arr = 10 * np.log10(arr + 1e-6)

    # 3. Basic processing: centering and standardization
    # Standardizing in dB scale is often more helpful for model learning
    arr = arr - np.nanmean(arr)
    if normalize:
        std = np.nanstd(arr)
        if std > 1e-6:
            arr = arr / std

    # 4. Merge channels (ncnt -> 1)
    if merge_ncnt:
        arr = np.nanmean(arr, axis=0)  # Shape becomes (F, T)

    # 5. Dimension alignment and format conversion
    # Return shape: (Channel, Time, Frequency)
    if arr.ndim == 2:  # (F, T)
        return arr.T[np.newaxis, :, :].astype(np.float32)
    else:  # (ncnt, F, T)
        return np.transpose(arr, (0, 2, 1)).astype(np.float32)


def preprocess(arr: np.ndarray, direc_down=10,  normalize=True):
    if arr.ndim != 3:
        raise ValueError("expect arr shape (ncnt, freq, time)")

    # 1. Basic processing: downsampling and standardization (keep original logic)
    if direc_down > 1:
        arr = arr[::direc_down, :, :]

    arr = arr - np.nanmean(arr)
    if normalize:
        std = np.nanstd(arr)
        if std > 1e-6:
            arr = arr / std

    # Axis permutation (direction, velocity, time) -> (direction, time, velocity)
    return np.transpose(arr, (0, 2, 1)).astype(np.float32)
