import matplotlib.pyplot as plt
import numpy as np

from scipy.signal import correlate
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, hilbert

def smooth(x, win=20):
    # Simple moving average with reflect padding
    if win <= 1:
        return x
    pad = win // 2
    xp = np.pad(x, pad, mode='reflect')
    kern = np.ones(win) / win
    y = np.convolve(xp, kern, mode='valid')
    return y

def envelope(x):
    # Envelope of the analytic signal via Hilbert transform
    return np.abs(hilbert(x))

def zscore(x):
    m = np.mean(x)
    s = np.std(x)
    if s == 0:
        return x - m
    return (x - m) / s

def xcorr_align(ref, sig, upsample=1):
    # ref, sig: 1D numpy arrays (same length; otherwise they will be zero-padded)
    # upsample: 1 for integer-sample lag; >1 for finer (interpolated) lag resolution
    n = max(len(ref), len(sig))

    # Pad to the same length
    a = np.zeros(n); b = np.zeros(n)
    a[:len(ref)] = ref
    b[:len(sig)] = sig

    if upsample == 1:
        corr = correlate(b, a, mode='full')  # lag indices: -(n-1) .. (n-1)
        lag = np.argmax(corr) - (n - 1)
        return lag
    else:
        # Coarse: integer lag
        corr = correlate(b, a, mode='full')
        lag0 = np.argmax(corr) - (n - 1)

        # Refine around lag0 by upsampling via interpolation
        interp_factor = upsample
        t = np.arange(n)
        t_up = np.linspace(0, n - 1, n * interp_factor)
        a_up = np.interp(t_up, t, a)
        b_up = np.interp(t_up, t, b)

        corr_up = correlate(b_up, a_up, mode='full')
        lag_up = np.argmax(corr_up) - (len(a_up) - 1)

        # Convert back to original-sample units (may be fractional)
        lag_refined = lag_up / interp_factor
        return lag_refined

# Align multiple sequences to the reference sequence (seqs[ref_idx] is the reference)
def align_multi_by_xcorr(seqs, ref_idx=0, upsample=4):
    ref = seqs[ref_idx]
    shifts = []
    aligned = []
    for i, s in enumerate(seqs):
        if i == ref_idx:
            shifts.append(0.0)
            aligned.append(s.copy())
            continue

        lag = xcorr_align(ref, s, upsample=upsample)
        shifts.append(lag)

        # Apply the shift:
        # A positive lag means `sig` is delayed w.r.t. `ref`, so shift it left to align.
        # Use interpolation to support fractional lags.
        t = np.arange(len(s))
        ft = interp1d(t - lag, s, kind='linear', bounds_error=False, fill_value=0.0)
        aligned_s = ft(t)
        aligned.append(aligned_s)
    return aligned, shifts

# Onset detection
def detect_onset(envel, thr_rel=0.2, min_len=5):
    lo = np.min(envel)
    hi = np.max(envel)
    thr = lo + thr_rel * (hi - lo)
    above = envel > thr

    cnt = 0
    for i, v in enumerate(above):
        if v:
            cnt += 1
            if cnt >= min_len:
                return max(0, i - cnt + 1)
        else:
            cnt = 0
    return None

# Offset detection (adaptive threshold + M consecutive "still" samples)
def detect_offset(v, K=80, M=5, scale=3.0):
    """
    v     : velocity sequence
    K     : use the first K samples to estimate noise level
    M     : declare "stop" only if M consecutive samples are below the threshold
    scale : threshold = noise * scale
    """
    K = min(K, len(v) // 3)
    noise = np.median(np.abs(v[:K]))
    th = noise * scale

    offset = 0
    N = len(v)
    for i in range(0, N - M):
        if np.max(np.abs(v[i:i + M])) > th:
            offset = i
    return offset


# =============================
#   Integrated pipeline: align + trim
# =============================
def align_by_onset_and_trim(seqs,
                            thr_rel=0.1, min_len=20,   # onset parameters
                            fill_value=0.0,
                            K=80, M=5, scale=3.0      # offset parameters
                            ):
    """
    seqs: list of 1D numpy arrays

    Returns:
        trimmed       : trimmed motion-only segments (same length for all),
                        extracted from ref_onset .. final_offset
        aligned       : aligned sequences (not trimmed)
        onsets        : onset indices for each sequence (int or None)
        offsets       : offset indices for each sequence (int, relative to aligned)
        ref_onset     : reference onset used for alignment (int)
        final_offset  : shared end point across all sequences (relative to aligned)
    """
    # ===== 1) envelope + onset detection =====
    envelopes = [smooth(np.abs(s)) for s in seqs]
    onsets = [detect_onset(env, thr_rel=thr_rel, min_len=min_len) for env in envelopes]

    valid = [o for o in onsets if o is not None]
    ref_onset = int(np.median(valid)) if len(valid) else 0

    # ===== 2) onset-based alignment =====
    aligned = []
    maxlen = 0
    for s, o in zip(seqs, onsets):
        if o is None:
            o = ref_onset

        shift = o - ref_onset
        if shift > 0:     # cut head
            new = s[shift:]
        elif shift < 0:   # pad head
            new = np.concatenate((np.full(-shift, fill_value), s))
        else:
            new = s.copy()

        aligned.append(new)
        maxlen = max(maxlen, len(new))

    # Equalize lengths (pad tail on the right)
    aligned = [
        np.pad(a, (0, maxlen - len(a)), constant_values=fill_value)
        for a in aligned
    ]

    # ===== 3) offset detection =====
    offsets = [detect_offset(a, K=K, M=M, scale=scale) for a in aligned]

    # Filter valid offsets and take the minimum as the shared end point (conservative)
    valid_offsets = [o for o in offsets if (o is not None and o > ref_onset)]
    if len(valid_offsets) == 0:
        # If no valid offset is found, fall back to the shortest aligned length
        final_offset = min(len(a) for a in aligned)
    else:
        final_offset = min(valid_offsets)

    # Ensure final_offset is at least ref_onset + 1
    if final_offset <= ref_onset:
        final_offset = min(max(ref_onset + 1, final_offset), min(len(a) for a in aligned))

    # ===== 4) Plot (envelope + onset + offset) =====
    plt.figure(figsize=(10, 5))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for k in range(len(envelopes)):
        c = colors[k % len(colors)]
        env = envelopes[k]
        plt.plot(env, label=f"Candidate {k + 1}", color=c, linewidth=2)

        # Plot onset marker (if available)
        onset = onsets[k]
        if onset is not None and 0 <= onset < len(env):
            plt.scatter(onset, env[onset], color=c, marker='o', s=60,
                        edgecolor='k', zorder=5, label=f"Onset {k+1}")

        # Plot offset marker (if available)
        off = offsets[k]
        if off is not None and 0 <= off < len(env):
            plt.scatter(off, env[off], color=c, marker='X', s=60,
                        edgecolor='k', zorder=6, label=f"Offset {k+1}")

    # Mark reference onset and shared final_offset (vertical lines)
    plt.axvline(ref_onset, color='k', linestyle='--', label='ref_onset')
    plt.axvline(final_offset, color='grey', linestyle='-.', label='final_offset')

    plt.legend()
    plt.title("Envelopes with Onset/Offset")
    plt.xlabel("Time (samples)")
    plt.ylabel("Envelope")
    plt.grid(True)
    plt.show()

    # ===== 5) Trim all sequences: [ref_onset, final_offset) =====
    trimmed = []
    for a in aligned:
        start = int(ref_onset)
        end = int(min(final_offset, len(a)))
        if start >= end:
            # Guard: if boundaries are invalid, return an empty array
            trimmed.append(np.array([], dtype=a.dtype))
        else:
            trimmed.append(a[start:end])

    return trimmed, aligned, onsets, offsets, ref_onset, final_offset



from numpy.fft import fft, ifft
def gcc_phat(x, y, fs=1.0, interp=1):
    """
    Estimate time delay (in samples; may be fractional).
    Positive means y is delayed with respect to x.
    """
    n = x.size + y.size
    X = fft(x, n)
    Y = fft(y, n)
    R = X * np.conj(Y)
    R /= (np.abs(R) + 1e-12)  # PHAT weighting
    cc = np.real(ifft(R))
    max_shift = int(n / 2)
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))
    shift = np.argmax(cc) - max_shift
    return shift / interp

def align_with_gcc(seqs, ref_idx=0, upsample=1):
    ref = seqs[ref_idx]
    aligned = []
    shifts = []
    for i, s in enumerate(seqs):
        if i == ref_idx:
            aligned.append(ref.copy())
            shifts.append(0.0)
            continue

        # Optionally pre-filter / take envelope to emphasize movement before GCC-PHAT
        lag = gcc_phat(ref, s, interp=upsample)
        shifts.append(lag)

        # Fractional shift via interpolation
        t = np.arange(len(s))
        ft = interp1d(t - lag, s, kind='linear', bounds_error=False, fill_value=0.0)
        aligned.append(ft(t))
    return aligned, shifts


def find_plcr_from_spectrogram(spectrogram, fre_to_speed, plcr_ratio):
    # 1) Downsample along the time axis (optional)
    if plcr_ratio != 1:
        group_size = plcr_ratio
        n_groups = int(np.ceil(spectrogram.shape[1] / group_size))
        spectrogram_dec = np.zeros((spectrogram.shape[0], n_groups))
        valid_mask = np.zeros(n_groups, dtype=bool)

        for ii in range(n_groups):
            start = ii * group_size
            end = min((ii + 1) * group_size, spectrogram.shape[1])
            block = spectrogram[:, start:end]
            spectrogram_dec[:, ii] = np.mean(block, axis=1)
            valid_mask[ii] = block.sum() > 0
    else:
        # No downsampling
        spectrogram_dec = spectrogram.copy()
        valid_mask = np.any(spectrogram_dec > 0, axis=0)
        n_groups = spectrogram_dec.shape[1]

    # 2) Dynamic programming (find the maximum-energy smooth frequency track)
    pow_acc = np.zeros_like(spectrogram_dec)
    last_freq = np.zeros_like(spectrogram_dec, dtype=int)

    for ii in range(n_groups):
        if not valid_mask[ii]:
            # Invalid frame -> carry over the previous accumulated power (no DP update)
            if ii > 0:
                pow_acc[:, ii] = pow_acc[:, ii - 1]
            continue

        if ii == 0:
            pow_acc[:, ii] = spectrogram_dec[:, ii]
        else:
            for jj in range(spectrogram_dec.shape[0]):
                if jj == 0:
                    neighbors = slice(jj, jj + 2)
                elif jj == spectrogram_dec.shape[0] - 1:
                    neighbors = slice(jj - 1, jj + 1)
                else:
                    neighbors = slice(jj - 1, jj + 2)

                max_idx = np.argmax(pow_acc[neighbors, ii - 1])
                last_freq[jj, ii] = neighbors.start + max_idx
                pow_acc[jj, ii] = pow_acc[last_freq[jj, ii], ii - 1] + spectrogram_dec[jj, ii]

    # 3) Backtracking
    f = np.zeros(n_groups, dtype=int)
    last_valid = np.where(valid_mask)[0][-1]
    f[last_valid] = np.argmax(pow_acc[:, last_valid])

    for ii in range(last_valid, 0, -1):
        if valid_mask[ii]:
            f[ii - 1] = last_freq[f[ii], ii]
        else:
            f[ii] = f[ii + 1]  # Invalid frame -> fill with the next valid frame

    # 4) Map frequency indices to PLCR (speed)
    plcr = fre_to_speed[f]
    plcr[~valid_mask] = 0     # Set speed to 0 for invalid frames
    return plcr, f


def find_plcr_from_spectrogram_and_plot(spectrogram, fre_to_speed, plcr_ratio, sample_rate):
    # Compute PLCR using the original function
    plcr, f_idx = find_plcr_from_spectrogram(spectrogram, fre_to_speed, plcr_ratio)

    # ---------- Plot ----------
    n_freq, n_time = spectrogram.shape
    time_axis = np.arange(n_time) / sample_rate  # Time axis in seconds

    plt.figure(figsize=(12, 5))

    # 1) Plot Doppler spectrum (shown in speed units after mapping)
    plt.imshow(
        spectrogram,
        aspect='auto',
        origin='lower',
        extent=[time_axis[0], time_axis[-1], fre_to_speed[0], fre_to_speed[-1]],
        cmap='viridis'
    )
    plt.colorbar(label='Magnitude')
    plt.xlabel('Time [s]')
    plt.ylabel('Speed [m/s] (mapped from frequency)')

    # 2) Plot PLCR curve
    # PLCR corresponds to one speed value per (downsampled) frame.
    if plcr_ratio != 1:
        plcr_time_axis = np.arange(len(plcr)) * plcr_ratio / sample_rate
    else:
        plcr_time_axis = time_axis

    plt.plot(plcr_time_axis, plcr, color='r', linewidth=2, label='PLCR')
    plt.legend()
    plt.title('Doppler Spectrum and PLCR Curve')
    plt.tight_layout()
    plt.show()

    return plcr, f_idx

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

def smooth_vel(x, method="moving", window=20, polyorder=9, sigma=1.0):
    """
    1D signal smoothing utility.

    Parameters
    ----------
    x : array-like
        Input signal
    method : str
        "moving"   - moving average
        "gaussian" - Gaussian smoothing
        "savgol"   - Savitzky-Golay filter
    window : int
        Window size (should be odd for Savitzky-Golay)
    polyorder : int
        Polynomial order for Savitzky-Golay
    sigma : float
        Sigma for the Gaussian kernel
    """
    x = np.asarray(x)

    if method == "moving":
        if window < 1:
            return x
        kernel = np.ones(window) / window
        return np.convolve(x, kernel, mode='same')

    elif method == "gaussian":
        return gaussian_filter1d(x, sigma=sigma)

    elif method == "savgol":
        if window % 2 == 0:
            window += 1
        return savgol_filter(x, window_length=window, polyorder=polyorder)

    else:
        raise ValueError("Unknown method.")
