import numpy as np
from scipy import signal
from typing import Dict, Optional, Tuple, List

from src.Apps.FusionGait.MatlabPlugins.utils import ensure_contiguous


class FusionPreprocessor:
    def __init__(
            self,
            reference_modality: str = "wifi",
            time_align_mode: str = "correlation",  # options: "correlation" or "min"
            motion_threshold: float = 0.15,
            padding: int = 5,
            rx_policy: Optional[Dict[str, List[int]]] = None,
            enable_sanity_check: bool = True,
    ):
        self.reference_modality = reference_modality
        self.time_align_mode = time_align_mode
        self.motion_threshold = motion_threshold
        self.padding = padding
        self.rx_policy = rx_policy or {}
        self.enable_sanity_check = enable_sanity_check

    def process(self, modality_params: Dict[str, dict]) -> Dict[str, dict]:
        """Main entry point: choose the alignment strategy based on `time_align_mode`."""
        if self.reference_modality not in modality_params:
            raise ValueError(f"Reference '{self.reference_modality}' not found!")

        # --- STEP 1: Choose the alignment strategy ---
        if self.time_align_mode == "correlation":
            # Strategy A: cross-correlation alignment (shift + pad/crop)
            modality_params = self._align_by_correlation(modality_params)
        elif self.time_align_mode == "min":
            # Strategy B: min-length alignment (crop all modalities to the shortest PLCR length)
            modality_params = self._align_by_min_length(modality_params)
        else:
            # No global alignment; proceed to downstream processing as-is
            pass

        # --- STEP 2: Detect motion window on the reference modality ---
        ref_plcr = modality_params[self.reference_modality]["plcr"]
        motion_start, motion_end = self._detect_motion_window(ref_plcr)

        # --- STEP 3: Apply unified cropping to all modalities ---
        for m in modality_params:
            modality_params[m] = self._unified_crop(modality_params[m], motion_start, motion_end)

        # --- STEP 3.5: Force Doppler length alignment (fix off-by-1 caused by float rounding) ---
        modality_params = self._force_align_doppler(modality_params)

        # --- STEP 4: Receiver selection and final checks ---
        modality_params = self._apply_rx_selection(modality_params)
        if self.enable_sanity_check:
            self._sanity_check(modality_params)

        return modality_params

    # ============================================================
    # Strategy A: Cross-correlation alignment
    # ============================================================

    def _align_by_correlation(self, modality_params: Dict[str, dict]) -> Dict[str, dict]:
        ref_params = modality_params[self.reference_modality]
        ref_env = self._get_envelope(ref_params["plcr"])
        ref_T = len(ref_env)

        new_modality_params = {}
        for m, params in modality_params.items():
            if m == self.reference_modality:
                new_modality_params[m] = params
                continue

            target_env = self._get_envelope(params["plcr"])
            corr = signal.correlate(target_env - np.mean(target_env),
                                    ref_env - np.mean(ref_env), mode='full')
            lags = signal.correlation_lags(len(target_env), len(ref_env), mode='full')
            best_lag = lags[np.argmax(corr)]

            new_modality_params[m] = self._shift_to_reference(params, best_lag, ref_T)
        return new_modality_params

    # ============================================================
    # Core fix: force a minimum-length alignment for Doppler spectra
    # ============================================================
    def _force_align_doppler(self, modality_params: Dict[str, dict]) -> Dict[str, dict]:
        """Find the minimum `doppler_spectrum` time length across modalities and truncate all to it."""
        d_lengths = [
            p["doppler_spectrum"].shape[-1]
            for p in modality_params.values() if "doppler_spectrum" in p
        ]

        if d_lengths:
            min_d_len = min(d_lengths)
            for m in modality_params:
                if "doppler_spectrum" in modality_params[m]:
                    # Truncate to the same minimum length to ensure shapes match exactly
                    modality_params[m]["doppler_spectrum"] = \
                        modality_params[m]["doppler_spectrum"][..., :min_d_len]
        return modality_params

    # ============================================================
    # Strategy B: Min-length alignment
    # ============================================================
    def _align_by_min_length(self, modality_params: Dict[str, dict]) -> Dict[str, dict]:
        """Find the minimum PLCR length across modalities and crop all modalities to that length."""
        # Time-axis lengths of PLCR for all modalities
        lengths = [p["plcr"].shape[-1] for p in modality_params.values()]
        min_len = min(lengths)

        # Crop all to [0, min_len)
        for m in modality_params:
            modality_params[m] = self._unified_crop(modality_params[m], 0, min_len)
        return modality_params

    # ============================================================
    # Common helper methods (keep original logic)
    # ============================================================

    def _shift_to_reference(self, params: dict, lag: int, ref_T: int) -> dict:
        orig_T = params["plcr"].shape[-1]

        def do_shift(arr, current_lag, target_T, ratio):
            # Convert lag/target_T into the array's own time resolution (ratio may be fractional)
            r_lag = int(round(current_lag * ratio))
            r_target_T = int(round(target_T * ratio))

            new_shape = list(arr.shape)
            new_shape[-1] = r_target_T
            new_arr = np.zeros(new_shape, dtype=arr.dtype)

            # src_start: start index in source array
            # dst_start: start index in destination array
            src_start, dst_start = max(0, -r_lag), max(0, r_lag)

            # Copy length limited by both source and destination available ranges
            length = min(arr.shape[-1] - src_start, r_target_T - dst_start)
            if length > 0:
                slc = lambda s, l: [slice(None)] * (arr.ndim - 1) + [slice(s, s + l)]
                new_arr[tuple(slc(dst_start, length))] = arr[tuple(slc(src_start, length))]
            return new_arr

        # Shift time-dependent fields at PLCR resolution
        for key in ["plcr", "direc", "duration"]:
            if key in params:
                params[key] = do_shift(params[key], lag, ref_T, 1.0)

        # Shift Doppler spectrum at its own resolution (may have different T due to STFT/windowing)
        if "doppler_spectrum" in params:
            ratio = params["doppler_spectrum"].shape[-1] / orig_T
            params["doppler_spectrum"] = do_shift(params["doppler_spectrum"], lag, ref_T, ratio)

        return params

    def _unified_crop(self, params: dict, start: int, end: int) -> dict:
        """Crop all supported fields in a modality dict to the time range [start, end)."""
        p_len = params["plcr"].shape[-1]

        # Crop time-dependent fields at PLCR resolution
        for key in ["plcr", "direc", "duration"]:
            if key in params:
                params[key] = params[key][..., start:end]

        # Crop Doppler spectrum using a proportional mapping from PLCR indices to Doppler indices
        if "doppler_spectrum" in params:
            ratio = params["doppler_spectrum"].shape[-1] / (p_len if p_len > 0 else 1)
            ds, de = int(round(start * ratio)), int(round(end * ratio))
            params["doppler_spectrum"] = params["doppler_spectrum"][..., ds:de]

        return params

    def _detect_motion_window(self, plcr: np.ndarray) -> Tuple[int, int]:
        env = self._get_envelope(plcr)
        env_smooth = np.convolve(env, np.ones(5) / 5, mode='same')
        max_val = np.max(env_smooth)
        if max_val == 0:
            return 0, len(env)

        # Active region: envelope above a fraction of the peak value
        active = np.where(env_smooth > max_val * self.motion_threshold)[0]
        if len(active) == 0:
            return 0, len(env)

        # Add padding around the detected active region
        return int(max(0, active[0] - self.padding)), int(min(len(env), active[-1] + self.padding))

    def _get_envelope(self, plcr: np.ndarray) -> np.ndarray:
        # If plcr is multi-link (N x T), use RMS over links; otherwise use absolute value
        return np.sqrt(np.mean(np.square(plcr), axis=0)) if plcr.ndim > 1 else np.abs(plcr)

    def _apply_rx_selection(self, modality_params: Dict[str, dict]) -> Dict[str, dict]:
        for mod, policy in self.rx_policy.items():
            if mod not in modality_params:
                continue

            indices = policy.get("indices", [])
            if not indices:
                continue

            p = modality_params[mod]

            # --- Robustness check: ensure indices are in bounds ---
            # Use PLCR as the reference to determine the receiver/link count for this modality.
            actual_rx_count = p["plcr"].shape[0] if "plcr" in p else 0
            if actual_rx_count == 0:
                continue

            # Filter out invalid indices (out of range or negative)
            valid_idx = [i for i in indices if 0 <= i < actual_rx_count]

            if len(valid_idx) != len(indices):
                # Warn if some indices are dropped
                print(
                    f"[Warning] Modality '{mod}': Some indices {set(indices) - set(valid_idx)} "
                    f"are out of bounds (Max: {actual_rx_count - 1})."
                )

            if not valid_idx:
                continue

            # --- Apply selection ---
            # rx_loc is typically shaped [2, N_rx]
            if "rx_loc" in p:
                # Ensure rx_loc has enough columns
                if p["rx_loc"].shape[1] >= actual_rx_count:
                    p["rx_loc"] = p["rx_loc"][:, valid_idx]

            # Core fields are typically shaped [N_rx, ...]
            for key in ["plcr", "doppler_spectrum", "direc"]:
                if key in p:
                    p[key] = ensure_contiguous(p[key][valid_idx, ...])

        return modality_params

    def _sanity_check(self, modality_params: Dict[str, dict]):
        lengths = [p["plcr"].shape[-1] for p in modality_params.values()]
        if len(set(lengths)) > 1:
            raise AssertionError(f"Final alignment failed! Lengths: {lengths}")
