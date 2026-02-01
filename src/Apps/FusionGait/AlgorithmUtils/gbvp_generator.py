import numpy as np
from matplotlib import pyplot as plt

from src.Apps.FusionGait.AlgorithmUtils.bvp_generator import BVPGenerator, VelocityMappingParams, PhysicalSetup
from src.Apps.FusionGait.MatlabPlugins.matlab_runtime_singleton import MatlabManager
from src.Apps.FusionGait.MatlabPlugins.utils import ensure_contiguous


def angular_mean(angles, axis=None):
    """Circular mean: angles in radians."""
    s = np.sin(angles)
    c = np.cos(angles)
    mean_ang = np.arctan2(np.mean(s, axis=axis), np.mean(c, axis=axis))
    return mean_ang

def wrap_to_180_from_rad(angle_rad):
    """
    Input: angle_rad (radians)
    Output: angle_deg (degrees), normalized to [-180, 180)
    """
    angle_deg = np.degrees(angle_rad)   # convert to degrees
    return (angle_deg + 180) % 360 - 180

class GBVPGenerator(BVPGenerator):
    def __init__(self, phys_cfg: PhysicalSetup, vel_cfg: VelocityMappingParams):
        super().__init__(phys_cfg, vel_cfg)


    def _estOrientation(self, pos, dt=1.0, win=10, min_speed=0.1,
                        method='circular_median', mode='forward', plot=False, arrow_scale=0.3):
        """
        Compute headings and speeds.
        mode: 'forward' means speeds[i] = norm(pos[i] - pos[i-1]) / dt  (length N, first item is 0)
              'center'  means speeds length N-1, corresponding to pos[0]->pos[1], ... (more strict)
        Returns: headings (N,), speeds (N,), reliability (N,)
        """
        pos = np.asarray(pos)
        if pos.ndim != 2:
            raise ValueError("pos must be (2,N) or (N,2)")
        if pos.shape[0] == 2:
            xs = pos[0, :]
            ys = pos[1, :]
        else:
            xs = pos[:, 0]
            ys = pos[:, 1]

        N = len(xs)

        # use forward diffs (i: difference from i-1 -> i)
        dx = np.diff(xs, prepend=xs[0])
        dy = np.diff(ys, prepend=ys[0])

        # dt -> array
        if np.isscalar(dt):
            dts = np.full(N, dt)
        else:
            dts = np.asarray(dt)
            if dts.size == N - 1:
                dts = np.concatenate([dts, dts[-1:]])
            elif dts.size == N:
                pass
            else:
                raise ValueError("dt length mismatch")

        speeds = np.sqrt(dx * dx + dy * dy) / np.maximum(dts, 1e-12)

        # raw heading from displacement vector; when dx=dy=0 result is 0 (no movement)
        raw_head = np.arctan2(dy, dx)  # length N

        # smoothing: sliding window circular mean
        headings = np.zeros(N)
        half = int(win)
        for i in range(N):
            i0 = max(0, i - half)
            i1 = min(N - 1, i + half)
            window_angles = raw_head[i0:i1 + 1]
            # choose method: circular mean (default) or simple mean
            headings[i] = angular_mean(window_angles)

        # reliability
        reliability = speeds >= min_speed

        # if no reliable frames, fallback
        if not np.any(reliability):
            headings[:] = 0.0
            return headings, speeds, reliability

        # fill unreliable frames: forward-fill last reliable, then back-fill beginning
        last_good = None
        for i in range(N):
            if reliability[i]:
                last_good = headings[i]
            else:
                if last_good is not None:
                    headings[i] = last_good

        if not reliability[0]:
            first_good = np.where(reliability)[0][0]
            headings[:first_good] = headings[first_good]

        # optional plotting for debug
        if plot:
            plt.figure(figsize=(8, 8))
            plt.plot(xs, ys, 'o-', label='trajectory')
            for i in range(N):
                angle = headings[i]
                ux = np.cos(angle) * arrow_scale
                uy = np.sin(angle) * arrow_scale
                plt.arrow(xs[i], ys[i], ux, uy, head_width=arrow_scale * 0.35, head_length=arrow_scale * 0.35,
                          color='r')
            plt.scatter(xs[reliability], ys[reliability], c='g', label='reliable')
            plt.scatter(xs[~reliability], ys[~reliability], c='gray', label='unreliable')
            plt.axis('equal');
            plt.grid(True);
            plt.legend();
            plt.show()

        headings = wrap_to_180_from_rad(headings)

        return headings, speeds, reliability

    def _process_doppler_spectrum_parallel(self, doppler_spectrum, duration, freq_bin,  U_bound, ori_list, pos_list):
        """
        Parallel Doppler spectrum processing.
        Process via MATLAB function call.
        """

        # convert to contiguous arrays
        doppler_spectrum = ensure_contiguous(doppler_spectrum, dtype=np.float64)
        U_bound = ensure_contiguous(U_bound, dtype=np.float64)

        res =MatlabManager.call("segmentOptimizeGBVP",doppler_spectrum, pos_list, ori_list, self.vel_cfg.wave_length,
                                               self.vel_cfg.velocity_bin,
                                               freq_bin, self.phys_cfg.Tx_pos, self.phys_cfg.Rx_pos,
                                               self.phys_cfg.rx_cnt, 5, self.vel_cfg.V_bins, U_bound,
                                               self.vel_cfg.MaxFunctionEvaluations, self.vel_cfg.norm,
                                               self.vel_cfg.lambda_)


        velocity_spectrum = np.array(res, order='F')

        return velocity_spectrum


    def run(self, pos_list, doppler_spectrum, freq_bin, duration):
        """Run a full mapping pass."""
        ori, *_ = self._estOrientation(pos_list, duration)
        doppler_spectrum_max = np.max(doppler_spectrum)  # maximum over all dimensions
        U_bound = np.full((self.vel_cfg.V_bins, self.vel_cfg.V_bins), doppler_spectrum_max)  # MxM upper-bound matrix

        velocity_spectrum_ro = self._process_doppler_spectrum_parallel(
            doppler_spectrum, duration, freq_bin, U_bound=U_bound, ori_list=ori, pos_list=pos_list
        )

        #self.animate_velocity_spectrum(velocity_spectrum_ro)
        return velocity_spectrum_ro


class FusionGBVPGenerator(GBVPGenerator):
    def __init__(self, phys_cfg_dict: dict, vel_cfg_dict: dict, win_half=5):
        super().__init__(phys_cfg_dict["wifi"], vel_cfg_dict["wifi"])
        """
        phys_cfg_dict: dict, modality -> PhysicalSetup
        vel_cfg_dict: dict, modality -> VelocityMappingParams
        win_half: torso-centered velocity search half-window
        """
        self.phys_cfg_dict = phys_cfg_dict
        self.vel_cfg_dict = vel_cfg_dict
        self.win_half = win_half



    def _process_doppler_spectrum_parallel(self, modality_params: dict, pos_list, ori_list):
        """
        Call the MATLAB Fusion version of GBVP.
        doppler_spectrum_dict: dict modality -> np.ndarray (Nr x F x T)
        """
        modalities = list(modality_params.keys())

        # -----------------------
        # Prepare per-modality parameters
        # -----------------------
        doppler_spectrum_dict = {m: modality_params[m]["doppler_spectrum"] for m in modalities}
        wave_length_dict = {m: self.vel_cfg_dict[m].wave_length for m in modalities}
        freq_bin_dict = {m: modality_params[m]["freq_bin"] for m in modalities}
        Tx_pos_dict = {m: modality_params[m]["tx_loc"].T for m in modalities}
        Rx_pos_dict = {m: modality_params[m]["rx_loc"].T for m in modalities}  # transpose: MATLAB expects one Rx per row
        rx_cnt_dict = {m: modality_params[m]["rx_loc"].shape[1] for m in modalities}

        # Convert all involved dicts/arrays
        c_doppler_spectrum_dict = ensure_contiguous(doppler_spectrum_dict, dtype=np.float64)
        c_wave_length_dict = ensure_contiguous(wave_length_dict, dtype=np.float64)
        c_pos_list = ensure_contiguous(pos_list, dtype=np.float64)
        c_ori_list = ensure_contiguous(ori_list, dtype=np.float64)
        c_freq_bin_dict = ensure_contiguous(freq_bin_dict, dtype=np.float64)
        c_Tx_pos_dict = ensure_contiguous(Tx_pos_dict, dtype=np.float64)
        c_Rx_pos_dict = ensure_contiguous(Rx_pos_dict, dtype=np.float64)
        c_Rx_cnt_dict = ensure_contiguous(rx_cnt_dict, dtype=np.float64)

        first_mod = modalities[0]
        velocity_bin = self.vel_cfg_dict[first_mod].velocity_bin
        c_velocoty_bin = ensure_contiguous(velocity_bin)
        M = self.vel_cfg_dict[first_mod].V_bins

        # U_bound: global upper bound from the maximum Doppler across all modalities
        doppler_max = max(np.max(doppler_spectrum_dict[m]) for m in modalities)
        U_bound = np.full((M, M), doppler_max, dtype=np.float64)

        MaxFunctionEvaluations = self.vel_cfg_dict[first_mod].MaxFunctionEvaluations
        norm_type = self.vel_cfg_dict[first_mod].norm
        lambda_ = self.vel_cfg_dict[first_mod].lambda_

        # -----------------------
        # Call MATLAB Fusion GBVP
        # -----------------------
        res = MatlabManager.call("segmentOptimizeGBVP_Fusion",
            c_doppler_spectrum_dict,
            c_pos_list,
            c_ori_list,
            c_wave_length_dict,
            c_velocoty_bin,
            c_freq_bin_dict,
            c_Tx_pos_dict,
            c_Rx_pos_dict,
            c_Rx_cnt_dict,
            self.win_half,
            M,
            U_bound,
            MaxFunctionEvaluations,
            norm_type,
            lambda_
        )
       

        # MATLAB returns Fortran-order array; convert to NumPy
        velocity_spectrum = np.array(res, order='F')

        return velocity_spectrum


    def run(self, modality_params: dict, pos_list):
        """
        Run a full Fusion GBVP pass.
        modality_params: dict modality -> dict containing Doppler, freq_bin, init_loc, tx_loc, rx_loc, duration, etc.
        """
        modalities = list(modality_params.keys())

        # -----------------------
        # Build pos_list
        # -----------------------

        # -----------------------
        # Get duration (any modality is fine)
        # -----------------------
        duration = modality_params[modalities[0]]["duration"]

        # -----------------------
        # Estimate orientation
        # -----------------------
        ori_list, _, _ = self._estOrientation(pos_list, duration)

        # -----------------------
        # Call Fusion GBVP
        # -----------------------
        velocity_spectrum_ro = self._process_doppler_spectrum_parallel(
            modality_params, pos_list, ori_list
        )

        # Visualization
        #self.animate_velocity_spectrum(velocity_spectrum_ro)

        return velocity_spectrum_ro
