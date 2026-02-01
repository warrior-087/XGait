import os
from scipy import signal
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import decimate


from src.Apps.FusionGait.ConfigUtils.track_config import GroundTruthTrajectory
from src.Apps.FusionGait.GaitRecognition.Preprocessing.calc_velocity import find_plcr_from_spectrogram, \
    smooth_vel
from src.Apps.FusionGait.GaitRecognition.Preprocessing.fusion_preprocesor import FusionPreprocessor
from src.Apps.FusionGait.MatlabPlugins.matlab_runtime_singleton import MatlabManager
from src.Apps.FusionGait.MatlabPlugins.utils import ensure_contiguous

class PositionTracker:
    def __init__(self, trackConfig,  groundTruth:GroundTruthTrajectory):
        self.trackConfig = trackConfig
        self.ground_truth = groundTruth

    def _generateDFS(self, dfss_generator, path_pfx, traj_id, rep_id, user = None, modality=None, base_dir=None):
        if user is not None and modality is not None and base_dir is not None:
            # Load doppler_spectrum from cache
            is_load_success = user.data.load_base_data(base_dir, modality)
            if is_load_success:
                doppler_spectrum, freq_bin, locations, speeds, duration = \
                    user.data.get_base_data(modality).get_data_by_id(traj_id, rep_id)
                if doppler_spectrum is not None:
                    time_bin = np.arange(doppler_spectrum.shape[2]) * (1 / self.trackConfig.sample_rate)  # Generate the time axis based on the sampling rate
                else:
                    # Generate Doppler spectrum
                    doppler_spectrum, freq_bin, time_bin = dfss_generator.generate(path_pfx)
                    freq_bin = np.asarray(freq_bin).ravel()
                    time_bin = np.asarray(time_bin).ravel()
            else:
                # Generate Doppler spectrum
                doppler_spectrum, freq_bin, time_bin = dfss_generator.generate(path_pfx)
                freq_bin = np.asarray(freq_bin).ravel()
                time_bin = np.asarray(time_bin).ravel()
        else:
            # Generate Doppler spectrum
            doppler_spectrum, freq_bin, time_bin = self.dfss_generator.generate(path_pfx)
            freq_bin = np.asarray(freq_bin).ravel()
            time_bin = np.asarray(time_bin).ravel()

        return doppler_spectrum, freq_bin, time_bin

    def __velocity_estimation(self, dopplerSpectrum, freq_bin, is_downsample = True):
        if is_downsample:
            plcr_ratio = int(self.trackConfig.sample_rate / self.trackConfig.plcr_rate)
        else :
            plcr_ratio = 1

        freq_to_speed = freq_bin * self.trackConfig.wave_length
        plcrs = []

        for i, slice_i in enumerate(dopplerSpectrum):
            plcr, f = find_plcr_from_spectrogram(slice_i, freq_to_speed, plcr_ratio)

            plcr_smooth = smooth_vel(plcr,"gaussian")
            plcrs.append(plcr_smooth)


        return np.array(plcrs)  # Convert to a numpy array


    def __optimize_velocity(self, vs: np.ndarray,max_acc: float = 6.4):
        """
        Optimize multiple velocity curves.

        Args:
            vs: np.ndarray, shape (2, N) or (2, N, M), x/y velocities for each curve
            max_acc: maximum acceleration (limit on per-step velocity change)

        Returns:
            vs_opt: optimized velocities
        """
        vs_opt = vs.copy()

        # 1. Smooth velocities
        window_size = int(self.trackConfig.plcr_rate * 4)
        vs_opt = signal.savgol_filter(vs_opt, window_size, 2, mode='nearest')

        # 2. Acceleration constraint
        dv = np.diff(vs_opt, axis=0)
        dv[dv > max_acc] = max_acc
        dv[dv < -max_acc] = -max_acc
        vs_opt[1:] = vs_opt[0] + np.cumsum(dv)

        return vs_opt

    def __direction_estimation(self,plcrs):
        # todo...
        directions = -np.sign(plcrs) ## Some spectra are symmetric, so this may be judged incorrectly

        return np.array(directions)

    def __calc_params(self, dopplerSpectrum, freq_bin, time_bin, is_downsample = True):
        if is_downsample:
            # Compute the tracking downsampling ratio
            track_ratio = int(self.trackConfig.plcr_rate / self.trackConfig.track_rate)
        else:
            track_ratio = 1

        plcrs = self.__velocity_estimation(dopplerSpectrum, freq_bin)
        direcs = self.__direction_estimation(plcrs)
        plcrs = np.abs(plcrs) # What is passed to the tracking function is the absolute PLCR value

        # Downsample the PLCR signal
        plcr_dec = np.zeros((plcrs.shape[0], int(np.ceil(plcrs.shape[1] / track_ratio))))
        if is_downsample == False:
            plcr_dec = plcrs
            direc_dec = direcs
            duration = np.full_like(time_bin, time_bin[1] - time_bin[0])
        else:
            for ii in range(plcrs.shape[0]):
                plcr_dec[ii, :] = decimate(plcrs[ii, :], track_ratio, ftype='fir', zero_phase=True)

            direc_dec = direcs[:, ::track_ratio] # Downsample the direction signal synchronously (simple slicing)
            duration = np.ones(plcr_dec.shape[1]) * (1 / self.trackConfig.track_rate) # Corresponding time-interval vector

        # plt.figure(figsize=(8, 6))
        # for k in range(len(plcrs)):
        #     plt.plot(plcrs[k], label=f"Candidate {k + 1}")
        #
        # plt.xlabel("Slice index")
        # plt.ylabel("PLCR value")
        # plt.title("PLCR Curves")
        # plt.legend()
        # plt.grid(True)
        # plt.show()

        # Prepare to call MATLAB functions: first convert parameters to contiguous arrays
        plcr_dec = ensure_contiguous(plcr_dec, dtype=float)
        direc_dec = ensure_contiguous(direc_dec, dtype=float)
        duration = ensure_contiguous(duration, dtype=float)

        return plcr_dec, direc_dec, duration


    def plot_tracking(self, ls, ground, start, end, tx_loc=None, rx_loc=None):
        """
        Plot the estimated trajectory and the ground-truth straight-line path, and mark TX/RX positions.

        Args:
            ls: ndarray, shape=[2, N], estimated trajectory [x, y]
            start: tuple(float, float), line start point (x0, y0)
            end: tuple(float, float), line end point (x1, y1)
            tx_loc: tuple(float, float), optional, transmitter position
            rx_loc: tuple(float, float), optional, receiver position
        """
        plt.figure()

        # Plot the ground-truth straight-line trajectory
        # line_x = np.linspace(start[0], end[0], 100)
        # line_y = np.linspace(start[1], end[1], 100)
        if ground.shape[1] == 2:
            ground = ground.T
        plt.plot(ground[0,:], ground[1, :], '--', color=[0.5, 0.5, 0.5], label='Groundtruth')

        # Plot the estimated trajectory
        plt.plot(ls[0, :], ls[1, :], 'b-*', label='Tracking Result')

        # Mark the start point
        plt.plot(ls[0, 0], ls[1, 0], 'rx', label='Start')

        # Mark TX/RX
        if tx_loc is not None:
            plt.plot(tx_loc[0], tx_loc[1], 'gs', label='TX')
        if rx_loc is not None:
            plt.plot(rx_loc[0], rx_loc[1], 'm^', label='RX')

        # Set axis ranges
        xlim_d = min(-1, np.min(ls[0, :]), start[0], end[0])
        xlim_u = max(6, np.max(ls[0, :]), start[0], end[0])
        ylim_d = min(-1, np.min(ls[1, :]), start[1], end[1])
        ylim_u = max(6, np.max(ls[1, :]), start[1], end[1])
        plt.xlim([xlim_d, xlim_u])
        plt.ylim([ylim_d, ylim_u])

        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.legend(loc='upper right')
        plt.gcf().set_size_inches(5, 5)
        plt.show()


    def remove_nan_receivers(self,dopplerSpectrum, rx_loc):
        """
        Remove receivers that contain NaNs in doppler_spectrum, and remove the corresponding rx_loc entries.

        Args:
            doppler_spectrum: ndarray, shape (n_rx, n_freq, n_time)
            rx_loc: ndarray, shape (2, n_rx) or (n_rx, 2)

        Returns:
            doppler_spectrum_clean: spectrum after removing NaN receivers
            rx_loc_clean: receiver locations after removal
            removed_idx: indices of removed receivers
        """

        # ---- Step 0: reshape rx_loc to (2, n_rx) ----
        if rx_loc.shape[0] != 2 and rx_loc.shape[1] == 2:
            rx_loc = rx_loc.T  # Convert to (2, n_rx)

        if rx_loc.shape[0] != 2:
            raise ValueError("Invalid rx_loc shape; expected (2, n_rx) or (n_rx, 2)")

        # ---- Step 1: find receivers that contain NaNs ----
        nan_rx = np.isnan(dopplerSpectrum).any(axis=(1, 2))
        removed_idx = np.where(nan_rx)[0]

        if removed_idx.size > 0:
            print(f"[Warning] The following receivers contain NaNs and will be removed: {removed_idx}")

            # ---- Step 2: remove corresponding receivers in doppler_spectrum ----
            doppler_spectrum_clean = dopplerSpectrum[~nan_rx, :, :]
            dopplerSpectrum = doppler_spectrum_clean

            # ---- Step 3: remove corresponding receivers in rx_loc ----
            rx_loc_clean = rx_loc[:, ~nan_rx]
            self.trackConfig.rx_locs = rx_loc_clean.T  # Update rx_locs in trackConfig

        else:
            print("[Info] All receivers are valid; no NaNs found.")
            doppler_spectrum_clean = dopplerSpectrum
            rx_loc_clean = rx_loc
        return doppler_spectrum_clean, rx_loc_clean, removed_idx

        # ======================================================
        # New interface: extract tracking parameters (for multi-modal tracker)
        # ======================================================
    def extract_tracking_params(self, dfss_generator, path, traj_id, rep_id, user = None, modality=None, base_dir=None):
        """
        Extract parameters required by successiveTrack from raw data.
        user/modality/base_dir are optional parameters used to load doppler_spectrum from existing cached data.
        This design is not ideal; refactor later when time permits.
        """
        # 1. Generate DFS
        doppler_spectrum, freq_bin, time_bin = self._generateDFS(dfss_generator, path, traj_id, rep_id, user, modality, base_dir)
        # 2. TX / RX locations
        traj_cfg = self.trackConfig.trajectories[traj_id]
        init_loc = np.array(traj_cfg.start if rep_id % 2 == 1 else traj_cfg.end).reshape(2, 1)
        tx_loc = np.array(self.trackConfig.tx_loc).reshape(2, 1)
        rx_loc = ensure_contiguous(self.trackConfig.rx_locs.T)
        # 3. Remove abnormal RX
        doppler_spectrum, rx_loc, *_ = self.remove_nan_receivers(
            doppler_spectrum, rx_loc
        )
        # 4. Compute PLCR / direction / duration
        plcr_dec, direc_dec, duration = self.__calc_params(
            dopplerSpectrum=doppler_spectrum,
            freq_bin=freq_bin,
            time_bin=time_bin
        )

        return {
            "doppler_spectrum": doppler_spectrum,
            "freq_bin": ensure_contiguous(freq_bin, dtype=float),
            "init_loc": ensure_contiguous(init_loc, dtype=float),
            "tx_loc": ensure_contiguous(tx_loc, dtype=float),
            "rx_loc": ensure_contiguous(rx_loc, dtype=float),
            "plcr": ensure_contiguous(plcr_dec, dtype=float),
            "direc": ensure_contiguous(direc_dec, dtype=float),
            "duration": ensure_contiguous(duration, dtype=float),
            "traj_cfg": traj_cfg,
        }

    def _successive_track(self, dfss_generator, path, traj_id, rep_id, user = None, modality=None, base_dir=None):

        track_params = self.extract_tracking_params(dfss_generator, path, traj_id, rep_id, user, modality, base_dir)

        res = MatlabManager.call("successiveTrack",track_params["init_loc"], track_params["tx_loc"], track_params["rx_loc"],
                                             track_params["plcr"], track_params["direc"], track_params["duration"])


        pos = np.array(res['ls'], order='F')
        vs = np.array(res['vs'])

        groundtruth_path = os.path.join(user.track_groundTruth_dir, f"user-{user.user_id}-{traj_id}-{rep_id}.npy")
        #ground_truth_pos = self.ground_truth.generate(track_params["traj_cfg"])
        ground_truth_pos = self.ground_truth.get_trajectory(track_params["traj_cfg"], groundtruth_path)

        # # Plot tracking results
        self.plot_tracking(pos, ground_truth_pos, track_params["init_loc"],
                           track_params["traj_cfg"].end if rep_id % 2 == 1 else track_params["traj_cfg"].start,
                           track_params["tx_loc"], track_params["rx_loc"])

        return track_params["doppler_spectrum"], track_params["freq_bin"], pos, vs, track_params["duration"]


class MultiModalPositionTracker:
    def __init__(self, trackers: dict, groundTruth, base_dir, fusion_cfg, matlab_func_name="fusionSuccessiveTrack"):
        """
        Args:
            trackers: dict
                {
                    'wifi': PositionTracker,
                    'acoustic': PositionTracker,
                    ...
                }
            groundTruth: GroundTruthTrajectory
            base_dir: str, base directory for multi-modal data
            matlab_func_name: multi-modal MATLAB tracking function name
        """

        self.trackers = trackers
        self.ground_truth = groundTruth
        self.base_dir = base_dir

        # # Shared MATLAB Runtime
        # self.track_obj = MatlabRuntimeManager.get_runtime(
        #     initializer=lambda:fusion_successive_track.initialize()# Need to change the package name when registering
        # )

        self.fusion_preprocessor = FusionPreprocessor(
            time_align_mode=fusion_cfg.get("time_align_mode", "min"),
            reference_modality=fusion_cfg.get("reference_modality", None),
            rx_policy=fusion_cfg.get("rx_policy", {}),
            enable_sanity_check=fusion_cfg.get("enable_sanity_check", True),
        )

        self.matlab_func_name = matlab_func_name

    def _update_base_dir(self, base_dir):
        self.base_dir = base_dir

    def _fusion_successive_track(self, dfss_generators, paths: dict, traj_id: int, rep_id: int, user = None):
        """
        Args:
            paths:
                {
                    'wifi': path_prefix_wifi,
                    'acoustic': path_prefix_acoustic
                }
        """

        # ---------- 1. Initial position ----------
        any_tracker = next(iter(self.trackers.values()))
        traj_cfg = any_tracker.trackConfig.trajectories[traj_id]

        init_loc = np.array(
            traj_cfg.start if rep_id % 2 == 1 else traj_cfg.end
        ).reshape(2, 1)

        # ---------- 2. Extract parameters for each modality ----------
        modality_params = {}
        for modality, tracker in self.trackers.items():
            if modality not in paths:
                raise ValueError(f"Missing path for modality: {modality}")

            modality_dir = os.path.join(self.base_dir, modality)
            modality_params[modality] = tracker.extract_tracking_params(dfss_generators[modality],
                paths[modality], traj_id, rep_id,user, modality, modality_dir
            )

        modality_params = self.fusion_preprocessor.process(modality_params)

        # ---------- 3. Build MATLAB inputs ----------
        # MATLAB-friendly struct format
        matlab_input = {}
        for name, params in modality_params.items():
            matlab_input[name] = {
                "tx_loc": params["tx_loc"],
                "rx_loc": params["rx_loc"],
                "plcr": params["plcr"],
                "direc": params["direc"],
                "duration": params["duration"],
            }

        # ---------- 4. Call multi-modal MATLAB tracking ----------
        res = MatlabManager.call("fusionSuccessiveTrack",init_loc, matlab_input)

        # ---------- 5. Parse results ----------
        pos = np.array(res["ls"], order="F")
        vs = np.array(res["vs"])

        # ---------- 6. Optional: plotting ----------
        groundtruth_path = os.path.join(user.track_groundTruth_dir, f"user{user.user_id}-{traj_id}-{rep_id}.npy")
        ground_truth_pos = self.ground_truth.get_trajectory(traj_cfg, groundtruth_path)
        # Call the newly written plotting function
        # self.plot_fusion_tracking(
        #     ls=pos,
        #     ground=ground_truth_pos,
        #     modality_params=modality_params,  # Pass in the dict containing all modality info
        #     traj_cfg=traj_cfg
        # )

        return modality_params, pos, vs


    def plot_fusion_tracking(self, ls, ground, modality_params, traj_cfg):
        """
        Plot the multi-modal estimated trajectory, ground truth, and TX/RX positions for all modalities.
        """
        plt.figure(figsize=(6, 6))

        # 1. Plot ground truth (Groundtruth)
        if ground.shape[1] == 2:
            ground = ground.T
        plt.plot(ground[0, :], ground[1, :], '--', color=[0.5, 0.5, 0.5], label='Groundtruth', alpha=0.8)

        # 2. Plot estimated trajectory (Tracking Result)
        plt.plot(ls[0, :], ls[1, :], 'b-o', markersize=4, label='Tracking Result')

        # 3. Mark the start point (Start)
        plt.plot(ls[0, 0], ls[1, 0], 'ro', markersize=8, label='Start Node')

        # 4. Iterate and plot TX/RX for each modality
        # Define color mapping for different modalities for easier distinction
        colors = {'wifi': 'g', 'acoustic': 'm', 'radar': 'c'}
        markers = {'wifi': 's', 'acoustic': '^', 'radar': 'd'}

        for mod_name, params in modality_params.items():
            color = colors.get(mod_name.lower(), 'k')
            marker = markers.get(mod_name.lower(), 'x')

            # Plot TX for this modality (typically one TX per modality)
            tx = params["tx_loc"]
            plt.plot(tx[0], tx[1], marker=marker, color=color,
                     linestyle='None', markersize=10, label=f'{mod_name} TX')

            # Plot all RXs for this modality (typically multiple RXs per modality)
            rx_list = params["rx_loc"]  # Assume shape is [2, N_rx]
            plt.scatter(rx_list[0, :], rx_list[1, :], s=50, edgecolors=color,
                        facecolors='none', marker=marker, label=f'{mod_name} RXs')

        # 5. Dynamically set axis ranges (include all trajectory points and anchor locations)
        all_x = [ls[0, :], ground[0, :]]
        all_y = [ls[1, :], ground[1, :]]
        for p in modality_params.values():
            all_x.extend([p["tx_loc"][0], p["rx_loc"][0, :]])
            all_y.extend([p["tx_loc"][1], p["rx_loc"][1, :]])

        flatten_x = np.concatenate([np.atleast_1d(x).flatten() for x in all_x])
        flatten_y = np.concatenate([np.atleast_1d(y).flatten() for y in all_y])

        plt.xlim([np.min(flatten_x) - 1, np.max(flatten_x) + 1])
        plt.ylim([np.min(flatten_y) - 1, np.max(flatten_y) + 1])

        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.grid(True, linestyle=':', alpha=0.6)
        # Avoid an overly long legend; consider showing only part of it
        plt.legend(loc='best', fontsize='small', ncol=2)
        plt.title(f'Multi-modal Tracking Result')
        plt.show()
