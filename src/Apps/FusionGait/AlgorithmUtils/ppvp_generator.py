from dataclasses import dataclass
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from src.Apps.FusionGait.AlgorithmUtils.bvp_generator import VelocityMappingParams, PhysicalSetup
from src.Apps.FusionGait.AlgorithmUtils.gbvp_generator import GBVPGenerator


@dataclass
class PolarVelocityMappingParams(VelocityMappingParams):
    """Polar-coordinate velocity mapping parameters (used by STAGR)"""
    Theta_min: float = 0.0
    Theta_max: float = 360.0
    Theta_bins: int = 360

    V_max: float = 3.5
    V_min: float = 0.0
    V_bins: int = 150  # Default: 100 velocity bins (0~2 m/s)

    @property
    def velocity_bin(self):
        return np.linspace(self.V_min, self.V_max, self.V_bins, endpoint=False)

    @property
    def theta_bin(self):
        return np.linspace(self.Theta_min, self.Theta_max, self.Theta_bins, endpoint=False)


class PPVPGenerator(GBVPGenerator):
    """Polarized Perception Velocity Profile generator"""
    def __init__(self, phys_cfg: PhysicalSetup, vel_cfg: PolarVelocityMappingParams):
        super().__init__(phys_cfg, vel_cfg)

    def fuse_ppvp_multiplicative(self, ppvp, eps=1e-12):
        """
        Multi-link multiplicative fusion (original definition in the STAGR paper)
        Input:
            ppvp: np.ndarray, shape (rxcnt, Ttheta, Tv, T_pos)
        Returns:
            fused_logscaled: (Ttheta, Tv, T_pos)
            per_link_energy: (rxcnt, T_pos)
        """
        # 1. Prevent log(0)

        ppvp_safe = ppvp + eps

        # 2. Sum in log-domain (equivalent to multiplication)
        fused_log = np.sum(np.log(ppvp_safe), axis=0)  # sum over rxcnt

        # 3. Recover by exp + normalize
        fused = np.exp(fused_log)
        fused /= (np.max(fused, axis=(0, 1), keepdims=True) + eps)

        # 4. Log compression (for visualization / downstream processing)
        fused_logscaled = np.log1p(fused)

        # 5. Output overall energy per link (for debugging)
        per_link_energy = np.sum(ppvp, axis=(1, 2))  # (rxcnt, T_pos)

        return fused_logscaled, per_link_energy

    def animate_velocity_spectrum(self, VS: np.ndarray, interval: int = 500,
                                  cmap='viridis', db_scale: bool = False):
        """
        Animate the velocity spectrum, with the x-axis as velocity_bin and the y-axis as theta_bin.

        VS shape: (Nt, Nv, T)
            Nt = len(theta_bin)
            Nv = len(velocity_bin)
        """

        # --- Get bins ---
        velocity_bin = np.asarray(self.vel_cfg.velocity_bin)  # X axis
        theta_bin = np.asarray(self.vel_cfg.theta_bin)  # Y axis

        Nt = len(theta_bin)
        Nv = len(velocity_bin)

        # --- Check dimensions ---
        if VS.ndim != 3:
            raise ValueError("VS must be 3D (Nt, Nv, T)")

        if VS.shape[0] != Nt or VS.shape[1] != Nv:
            raise ValueError(f"VS shape mismatch: expected ({Nt},{Nv},T), got {VS.shape}")

        _, _, T = VS.shape

        # --- dB conversion ---
        if db_scale:
            eps = 1e-12
            VS = 10 * np.log10(VS + eps)

        vmin, vmax = float(VS.min()), float(VS.max())

        # --- extent specifies axis ranges: x=velocity, y=theta ---
        extent = [
            velocity_bin[0], velocity_bin[-1],  # x_min, x_max
            theta_bin[0], theta_bin[-1]  # y_min, y_max
        ]

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(VS[:, :, 0], origin='lower', aspect='auto',
                       vmin=vmin, vmax=vmax, interpolation='nearest',
                       animated=True, cmap=cmap, extent=extent)

        title_text = ax.set_title('Velocity Spectrum Frame 1', animated=True)

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Amplitude (dB)' if db_scale else 'Amplitude (linear)')

        ax.set_xlabel('Velocity (m/s)')
        ax.set_ylabel('Theta (rad)' if np.max(np.abs(theta_bin)) <= 3.5 else 'Theta (deg)')

        # --- Animation functions ---
        def init():
            im.set_data(VS[:, :, 0])
            title_text.set_text('Velocity Spectrum Frame 1')
            return (im, title_text)

        def update(frame):
            im.set_data(VS[:, :, frame])
            title_text.set_text(f'Frame {frame + 1}/{T}')
            return (im, title_text)

        ani = FuncAnimation(fig, update, frames=T, init_func=init,
                            interval=interval, blit=True, repeat=True)

        plt.tight_layout()
        plt.show()
        return ani

    def run(self, pos_list, doppler_spectrum, freq_bin, duration):
        """Optimized STAGR PLCR → PPVP mapping (vectorized accelerated version)"""
        doppler_spectrum = np.asarray(doppler_spectrum)
        rxcnt, freq, frames = doppler_spectrum.shape
        T_pos = pos_list.shape[1]

        velocity_bins = self.vel_cfg.velocity_bin
        theta_bins = self.vel_cfg.theta_bin
        Ttheta, Tv = len(theta_bins), len(velocity_bins)
        wave_length = self.vel_cfg.wave_length

        # --- [Optimization 1] Precompute angular information (compute once) ---
        theta_rad = np.deg2rad(theta_bins)
        cos_theta = np.cos(theta_rad)[:, None]  # (Ttheta, 1)
        sin_theta = np.sin(theta_rad)[:, None]  # (Ttheta, 1)

        # Number of Doppler frames per position point
        frames_per_pos = frames // T_pos

        # Initialize PPVP tensor
        ppvp = np.zeros((rxcnt, Ttheta, Tv, T_pos), dtype=float)

        for t in range(T_pos):
            pos = pos_list[:, t]
            # A matrix shape: (rxcnt, 2)
            A = -self._get_A_matrix(pos, self.phys_cfg.Tx_pos, self.phys_cfg.Rx_pos, rxcnt)

            start = t * frames_per_pos
            end = min((t + 1) * frames_per_pos, frames)

            # --- [Optimization 2] Mean over the time slice (compute once for all Rx) ---
            # spec_mean_t shape: (rxcnt, freq)
            spec_mean_t = np.mean(doppler_spectrum[:, :, start:end], axis=2)

            for i in range(rxcnt):
                spec_frame = spec_mean_t[i]  # (freq,)

                # --- [Optimization 3] Vectorized Doppler shift computation ---
                a_x, a_y = A[i]
                # Compute frequency projection f_d via broadcasting, shape: (Ttheta, Tv)
                proj = a_x * cos_theta + a_y * sin_theta
                f_d = proj * velocity_bins[None, :] / wave_length
                f_flat = f_d.ravel()

                # --- [Optimization 4] Use searchsorted instead of argmin (order-of-magnitude speedup) ---
                # searchsorted uses binary search; much faster than full absolute-difference matrix
                idx = np.searchsorted(freq_bin, f_flat)
                idx = np.clip(idx, 1, len(freq_bin) - 1)
                left = freq_bin[idx - 1]
                right = freq_bin[idx]
                # Correct to nearest neighbor
                idx -= (np.abs(f_flat - left) < np.abs(f_flat - right))

                # Fetch intensity values and fill
                values = spec_frame[idx].reshape(Ttheta, Tv)

                # --- [Optimization 5] In-place normalization to reduce memory copies ---
                max_val = np.max(values)
                ppvp[i, :, :, t] = values / (max_val + 1e-6)

        # Fusion logic
        fused_logscaled, *_ = self.fuse_ppvp_multiplicative(ppvp)
        return fused_logscaled



class FusionPPVPGenerator(PPVPGenerator):
    def __init__(self, phys_cfg_dict: dict, vel_cfg_dict: dict):
        super().__init__(phys_cfg_dict["wifi"], vel_cfg_dict["wifi"])
        """
        phys_cfg_dict: dict, modality -> PhysicalSetup
        vel_cfg_dict: dict, modality -> PolarVelocityMappingParams
        """
        self.phys_cfg_dict = phys_cfg_dict
        self.vel_cfg_dict = vel_cfg_dict

    def run(self, modality_params: dict, pos_list):
        """
        Run multi-modal Fusion PPVP
        modality_params: dict, modality -> dict containing doppler_spectrum, freq_bin, init_loc, tx_loc, rx_loc, duration, etc.
        """
        modalities = list(modality_params.keys())
        first_mod = modalities[0]

        duration = modality_params[first_mod]["duration"]

        # Velocity and theta bins (use the first modality as reference)
        velocity_bins = self.vel_cfg_dict[first_mod].velocity_bin
        theta_bins = self.vel_cfg_dict[first_mod].theta_bin
        Ttheta, Tv = len(theta_bins), len(velocity_bins)
        T_pos = pos_list.shape[1]

        # -----------------------
        # Store PPVP for each modality
        # -----------------------
        ppvp_mod_dict = {}
        for mod in modalities:
            doppler_spectrum = modality_params[mod]["doppler_spectrum"]
            freq_bin = modality_params[mod]["freq_bin"]
            Tx_pos = modality_params[mod]["tx_loc"].T
            Rx_pos = modality_params[mod]["rx_loc"].T
            rxcnt = Rx_pos.shape[0]
            wave_length = self.vel_cfg_dict[mod].wave_length

            # Initialize PPVP
            ppvp = np.zeros((rxcnt, Ttheta, Tv, T_pos), dtype=float)

            # --------- [Optimization 1] Theta-related terms computed once ----------
            theta_rad = np.deg2rad(theta_bins)
            cos_theta = np.cos(theta_rad)[:, None]  # (Ttheta, 1)
            sin_theta = np.sin(theta_rad)[:, None]

            # --------- [Optimization 2] Frame info computed once ----------
            frames = doppler_spectrum.shape[2]
            frames_per_pos = frames // T_pos

            for t in range(T_pos):
                pos = pos_list[:, t]
                A = -self._get_A_matrix(pos, Tx_pos, Rx_pos, rxcnt)

                start = t * frames_per_pos
                end = min((t + 1) * frames_per_pos, frames)

                # --------- [Optimization 3] Doppler mean for this slice computed once ----------
                spec_mean_t = np.mean(doppler_spectrum[:, :, start:end], axis=2)  # shape: (rxcnt, Freq)

                for i in range(rxcnt):
                    spec_frame = spec_mean_t[i]

                    a_x, a_y = A[i]
                    proj = a_x * cos_theta + a_y * sin_theta
                    f_d = proj * velocity_bins[None, :] / wave_length

                    f_flat = f_d.ravel()

                    # --------- [Optimization 4] Equivalent but faster nearest-frequency lookup ----------
                    idx = np.searchsorted(freq_bin, f_flat)
                    idx = np.clip(idx, 1, len(freq_bin) - 1)
                    left = freq_bin[idx - 1]
                    right = freq_bin[idx]
                    idx -= (np.abs(f_flat - left) < np.abs(f_flat - right))

                    values = spec_frame[idx].reshape(Ttheta, Tv)
                    ppvp[i, :, :, t] = values / (np.max(values) + 1e-6)

            ppvp_mod_dict[mod] = ppvp

        fused_logscaled = self.fusion(ppvp_mod_dict)
        # self.animate_velocity_spectrum(fused_logscaled, db_scale=True)

        return fused_logscaled


    def fusion(self, ppvp_mod_dict):
        # -----------------------
        # Multi-modal fusion: multiply in log-domain
        # Supports different rxcnt
        # -----------------------
        ppvp_list = list(ppvp_mod_dict.values())

        fused_log = None

        for ppvp in ppvp_list:
            # ppvp: (rxcnt, Ttheta, Tv, T_pos)
            cur_log = np.sum(np.log(ppvp + 1e-12), axis=0)  # sum over rx

            if fused_log is None:
                fused_log = cur_log
            else:
                fused_log += cur_log  # sum over modality

        # Back to linear domain
        fused = np.exp(fused_log)

        # Normalize (your original logic)
        fused /= (np.max(fused, axis=(0, 1), keepdims=True) + 1e-12)

        fused_logscaled = np.log1p(fused)
        return fused_logscaled


class FusionProGenerator(FusionPPVPGenerator):
    """
    Physics-Consistent Velocity Profile (PCVP)
    A physics-consistent fusion version based on PPVP
    """

    # =====================================================
    # Core fusion logic
    # =====================================================
    def fusion(self, ppvp_mod_dict):
        """
        ppvp_mod_dict:
            modality -> ppvp
            ppvp shape: (rxcnt, Ttheta, Tv, T_pos)

        return:
            pcvp_logscaled: (Ttheta, Tv, T_pos)
        """

        eps = 1e-12
        modality_energy = {}

        # =====================================================
        # Step 1: Intra-modality fusion (PPVP-style, multi-link consistency)
        # =====================================================
        for mod, ppvp in ppvp_mod_dict.items():
            # ppvp: (rxcnt, Ttheta, Tv, T)

            log_ppvp = np.log(ppvp + eps)  # (rxcnt, Ttheta, Tv, T)
            log_energy = np.sum(log_ppvp, axis=0)  # (Ttheta, Tv, T)

            energy = np.exp(log_energy)
            energy /= (np.max(energy, axis=(0, 1), keepdims=True) + eps)

            modality_energy[mod] = energy  # (Ttheta, Tv, T)

        # =====================================================
        # Step 2: Inter-modality consistency fusion (PCVP core)
        # =====================================================
        energy_stack = np.stack(
            list(modality_energy.values()), axis=0
        )  # (N_mod, Ttheta, Tv, T)

        log_fused = np.sum(np.log(energy_stack + eps), axis=0)
        fused = np.exp(log_fused)
        fused /= (np.max(fused, axis=(0, 1), keepdims=True) + eps)

        pcvp_logscaled = np.log1p(fused)

        # =====================================================
        # Step 3: Interpretable comparison animation (directly inside fusion)
        # =====================================================
        try:
            VS_list = []
            titles = []

            # Per-modality PPVP (intra-modality fusion result)
            for mod, energy in modality_energy.items():
                VS_list.append(np.log1p(energy))
                titles.append(f"{mod.upper()}-PPVP")

            # PCVP
            VS_list.append(pcvp_logscaled)
            titles.append("PCVP (Fusion)")

            # 3×1 side-by-side animation
            # self.animate_velocity_spectrum_comparison(
            #     VS_list=VS_list,
            #     titles=titles,
            #     db_scale=False
            # )

        except Exception as e:
            print(f"[FusionProGenerator] Visualization skipped: {e}")

        # =====================================================
        # Step 4: Cache for subsequent analysis / paper plotting
        # =====================================================
        self._last_modality_energy = modality_energy
        self._last_pcvp = pcvp_logscaled

        return pcvp_logscaled

    def animate_velocity_spectrum_comparison(
        self,
        VS_list: list,
        titles: list,
        interval: int = 500,
        cmap: str = "viridis",
        db_scale: bool = False,
    ):
        """
        Side-by-side comparison animation for multiple Velocity Spectra (1×N)

        VS_list:
            List[np.ndarray], each shape (Ttheta, Tv, T)
        titles:
            List[str]
        """

        assert len(VS_list) == len(titles), "VS_list and titles must have the same length"

        # --- bins ---
        velocity_bin = np.asarray(self.vel_cfg.velocity_bin)
        theta_bin = np.asarray(self.vel_cfg.theta_bin)

        Nt = len(theta_bin)
        Nv = len(velocity_bin)

        # --- Check dimensional consistency ---
        T = VS_list[0].shape[2]
        for VS in VS_list:
            if VS.shape[:2] != (Nt, Nv):
                raise ValueError("All VS must share the same first two dimensions")
            if VS.shape[2] != T:
                raise ValueError("All VS must share the same time dimension T")

        # --- dB scale ---
        if db_scale:
            eps = 1e-12
            VS_list = [10 * np.log10(VS + eps) for VS in VS_list]

        # --- Global color range (for comparison) ---
        vmin = min(VS.min() for VS in VS_list)
        vmax = max(VS.max() for VS in VS_list)

        # --- extent ---
        extent = [
            velocity_bin[0], velocity_bin[-1],
            theta_bin[0], theta_bin[-1]
        ]

        # --- Create subplots ---
        n = len(VS_list)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)

        if n == 1:
            axes = [axes]

        ims = []
        title_texts = []

        for ax, VS, title in zip(axes, VS_list, titles):
            im = ax.imshow(
                VS[:, :, 0],
                origin="lower",
                aspect="auto",
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
                cmap=cmap,
                extent=extent,
                animated=True,
            )
            ax.set_title(title)
            ax.set_xlabel("Velocity (m/s)")
            title_texts.append(
                ax.text(
                    0.5,
                    1.02,
                    "Frame 1",
                    transform=ax.transAxes,
                    ha="center",
                )
            )
            ims.append(im)

        axes[0].set_ylabel(
            "Theta (rad)" if np.max(np.abs(theta_bin)) <= 3.5 else "Theta (deg)"
        )

        # --- colorbar ---
        cbar = fig.colorbar(ims[0], ax=axes, shrink=0.85)
        cbar.set_label("Amplitude (dB)" if db_scale else "Amplitude")

        # --- animation ---
        def init():
            for im, VS, txt in zip(ims, VS_list, title_texts):
                im.set_data(VS[:, :, 0])
                txt.set_text("Frame 1")
            return ims + title_texts

        def update(frame):
            for im, VS, txt in zip(ims, VS_list, title_texts):
                im.set_data(VS[:, :, frame])
                txt.set_text(f"Frame {frame + 1}/{T}")
            return ims + title_texts

        ani = FuncAnimation(
            fig,
            update,
            frames=T,
            init_func=init,
            interval=interval,
            blit=True,
            repeat=True,
        )

        plt.tight_layout()
        plt.show()
        return ani
