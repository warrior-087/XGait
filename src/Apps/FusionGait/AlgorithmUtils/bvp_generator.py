from dataclasses import dataclass, field
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import rotate

from src.Apps.FusionGait.MatlabPlugins.utils import ensure_contiguous


@dataclass
class PhysicalSetup:
    # Candidate torso center positions (torso_pos)
    torso_pos: np.ndarray = field(default_factory=lambda: np.array([
        [1.365, 0.455], [0.455, 0.455], [0.455, 1.365], [1.365, 1.365],
        [0.91, 0.91], [2.275, 1.365], [2.275, 2.275], [1.365, 2.275]
    ]))
    # Candidate torso orientations (torso_ori)
    torso_ori: np.ndarray = field(default_factory=lambda: np.array([-90, -45, 0, 45, 90]))
    # Transmitter position
    Tx_pos: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    # Receiver positions
    Rx_pos: np.ndarray = field(default_factory=lambda: np.array([
        [0.455, -0.455], [1.365, -0.455], [2.0, 0.0],
        [-0.455, 0.455], [-0.455, 1.365], [0.0, 2.0]
    ]))
    rx_cnt: int = 6 # number of receivers

@dataclass
class VelocityMappingParams:
    seg_length: float = 100.0
    lambda_: float = 1e-7
    norm: int = 0
    V_max: float = 2.0
    V_min: float = -2.0
    V_bins: int = 20
    MaxFunctionEvaluations: int = 100000
    wave_length: float = 299792458 / 5.825e9

    @property
    def velocity_bin(self):
        # Discretized velocity bins in [-V_max, V_max], with 20 bins
        M = self.V_bins
        return ((np.arange(1, M+1) - M/2) / (M/2)) * self.V_max


class BVPGenerator:
    def __init__(self,  phys_cfg: PhysicalSetup,
                 vel_cfg: VelocityMappingParams):
                 # dfss_generator: DFSSGeneratorBase):
        self.phys_cfg = phys_cfg
        self.vel_cfg = vel_cfg
        # self.dsg = dfss_generator

    # def _generateDFS(self, path_pfx):
    #     # Generate Doppler spectrum
    #     doppler_spectrum, freq_bin = self.dsg.generate(path_pfx)
    #     freq_bin = np.asarray(freq_bin).ravel()
    #     return doppler_spectrum, freq_bin

    def _get_A_matrix(self, torso_pos, Tx_pos, Rx_pos, rxcnt):
        """
        Vectorized version: compute the geometric projection matrix A
        from velocity to Doppler frequency shift.

        Args:
            torso_pos : array-like, shape (2,)
                Torso center position, e.g., [x, y]
            Tx_pos : array-like, shape (2,)
                Transmitter position
            Rx_pos : array-like, shape (N_rx, 2)
                Receiver positions, one Rx per row
            rxcnt : int
                Number of receivers actually used (<= Rx_pos.shape[0])

        Returns:
            A : ndarray, shape (rxcnt, 2)
                Geometry matrix describing how the velocity vector maps to Doppler shift
        """
        torso_pos = np.asarray(torso_pos).reshape(1, 2)
        Tx_pos = np.asarray(Tx_pos).reshape(1, 2)
        Rx_pos = np.asarray(Rx_pos)

        if rxcnt > Rx_pos.shape[0]:
            raise ValueError("Error Rx Count! rxcnt cannot exceed number of Rx positions.")

        Rx_selected = Rx_pos[:rxcnt, :]

        # Unit vector from Tx to torso
        vec_tx = torso_pos - Tx_pos
        norm_tx = np.linalg.norm(vec_tx)
        unit_tx = vec_tx / norm_tx

        # Unit vectors from Rx to torso (rxcnt x 2)
        vec_rx = torso_pos - Rx_selected
        norm_rx = np.linalg.norm(vec_rx, axis=1, keepdims=True)
        unit_rx = vec_rx / norm_rx

        # Matrix A
        A = unit_tx + unit_rx  # broadcast addition

        return A

    def _get_velocity2doppler_mapping_matrix(self, A, wave_length, velocity_bin, freq_bin, rx_cnt):
        """
        Build the velocity-to-doppler mapping matrix (VDM)

        Returns:
            VDM : ndarray, shape (M, M, rx_cnt, F)
        """
        # ---- Ensure velocity_bin and freq_bin are at least 2D (1, M) ----
        velocity_bin = np.atleast_2d(velocity_bin)
        freq_bin = np.atleast_2d(freq_bin)

        if A.shape[0] != rx_cnt:
            raise ValueError("Error Rx Count!")

        M = velocity_bin.shape[1]  # take 2nd dim
        F = freq_bin.shape[1]  # take 2nd dim

        freq_min = int(np.min(freq_bin))
        freq_max = int(np.max(freq_bin))

        # Initialize VDM (rx_cnt, M, M, F)
        VDM = np.zeros((rx_cnt, M, M, F))

        for ii in range(rx_cnt):
            for i in range(M):  # x velocity
                for j in range(M):  # y velocity

                    v = np.array([velocity_bin[0, i], velocity_bin[0, j]])  # shape (2,)
                    plcr_hz = int(np.round(np.dot(A[ii, :].ravel(), v) / wave_length))

                    if plcr_hz < freq_min or plcr_hz > freq_max:
                        VDM[ii, i, j, :] = 1e10
                        continue

                    idx = plcr_hz - freq_min
                    VDM[ii, i, j, idx] = 1

        # Permute (rx_cnt, M, M, F) -> (M, M, rx_cnt, F)
        VDM = np.transpose(VDM, (1, 2, 0, 3))

        return VDM

    def _process_doppler_spectrum_parallel(self, doppler_spectrum, VDM, U_bound, ori):
        """
        Parallel version of Doppler spectrum processing.
        Calls MATLAB functions for optimization.
        """
        # Convert to contiguous arrays in memory
        doppler_spectrum = ensure_contiguous(doppler_spectrum, dtype=np.float64)
        VDM = ensure_contiguous(VDM, dtype=np.float64)
        U_bound = ensure_contiguous(U_bound, dtype=np.float64)

        obj = segmentOptimize.initialize()
        # import matlab.engine
        #eng = matlab.engine.connect_matlab()  # connect to shared MATLAB session
        res = obj.segmentOptimize(doppler_spectrum, VDM, self.vel_cfg.seg_length, self.vel_cfg.V_bins, U_bound, self.vel_cfg.MaxFunctionEvaluations,
                                  self.vel_cfg.norm, self.vel_cfg.lambda_, 1)
        velocity_spectrum = np.array(res,order='F')

        # ---- Rotate velocity spectrum according to orientation ----
        velocity_spectrum_ro = self._get_rotated_spectrum(velocity_spectrum, ori)
        return velocity_spectrum_ro

    def _get_rotated_spectrum(self, velocity_spectrum, orientation):
        """
            Rotate Velocity Spectrum to global coordinates.

            Args:
                velocity_spectrum : ndarray, shape (N, N, T)
                    Original Velocity Spectrum
                orientation : float
                    Orientation angle (degrees)

            Returns:
                VS_ro : ndarray, shape (N, N, T)
                    Rotated Velocity Spectrum
            """
        if velocity_spectrum.shape[0] != velocity_spectrum.shape[1]:
            raise ValueError("Not Square Matrix")

        N, _, T = velocity_spectrum.shape

        if N % 2 == 0:
            # Pad one row of zeros at the top
            VS_1 = np.concatenate([np.zeros((1, N, T)), velocity_spectrum], axis=0)  # (N+1, N, T)
            # Pad one column of zeros at the left
            VS_2 = np.concatenate([np.zeros((N + 1, 1, T)), VS_1], axis=1)  # (N+1, N+1, T)

            VS_2_ro = np.zeros_like(VS_2)
            for t in range(T):
                VS_2_ro[:, :, t] = rotate(
                    VS_2[:, :, t], orientation, axes=(0, 1),
                    order=0, reshape=False
                )

            # remove padding
            VS_ro = VS_2_ro[1:, 1:, :]
        else:
            VS_ro = np.zeros_like(velocity_spectrum)
            for t in range(T):
                VS_ro[:, :, t] = rotate(
                    velocity_spectrum[:, :, t], orientation, axes=(0, 1),
                    order=0, reshape=False
                )

        return VS_ro

    def animate_velocity_spectrum(self, VS: np.ndarray, interval: int = 500,
                                  cmap='viridis', db_scale: bool = True):
        """
        Animate the velocity spectrum (supports dB-scale visualization;
        axes use velocity_bin as coordinates).
        """
        if VS.ndim != 3:
            raise ValueError("VS must be 3D (N, N, T)")

        velocity_bin = np.asarray(self.vel_cfg.velocity_bin)
        N = len(velocity_bin)
        if VS.shape[0] != N or VS.shape[1] != N:
            raise ValueError("Velocity bin length does not match VS dimensions")

        # --- Convert to dB scale ---
        if db_scale:
            eps = 1e-12
            VS = 10 * np.log10(VS + eps)

        _, _, T = VS.shape
        vmin, vmax = float(VS.min()), float(VS.max())

        # --- Use real velocity values on axes ---
        extent = [velocity_bin[0], velocity_bin[-1],
                  velocity_bin[0], velocity_bin[-1]]  # [xmin, xmax, ymin, ymax]

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(VS[:, :, 0], origin='lower', aspect='auto',
                       vmin=vmin, vmax=vmax, interpolation='nearest',
                       animated=True, cmap=cmap, extent=extent)

        title_text = ax.set_title('Velocity Spectrum Frame 1', animated=True)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Amplitude (dB)' if db_scale else 'Amplitude (linear)')

        ax.set_xlabel('Velocity bin X')
        ax.set_ylabel('Velocity bin Y')

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

    def run(self, pfx_ges, torso_idx = 1, ori_idx = 1):
        """Run the full mapping pipeline once"""
        # 1. Generate Doppler spectrum
        doppler_spectrum, freq_bin = self.dsg.generate(pfx_ges)

        # 2. Compute A matrix
        A = self._get_A_matrix(self.phys_cfg.torso_pos[torso_idx-1,:], self.phys_cfg.Tx_pos,
                               self.phys_cfg.Rx_pos, self.phys_cfg.rx_cnt)

        # 3. Build VDM
        VDM = self._get_velocity2doppler_mapping_matrix(A, self.vel_cfg.wave_length, self.vel_cfg.velocity_bin,
                                                        freq_bin, self.phys_cfg.rx_cnt)

        doppler_spectrum_max = np.max(doppler_spectrum)  # max over all dimensions
        U_bound = np.full((self.vel_cfg.V_bins, self.vel_cfg.V_bins), doppler_spectrum_max)  # MxM upper-bound matrix

        # 4. Optimization
        velocity_spectrum_ro = self._process_doppler_spectrum_parallel(doppler_spectrum, VDM, U_bound, self.phys_cfg.torso_ori[ori_idx])

        self.animate_velocity_spectrum(velocity_spectrum_ro)
        return velocity_spectrum_ro
