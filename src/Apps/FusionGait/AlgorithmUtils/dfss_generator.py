from abc import ABC, abstractmethod
import numpy as np
from matplotlib import pyplot as plt
from src.Apps.FusionGait.ConfigUtils.dfs_config import WiFiDFSConfig, DFSConfigBase
from src.Apps.FusionGait.MatlabPlugins.matlab_runtime_singleton import MatlabManager


class DFSSGeneratorBase(ABC):
    def __init__(self, cfg):
        self.cfg = cfg


    def spectrum_view(self, doppler_spectrum, freq_range=(-60, 60), time_unit='ms'):
        rx_cnt, freq_bin_num, time_len = doppler_spectrum.shape
        freqs = np.linspace(freq_range[0], freq_range[1], freq_bin_num)
        times = np.arange(time_len)

        fig, axes = plt.subplots(2, 3, figsize=(12, 6))
        axes = axes.flatten()

        for kk in range(min(rx_cnt, 6)):
            ax = axes[kk]
            im = ax.pcolormesh(times, freqs, doppler_spectrum[kk, :, :],
                               shading='auto', cmap='jet')
            ax.set_xlabel(f'Time ({time_unit})')
            ax.set_ylabel('Frequency (Hz)')
            ax.set_yticks(np.arange(freq_range[0], freq_range[1] + 1, 20))
            ax.set_title(f'Receiver {kk + 1}')
            ax.grid(False)

        fig.tight_layout()
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
        cbar.set_label('Amplitude')
        plt.show()

    @abstractmethod
    def generate(self, Gspfx_ges):
        """Generate Doppler spectrum"""
        pass


class WiFiDFSSGenerator(DFSSGeneratorBase):

    def generate(self, Gspfx_ges):
        res = MatlabManager.call("timeFrequencyAnalysisWiFi",Gspfx_ges, self.cfg.samp_rate, self.cfg.lpf_order, self.cfg.lpf_stop,
                                                  self.cfg.hpf_order, self.cfg.hpf_stop, self.cfg.rx_cnt, self.cfg.rx_acnt)

        doppler_spectrum = np.array(res['freq_time_prof_allfreq'], order='F')
        freq_bin = np.array(res['freq_bin'])
        time_bin = np.array(res['time_bin'], dtype=np.float64)

        # Visualization
        # self.doppler_spectrum_view(doppler_spectrum)

        return doppler_spectrum, freq_bin, time_bin

class AcousticDFSSGenerator(DFSSGeneratorBase):

    def generate(self, Gspfx_ges):

        res = MatlabManager.call("timeFrequencyAnalysisAcoustic",Gspfx_ges, self.cfg.samp_rate, self.cfg.ultrasonic_freq,
                                                        self.cfg.fft_size,
                                                        self.cfg.fft_overlap, self.cfg.bsl_order,
                                                        self.cfg.bsb, self.cfg.freq_range_low,
                                                        self.cfg.freq_range_high, self.cfg.rx_cnt)


        doppler_spectrum = np.array(res['freq_time_prof_allfreq'], order='F')
        freq_bin = np.array(res['freq_bin'])
        time_bin = np.array(res['time_bin'],dtype=np.float64).ravel()
        # Visualization
        # self.doppler_spectrum_view(doppler_spectrum)

        return doppler_spectrum, freq_bin, time_bin

class DFSSGeneratorFactory:

    @staticmethod
    def create(modality: str, cfg: DFSConfigBase):
        modality = modality.lower()

        if modality == "wifi":
            return WiFiDFSSGenerator(cfg)
        elif modality == "acoustic":
            return AcousticDFSSGenerator(cfg)
        else:
            raise ValueError(f"Unknown modality: {modality}")


if __name__ == "__main__":
    # Select configuration
    cfg = WiFiDFSConfig()
    # Create the correct generator
    gen = DFSSGeneratorFactory.create("wifi", cfg)
