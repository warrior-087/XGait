from dataclasses import dataclass
from scipy.signal import butter


# Doppler Frequency Shift Config

class DFSConfigBase:
    """Common configuration items"""
    def __init__(self, samp_rate:float, rx_cnt, method = 'stft'):
        self.samp_rate:float = samp_rate   # sampling rate
        self.rx_cnt:int = rx_cnt
        self.method:str = method    # time-frequency analysis method: 'stft' or 'cwt'


class WiFiDFSConfig(DFSConfigBase):
    def __init__(self, samp_rate=1000.0, lpf_order=6, lpf_stop=80.0,
                 hpf_order=3, hpf_stop=2.0, rx_cnt=3, rx_acnt=3):
        super().__init__(samp_rate, rx_cnt)

        # band-pass filter parameters
        self.lpf_order:int = lpf_order
        self.lpf_stop:float = lpf_stop
        self.hpf_order:int = hpf_order
        self.hpf_stop:float = hpf_stop

        # number of antennas
        self.rx_acnt = rx_acnt

class AcousticDFSConfig(DFSConfigBase):
    def __init__(self, freq = 19000.0, samp_rate=48000.0, fft_size = 8192, fft_overlap = 7808, bs_order=5, bsb=15.0,
                  freq_range_low = 18000.0, freq_range_high = 20000.0, rx_cnt=3):
        super().__init__(samp_rate, rx_cnt)

        # band-stop filter parameters
        self.bsl_order: int = bs_order
        self.bsb :float = bsb

        # target frequency range (can be regarded as a band-pass filter)
        self.freq_range_low: float = freq_range_low
        self.freq_range_high: float = freq_range_high

        self.fft_size: int = fft_size
        self.fft_overlap: int = fft_overlap
        self.ultrasonic_freq: float = freq
