import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import stft

def plot_wav_stft(wav_path, nperseg=8192, noverlap=7808):
    """
    Read WAV file, and plot time domain waveform and STFT spectrogram
    :param wav_path: WAV file path
    :param nperseg: Length of each frame (typically 1024 or 2048)
    :param noverlap: Overlap between frames (typically 50%)
    """
    # ===== 1. Read WAV =====
    sample_rate, data = wavfile.read(wav_path)

    print(f"Sample Rate: {sample_rate} Hz")
    print(f"Data Type: {data.dtype}")
    print(f"Shape: {data.shape}")

    # If multi-channel, use the first channel
    if len(data.shape) == 2:
        data = data[:, 0]

    # Convert to float
    data = data.astype(float)

    # ===== 2. Plot Time Domain Waveform =====
    t = np.arange(len(data)) / sample_rate

    plt.figure(figsize=(12, 4))
    plt.plot(t, data)
    plt.title("Time Domain Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()

    # ===== 3. Calculate STFT =====
    f, time, Zxx = stft(
        data,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap
    )

    magnitude = np.abs(Zxx)

    # ===== 4. Plot STFT Spectrogram =====
    # Frequency range
    f_min = 18500
    f_max = 19500

    # Find frequency indices
    idx = np.where((f >= f_min) & (f <= f_max))[0]

    # Crop frequency
    f_zoom = f[idx]

    # magnitude might have an extra dimension, needs squeeze
    magnitude_zoom = magnitude.squeeze()[idx, :]

    # ===== 4. Plot STFT Spectrogram =====
    plt.figure(figsize=(12, 6))
    plt.pcolormesh(time, f_zoom, 20 * np.log10(magnitude_zoom + 1e-6), shading='gouraud')
    plt.title("STFT Spectrogram (18.5–19.5 kHz)")
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")
    plt.colorbar(label="Magnitude (dB)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Change to your wav file path
    wav_file = "temp_record.wav"
    plot_wav_stft(wav_file)
