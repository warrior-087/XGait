function [x_ds, fs_new] = demod_and_downsample(x_filt, fs, fc, fs_new)
% Inputs:
%   x_filt  - Band-pass filtered signal that still contains the carrier (e.g., ~20 kHz)
%   fs      - Original sampling rate in Hz (e.g., 48000)
%   fc      - Carrier frequency in Hz (e.g., 20000)
%   fs_new  - Target sampling rate after downsampling in Hz (e.g., 1000)
%
% Outputs:
%   x_ds    - Baseband signal after downconversion and resampling
%   fs_new  - Output sampling rate (same as the input argument fs_new)

    % Time axis (seconds)
    t = (0:length(x_filt)-1)' / fs;

    % ---------------------------
    % 1) Obtain the analytic signal via Hilbert transform
    % ---------------------------
    x_analytic = hilbert(x_filt);

    % ---------------------------
    % 2) Downconvert (remove the carrier at fc)
    % ---------------------------
    x_baseband = x_analytic .* exp(-1j * 2 * pi * fc * t);

    % ---------------------------
    % 3) Low-pass filter to keep Doppler content (e.g., up to ~500 Hz)
    % ---------------------------
    lpFilt = designfilt('lowpassiir', ...
                        'FilterOrder', 8, ...
                        'PassbandFrequency', 500, ...   % Doppler upper bound
                        'PassbandRipple', 0.2, ...
                        'SampleRate', fs);

    % Use zero-phase filtering to avoid phase distortion
    x_lp = filtfilt(lpFilt, x_baseband);

    % ---------------------------
    % 4) Resample to the new fs_new
    % ---------------------------
    x_ds = resample(x_lp, fs_new, fs);
end
