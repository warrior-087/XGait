function y = bandpass_filter(x, Fs, lowF, highF, N)
    % bandpass_filter  Apply a band-pass filter to signal x
    %
    % Inputs:
    %   x      : Input signal (vector)
    %   Fs     : Sampling rate in Hz (e.g., 48000)
    %   lowF   : Lower cutoff frequency in Hz (e.g., 18000)
    %   highF  : Upper cutoff frequency in Hz (e.g., 20000)
    %   N      : Butterworth filter order (typically 4–8)
    %
    % Output:
    %   y      : Band-pass filtered signal

    % Normalize cutoff frequencies to the Nyquist frequency (Fs/2)
    Wn = [lowF highF] / (Fs/2);

    % Design an Nth-order Butterworth band-pass filter
    [b, a] = butter(N, Wn, 'bandpass');

    % Zero-phase filtering using filtfilt (forward-backward filtering)
    y = filtfilt(b, a, x);
end
