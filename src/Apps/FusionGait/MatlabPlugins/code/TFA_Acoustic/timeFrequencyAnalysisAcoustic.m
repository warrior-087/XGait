function res = timeFrequencyAnalysisAcoustic(spfx_ges, samp_rate, central_freq, fft_size, fft_overlap, bandstop_orde, bandstop_band, ...
    freq_range_low, freq_range_high, rx_cnt)
% timeFrequencyAnalysisAcoustic  Perform time-frequency analysis on acoustic data and generate Doppler spectra
%
% Inputs:
%   spfx_ges        - File name prefix for the data, corresponding to one gesture/motion
%   samp_rate       - Sampling rate (Hz)
%   central_freq    - Carrier/center frequency for demodulation (Hz), e.g., 20000
%   fft_size        - FFT size (currently not used in this implementation)
%   fft_overlap     - FFT overlap (currently not used in this implementation)
%   bandstop_orde   - Filter order used in preprocessing (used as band-pass order here)
%   bandstop_band   - Width of the DC notch region to fill around 0 Hz (same unit as freq_bin)
%   freq_range_low  - Lower frequency bound of the band of interest (Hz)
%   freq_range_high - Upper frequency bound of the band of interest (Hz)
%   rx_cnt          - Number of receivers
%
% Outputs:
%   res.freq_time_prof_allfreq - Doppler time-frequency profile (receiver × frequency × time)
%   res.freq_bin               - Frequency bin indices (can be mapped to Hz depending on definition)
%   res.time_bin               - Time axis (seconds)

% Force scalar types to double to avoid integer propagation when called from Python/MCR.
% MATLAB Runtime can treat Python integers as integer types, which may affect computations.
samp_rate        = double(samp_rate);
fft_size         = double(fft_size);
fft_overlap      = double(fft_overlap);
bandstop_orde    = double(bandstop_orde);
bandstop_band    = double(bandstop_band);
freq_range_low   = double(freq_range_low);
freq_range_high  = double(freq_range_high);
central_freq     = double(central_freq);
rx_cnt           = double(rx_cnt);

downsamp_rate = 1000; % Target sampling rate after downconversion/downsampling
downsamp_band = 400;  % Keep Doppler content roughly within +/- downsamp_band

freq_bins_unwrap = [0:downsamp_rate/2-1 -downsamp_rate/2:-1]'/downsamp_rate;  % Normalized FFT frequency grid (cycles/sample)
freq_lpf_sele = freq_bins_unwrap <= downsamp_band / downsamp_rate & freq_bins_unwrap >= -downsamp_band / downsamp_rate;
freq_lpf_positive_max = sum(freq_lpf_sele(2:length(freq_lpf_sele)/2));    % #bins in positive-frequency region
freq_lpf_negative_min = sum(freq_lpf_sele(length(freq_lpf_sele)/2:end));  % #bins in negative-frequency region
freq_bin = [0:freq_lpf_positive_max -1*freq_lpf_negative_min:-1];

%% ------------------------------------------------
% Step 1: Construct filters
%% ------------------------------------------------
fprintf('Step 1: Constructing filters...\n');
half_rate = samp_rate / 2;

% Create filters
% [bsu, bsd] = butter(bandstop_orde, [bandstop_low bandstop_high] / half_rate, 'stop'); % Band-stop (unused / legacy)
[hu, hd] = butter(bandstop_orde, freq_range_low/half_rate, 'high'); % High-pass filter
[lu, ld] = butter(10, freq_range_high/half_rate, 'low');           % Low-pass filter

%% ------------------------------------------------
% Step 2: Acoustic preprocessing & baseband extraction (per receiver)
%% ------------------------------------------------
% Iterate over all receivers to:
%   - load audio,
%   - band-pass filter around the transmit band,
%   - downconvert (demodulate) to baseband,
%   - downsample to downsamp_rate,
% and record the maximum sequence length for later pre-allocation.
max_frame_len = 0;
motion_audio_all = cell(rx_cnt, 1);  % Store processed baseband signals for each receiver

for ii = 1:rx_cnt
    spth = [spfx_ges, '-r', num2str(ii), '.wav'];
    try
        % audioread/audioDatastore may be buggy under Linux MATLAB Runtime in some setups.
        % Use manual WAV parsing instead: skip 44-byte header and read int16 PCM payload.
        fid = fopen(spth, 'rb');
        if fid < 0
            error('Cannot open WAV file: %s', spth);
        end

        fseek(fid, 44, 'bof');           % Skip WAV header
        raw_data = fread(fid, Inf, 'int16'); % Read entire PCM payload
        fclose(fid);
        audio_data = double(raw_data) / 32768; % Convert to [-1, 1] double

        % Remove DC offset
        audio_data = audio_data - mean(audio_data, 1);

        % Band-pass filter around the carrier band of interest
        audio_band_pass = bandpass_filter(audio_data, samp_rate, freq_range_low, freq_range_high, bandstop_orde);

        % Downconvert to baseband and downsample
        downsample_data = demod_and_downsample(audio_band_pass, samp_rate, central_freq, downsamp_rate);

        % Track the maximum signal length across receivers
        max_frame_len = max(max_frame_len, size(downsample_data, 1));
        motion_audio_all{ii} = downsample_data; % Save processed signal
    catch err
        disp(err);
        motion_audio_all{ii} = [];
        continue;
    end
end

% Pre-allocate Doppler spectrum tensor:
%   doppler_spectrum: (receiver × frequency_bin × time)
freq_bin_count = length(freq_bin);
doppler_spectrum = zeros(rx_cnt, freq_bin_count, max_frame_len);

%% ------------------------------------------------
% Step 3: Compute time-frequency representation for each receiver
%% ------------------------------------------------
fprintf('Step 3: Process all receivers...\n');

for ii = 1:rx_cnt
    fprintf('Processing RX #%d...\n', ii);
    processed_audio = motion_audio_all{ii};

    % Remove residual DC (baseband)
    processed_audio = processed_audio - mean(processed_audio, 1);

    time_instance = 1:length(processed_audio);
    window_size = round(downsamp_rate/4 + 1);
    if(~mod(window_size,2))
        window_size = window_size + 1;
    end

    % Time-frequency representation (TFR) using a Gaussian window
    freq_time_prof_allfreq = tfrsp(processed_audio, time_instance, downsamp_rate, tftb_window(window_size, 'gauss'));

    % Keep only the Doppler band and normalize per time frame
    freq_time_prof = freq_time_prof_allfreq(freq_lpf_sele, :);
    freq_time_prof = abs(freq_time_prof) ./ repmat(sum(abs(freq_time_prof),1), size(freq_time_prof,1), 1);

    % % Optional: the dominant component may concentrate near 0 Hz; fill a DC notch if needed
    % freq_time_prof = smooth_dc_notch_fill(freq_time_prof, freq_bin, bandstop_band, 0);

    actual_frames = size(freq_time_prof, 2);

    % Store into doppler_spectrum
    doppler_spectrum(ii, :, 1:actual_frames) = freq_time_prof;
end

% Circularly shift the frequency axis ordering (align negative/positive bins as expected downstream)
[~,idx] = max(freq_bin);
circ_len = length(freq_bin) - idx;
doppler_spectrum = circshift(doppler_spectrum, [0 circ_len 0]);
freq_bin = circshift(freq_bin, [0 circ_len]);

% Fill the DC-notch region around 0 Hz for each receiver (smooth propagation)
for ii = 1:rx_cnt
    doppler_spectrum(ii,:,:) = smooth_dc_notch_fill(squeeze(doppler_spectrum(ii,:,:)), freq_bin, bandstop_band, 0);
end

fprintf('Done.\n');

% ------------------ Return struct ------------------
res.freq_time_prof_allfreq = doppler_spectrum;
res.freq_bin = freq_bin;
res.time_bin = (0:max_frame_len-1)/downsamp_rate;
end
