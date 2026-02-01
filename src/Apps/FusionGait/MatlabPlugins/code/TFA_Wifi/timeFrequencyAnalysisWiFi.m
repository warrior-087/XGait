function res = timeFrequencyAnalysisWiFi(spfx_ges, samp_rate, ...
    uppe_orde, uppe_stop, lowe_orde, lowe_stop, rx_cnt, rx_acnt)
% timeFrequencyAnalysisWiFi  Perform time-frequency analysis on CSI data and generate Doppler spectra
%
% Inputs:
%   spfx_ges   - File name prefix for the data, corresponding to one gesture/motion
%   samp_rate  - Sampling rate (Hz)
%   uppe_orde  - Low-pass filter order
%   uppe_stop  - Low-pass cutoff frequency (Hz)
%   lowe_orde  - High-pass filter order
%   lowe_stop  - High-pass cutoff frequency (Hz)
%   rx_cnt     - Number of receivers
%   rx_acnt    - Number of antennas per receiver
%
% Outputs:
%   res.freq_time_prof_allfreq - Doppler time-frequency profile (receiver × frequency × time)
%   res.freq_bin               - Frequency bin indices (FFT bin indices; can be mapped to Hz if needed)

fprintf('Start generate doppler spectrum...\n');
fprintf('Step 1: Constructing filters...\n');
half_rate = samp_rate / 2;                              % Nyquist frequency
[lu,ld] = butter(uppe_orde,uppe_stop/half_rate,'low');  % Low-pass filter
[hu,hd] = butter(lowe_orde,lowe_stop/half_rate,'high'); % High-pass filter

fprintf('Step 2: Building frequency bins...\n');
freq_bins_unwrap = [0:samp_rate/2-1 -samp_rate/2:-1]'/samp_rate;  % Normalized FFT frequency grid (cycles/sample)
freq_lpf_sele = freq_bins_unwrap <= uppe_stop / samp_rate & freq_bins_unwrap >= -uppe_stop / samp_rate;
freq_lpf_positive_max = sum(freq_lpf_sele(2:length(freq_lpf_sele)/2));    % #bins in positive-frequency region
freq_lpf_negative_min = sum(freq_lpf_sele(length(freq_lpf_sele)/2:end));  % #bins in negative-frequency region

%% ------------------------------------------------
% Step 3: Motion detection
%% ------------------------------------------------
fprintf('Step 3: CSI preprocessing & signal extraction...\n');
% First iterate over all receivers to preprocess signals, and record the maximum sequence length.
max_frame_len = 0;
motion_csi_all = cell(rx_cnt, 1);  % Store preprocessed WiFi CSI-derived signals for each receiver

for ii = 1:rx_cnt
    spth = [spfx_ges, '-r', num2str(ii), '.dat'];
    finfo = dir(spth);

    if isempty(finfo) || finfo.bytes == 0
        warning('Skipping 0KB file %s to protect MCR memory.', spth);
        motion_csi_all{ii} = [];
        continue; % IMPORTANT: never touch an empty file
    end

    [csi_data, ~] = csi_get_all(spth); % Read CSI data

    if isempty(csi_data) || size(csi_data, 1) < 2
        warning('Data is empty or too short for path: %s', spth);
        motion_csi_all{ii} = [];
        continue; % Skip directly; do not enter the try-block below
    end

    try
        % ----------------------------
        % Data preprocessing
        % ----------------------------
        csi_mean = mean(abs(csi_data));
        csi_var  = sqrt(var(abs(csi_data)));
        csi_mean_var_ratio = csi_mean./csi_var;

        % Select the reference antenna (grouped by 30 subcarriers per antenna).
        [~,idx] = max(mean(reshape(csi_mean_var_ratio,[30 rx_acnt]),1));
        csi_data_ref = repmat(csi_data(:,(idx-1)*30+1:idx*30), 1, rx_acnt);

        %% CSI entropy / ratio processing
        % Complex ratio (avoid division by zero)
        eps_val = 1e-12;
        csi_ratio = csi_data ./ (csi_data_ref + eps_val);   % (N × 30*rx_acnt)

        % Remove the 30 subcarriers corresponding to the selected reference antenna.
        target_csi = [csi_ratio(:, 1:30*(idx-1)), csi_ratio(:, 30*idx+1 : 30*rx_acnt)];

        % Keep only one antenna (first 30 subcarriers) for downstream processing.
        target_csi = target_csi(:,1:30);

        % Filtering: low-pass + high-pass (reuse lu/ld and hu/hd from Step 1)
        for jj = 1:size(target_csi, 2)
            target_csi(:, jj) = filter(lu, ld, target_csi(:, jj));
            target_csi(:, jj) = filter(hu, hd, target_csi(:, jj));
        end

        % Detrend / normalize:
        % - Remove per-subcarrier mean to eliminate DC offset
        % - Normalize per-subcarrier standard deviation for scale consistency
        target_csi = target_csi - repmat(mean(target_csi,1), size(target_csi,1), 1);
        col_std = std(target_csi, 0, 1) + eps_val;
        target_csi = target_csi ./ repmat(col_std, size(target_csi,1), 1);

        % PCA dimensionality reduction
        pca_coef = pca(target_csi);            % PCA returns principal component directions (columns)
        target_csi = target_csi * pca_coef(:,1); % Project onto the first principal component (N × 1)

        % Track the maximum signal length across receivers
        max_frame_len = max(max_frame_len, size(target_csi, 1));
        motion_csi_all{ii} = target_csi; % Save the processed signal
    catch err
        disp(err);
        motion_csi_all{ii} = [];
        continue;
    end
end

% Pre-allocate the Doppler spectrum tensor:
%   doppler_spectrum: (receiver × frequency_bin × time)
freq_bin = [0:freq_lpf_positive_max -1*freq_lpf_negative_min:-1];
freq_bin_count = length(freq_bin);
doppler_spectrum = zeros(rx_cnt, freq_bin_count, max_frame_len);

fprintf('Step 4: Process all receivers...\n');
for ii = 1:rx_cnt
    fprintf('  Processing RX #%d...\n', ii);
    processed_csi = motion_csi_all{ii};

    time_instance = 1:length(processed_csi);
    window_size = round(samp_rate/4+1);
    if(~mod(window_size,2))
        window_size = window_size + 1;
    end

    if isempty(processed_csi) || length(processed_csi) < window_size
        fprintf('Warning: processed_csi is empty or too short for TFRSP. Skipping...\n');
        freq_time_prof_allfreq = [];
    else
        try
            % Only compute when the input is valid
            freq_time_prof_allfreq = tfrsp(processed_csi, time_instance, samp_rate, tftb_window(window_size, 'gauss'));
        catch ME
            fprintf('Error occurred while computing freq_time_prof_allfreq: %s\n', ME.message);
            freq_time_prof_allfreq = [];
        end
    end

    if ~isempty(freq_time_prof_allfreq)
        % Keep only the frequency region within the low-pass cutoff and normalize per time frame.
        freq_time_prof = freq_time_prof_allfreq(freq_lpf_sele, :);
        freq_time_prof = abs(freq_time_prof) ./ repmat(sum(abs(freq_time_prof), 1), size(freq_time_prof, 1), 1);

        actual_frames = size(freq_time_prof, 2);

        % Store into doppler_spectrum
        doppler_spectrum(ii, :, 1:actual_frames) = freq_time_prof;
    else
        disp('freq_time_prof_allfreq is empty. Skipping further processing.');
        doppler_spectrum(ii, :, :) = NaN;
    end
end

% Circularly shift frequency bins so that negative frequencies are placed before positive (or vice versa),
% depending on the constructed bin order.
[~,idx] = max(freq_bin);
circ_len = length(freq_bin) - idx;
doppler_spectrum = circshift(doppler_spectrum, [0 circ_len 0]);
freq_bin = circshift(freq_bin, [0 circ_len]);

fprintf('Done.\n');

% ------------------ Return struct ------------------
res.freq_time_prof_allfreq = doppler_spectrum;
res.freq_bin = freq_bin;
res.time_bin = (0:max_frame_len-1)/samp_rate;
end
