function freq_time_prof = smooth_dc_notch_fill(freq_time_prof, freq_bin, notch_width, scale)
% smooth_dc_notch_fill  Smoothly fill the DC-notch region around 0 Hz.
%
% This utility is designed for a frequency axis ordered as [-max ... 0 ... +max].
% It fills bins within +/- notch_width by propagating values from the nearest
% "outer" neighbor bins and applying a local temporal smoothing window.
%
% Inputs:
%   freq_time_prof : F x T matrix (frequency x time), typically a Doppler TFR magnitude
%   freq_bin       : F x 1 vector, frequency bins ordered as [-max ... 0 ... +max]
%   notch_width    : Scalar frequency width around DC to fill (same unit as freq_bin, e.g., Hz)
%   scale          : Multiplicative attenuation factor applied during filling (< 1 is common).
%                   Default: 1 (no attenuation).
%
% Output:
%   freq_time_prof : F x T matrix after smooth DC-notch filling and column-wise normalization

if nargin < 4
    scale = 1;
end

freq_bin = freq_bin(:);
F = length(freq_bin);
T = size(freq_time_prof,2);

% ---------- Locate indices ----------
zero_idx = find(freq_bin == 0, 1);
neg_idx  = find(freq_bin < 0 & abs(freq_bin) <= notch_width);  % Bins to fill on the negative side
pos_idx  = find(freq_bin > 0 & freq_bin <= notch_width);       % Bins to fill on the positive side

% ---------- Fill negative side (from most negative -> towards 0) ----------
if ~isempty(neg_idx)
    neg_sorted = sort(neg_idx, 'ascend'); % From more negative to closer to 0
    for k = 1:length(neg_sorted)
        r = neg_sorted(k);

        if k == 1
            % Outer neighbor: one bin further away from DC (more negative).
            % If out of range, fall back to itself.
            prev_row = max(r - 1, 1);
        end

        % Initialize the whole row from the outer neighbor (scaled)
        freq_time_prof(r,:) = freq_time_prof(prev_row,:) * scale;

        % Temporal smoothing: replace each time sample by the mean of a local window
        cols = size(freq_time_prof,2);
        win  = 10; % half-window size in time samples
        for j = 1:cols
            j1 = max(1, j - win);
            j2 = min(cols, j + win);
            seg = freq_time_prof(prev_row, j1:j2);
            freq_time_prof(r,j) = mean(seg, 'omitnan') * scale;
        end

        % Next inner row uses the current filled row as its "outer" neighbor
        prev_row = r;
    end
end

% ---------- Fill positive side (from most positive -> towards 0) ----------
if ~isempty(pos_idx)
    pos_sorted = sort(pos_idx, 'descend'); % From more positive to closer to 0
    for k = 1:length(pos_sorted)
        r = pos_sorted(k);

        if k == 1
            % Outer neighbor: one bin further away from DC (more positive).
            % If out of range, fall back to itself.
            prev_row = min(r + 1, F);
        end

        % Initialize the whole row from the outer neighbor (scaled)
        freq_time_prof(r,:) = freq_time_prof(prev_row,:) * scale;

        % Temporal smoothing using the same local time window
        cols = size(freq_time_prof,2);
        win  = 10; % half-window size in time samples
        for j = 1:cols
            j1 = max(1, j - win);
            j2 = min(cols, j + win);
            seg = freq_time_prof(prev_row, j1:j2);
            freq_time_prof(r,j) = mean(seg, 'omitnan') * scale;
        end

        % Next inner row uses the current filled row as its "outer" neighbor
        prev_row = r;
    end
end

% ---------- Fill the 0 Hz row ----------
if ~isempty(neg_idx) && ~isempty(pos_idx)
    % If both sides exist, blend the closest-to-DC filled rows from each side
    freq_time_prof(zero_idx,:) = scale * (freq_time_prof(neg_sorted(end),:) + freq_time_prof(pos_sorted(end),:));
elseif ~isempty(neg_idx)
    % Only negative side exists
    freq_time_prof(zero_idx,:) = freq_time_prof(neg_sorted(end),:);
elseif ~isempty(pos_idx)
    % Only positive side exists
    freq_time_prof(zero_idx,:) = freq_time_prof(pos_sorted(end),:);
end

% ---------- Column-wise normalization (per time frame) ----------
colsum = sum(freq_time_prof, 1);
colsum(colsum == 0) = 1;
freq_time_prof = freq_time_prof ./ colsum;

end
