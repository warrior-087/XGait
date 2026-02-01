function [VDM_sub, v_torso, bounds] = subtract_VDM_by_torso_speed(A, VDM, doppler_seg_tgt, freq_bin, ...
    velocity_bin, wave_length, win_half)

rx_cnt = size(doppler_seg_tgt, 1);
F = size(doppler_seg_tgt, 2);

% 1) For each receiver, pick the peak Doppler bin (maximum energy)
[~, idx_peak] = max(doppler_seg_tgt, [], 2);    % idx_peak: rx_cnt x 1
f_peaks = freq_bin(idx_peak);                   % Map peak-bin indices to peak Doppler values

% 2) Weighted least squares to solve: A * v = f_peaks * wave_length
%    (Weights can be set based on per-receiver reliability, e.g., inverse energy variance)
W = eye(rx_cnt);
v_torso = (A' * W * A) \ (A' * W * f_peaks(:) * wave_length);  % 2 x 1

% 3) Find the indices of the torso velocity on the 2D velocity grid
[~, i0] = min(abs(velocity_bin - v_torso(1)));  % vx index
[~, j0] = min(abs(velocity_bin - v_torso(2)));  % vy index

% 4) Crop a local window in VDM centered at the torso velocity (controlled by win_half)
M = size(VDM, 1);

is_min = max(1, i0 - win_half);
is_max = min(M, i0 + win_half);
js_min = max(1, j0 - win_half);
js_max = min(M, j0 + win_half);

% Ensure a square crop (L x L) for downstream consistency
h = is_max - is_min + 1;
w = js_max - js_min + 1;
L = min([h, w]);

is_max = min(is_min + L - 1, M);
js_max = min(js_min + L - 1, M);

% 5) Crop VDM
VDM_sub = VDM(is_min:is_max, js_min:js_max, :, :);
bounds = [is_min, is_max, js_min, js_max];
end
