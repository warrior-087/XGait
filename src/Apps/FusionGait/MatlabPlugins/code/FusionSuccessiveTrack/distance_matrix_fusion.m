function dist_mtx = distance_matrix_fusion(curr_loc, tx_all, rx_all)
% distance_matrix_fusion  Compute the per-link projection/gradient matrix (supports per-link TX).
%
% This function builds the geometric direction vectors used for bistatic Doppler projection.
% For each link k, it computes:
%   u_tx = (curr_loc - tx_k) / ||curr_loc - tx_k||   (unit vector from TX to current location)
%   u_rx = (curr_loc - rx_k) / ||curr_loc - rx_k||   (unit vector from RX to current location)
% and returns:
%   dist_mtx(k,:) = u_tx' + u_rx'    (a 1x2 row vector)
%
% Inputs:
%   curr_loc - Current target location (2x1 or 1x2)
%   tx_all   - TX locations for all links (2xN), one column per link
%   rx_all   - RX locations for all links (2xN), one column per link
%
% Output:
%   dist_mtx - Per-link 2D projection/gradient matrix (N x 2), one row per link

N = size(tx_all, 2);
dist_mtx = zeros(N, 2);

for k = 1:N
    tx = tx_all(:, k);
    rx = rx_all(:, k);

    d_tx = (curr_loc - tx) / norm(curr_loc - tx);
    d_rx = (curr_loc - rx) / norm(curr_loc - rx);

    % 2D bistatic projection direction (row vector)
    dist_mtx(k, :) = d_tx.' + d_rx.';
end
end
