function VDM = get_velocity2doppler_mapping_matrix(A, wave_length, ...
        velocity_bin, freq_bin, rx_cnt)

% get_velocity2doppler_mapping_matrix  Build the Velocity-to-Doppler Mapping (VDM) tensor.
%
% Inputs:
%   A            - Geometry/projection matrix (rx_cnt x 2) computed by get_A_matrix.
%                  Each row A(ii,:) defines how a 2D velocity [vx; vy] projects onto
%                  the radial (link) velocity of receiver/link ii.
%   wave_length  - Signal wavelength (meters), i.e., lambda = c / f_c.
%   velocity_bin - Discrete velocity grid (1 x M), spanning [Vmin, Vmax] (typically symmetric).
%   freq_bin     - Discrete Doppler bins (1 x F). Units must match plcr_hz below (Hz or bin values).
%                  In this implementation, plcr_hz is rounded to an integer and matched to the
%                  nearest element in freq_bin.
%   rx_cnt       - Number of receivers/links.
%
% Output:
%   VDM          - 4D tensor of size (rx_cnt x M x M x F), where:
%                  - First dimension indexes receivers/links.
%                  - (i,j) corresponds to a velocity hypothesis:
%                        vx = velocity_bin(i), vy = velocity_bin(j).
%                  - Fourth dimension indexes Doppler frequency bins.
%                  - VDM(ii,i,j,f) = 1 indicates that this (vx,vy) hypothesis maps to
%                    Doppler bin f for link ii; otherwise it is 0.
%
% Notes:
% - This VDM is a "hard assignment" mapping (one-hot over frequency bins).
% - Velocities whose projected Doppler falls outside [min(freq_bin), max(freq_bin)] are skipped (all zeros).

    if size(A,1) ~= rx_cnt
        error('Error Rx Count!');
    end

    F = size(freq_bin, 2);
    M = size(velocity_bin, 2);
    freq_min = min(freq_bin);
    freq_max = max(freq_bin);

    VDM = zeros(rx_cnt, M, M, F);

    % For each link
    for ii = 1:rx_cnt
        for i = 1:M  % vx index
            for j = 1:M  % vy index

                % Project the 2D velocity (vx, vy) onto link ii and convert to Doppler (Hz-like units).
                % Here we round to the nearest integer to match discrete Doppler bins.
                plcr_hz = round(A(ii,:) * [velocity_bin(i) velocity_bin(j)]' / wave_length);

                % If the projected Doppler is out of the observable range, treat as invalid (leave zeros).
                if plcr_hz > freq_max || plcr_hz < freq_min
                    % VDM(ii,i,j,:) = 1e10 * ones(1, size(VDM,4)); % legacy "penalty" option (disabled)
                    continue;
                end

                % Find the nearest Doppler bin index (freq_bin may include negative values)
                [~, idx] = min(abs(freq_bin - plcr_hz));
                VDM(ii, i, j, idx) = 1;
            end
        end
    end
end
