function res = segmentOptimizeGBVP(doppler_spectrum, pos_list, ori_list, wave_length, velocity_bin, freq_bin,...
    Tx_pos, Rx_pos, rx_cnt, win_half, M, U_bound, MaxFunctionEvaluations, norm, lambda)
% segmentOptimizeGBVP  Segment-wise optimization of Doppler spectra to recover a 2D velocity spectrum
%
% Inputs:
%   doppler_spectrum       - Doppler spectrum tensor, size [rx_cnt × F × T] (receiver × frequency × time)
%   pos_list               - Torso positions per segment, size [2 × seg_number]
%   ori_list               - Torso orientations per segment (degrees), length seg_number
%   wave_length            - Wavelength of the carrier (meters)
%   velocity_bin           - Discrete velocity grid (1 × M), assumed symmetric, e.g. [-vmax,...,0,...,vmax]
%   freq_bin               - Doppler frequency bins (F × 1) or bin indices (must match get_velocity2doppler_mapping_matrix)
%   Tx_pos                 - Transmitter position(s) used by get_A_matrix
%   Rx_pos                 - Receiver position(s) used by get_A_matrix
%   rx_cnt                 - Number of receivers (links)
%   win_half               - Half window size for cropping around the estimated torso velocity (in velocity-bin indices)
%   M                      - Velocity grid size (M × M)
%   U_bound                - Upper bound constraint matrix (M × M), limiting each velocity cell magnitude
%   MaxFunctionEvaluations - Max function evaluations for fmincon
%   norm                   - Regularization type (0 = L0-like, 1 = L1-like; depends on DVM_target_func implementation)
%   lambda                 - Regularization weight (passed into DVM_target_func)
%
% Output:
%   res                    - Estimated velocity spectrum after time resampling, size [M × M × T_target]
%
% Notes:
% - This function runs per-segment optimization in parallel (parfor).
% - For each segment, it builds a Velocity-to-Doppler Mapping (VDM), crops a local window around
%   the estimated torso velocity to reduce the search space, then solves a constrained optimization
%   (fmincon) to estimate the velocity spectrum for that segment.

    if isempty(gcp('nocreate'))
        parpool('local');
    end

    % Segment the sequence according to the number of torso poses provided
    seg_number = floor(size(pos_list, 2));
    fprintf("seg_number")
    disp(seg_number)

    % Segment length in frames (fallback to at least 1)
    seg_length = max(1, floor(size(doppler_spectrum, 3) / seg_number));
    fprintf("seg_length")
    disp(seg_length)

    % Keep a local copy of the spectrum
    doppler_spectrum_ges = doppler_spectrum;

    % Allocate velocity spectrum storage (per segment)
    velocity_spectrum = zeros(M, M, seg_number);

    % === Parallel optimization for each segment ===
    parfor ii = 1:seg_number
        disp(ii)

        % Build A matrix for the current torso position and the Tx/Rx geometry
        A = get_A_matrix(pos_list(:,ii)', Tx_pos, Rx_pos, rx_cnt);

        % ---- Extract one time segment ----
        doppler_spectrum_seg = doppler_spectrum_ges(:, :, ...
            (ii - 1) * seg_length + 1 : ii * seg_length);

        % Time-average within the segment to obtain the target Doppler spectrum
        doppler_spectrum_seg_tgt = mean(doppler_spectrum_seg, 3);

        % ---- Inter-receiver normalization (path-loss compensation) ----
        for jj = 2:size(doppler_spectrum_seg_tgt, 1)
            if any(doppler_spectrum_seg_tgt(jj, :))
                doppler_spectrum_seg_tgt(jj, :) = ...
                    doppler_spectrum_seg_tgt(jj, :) * ...
                    sum(doppler_spectrum_seg_tgt(1, :)) / ...
                    sum(doppler_spectrum_seg_tgt(jj, :));
            end
        end

        % ---- Compute Velocity-to-Doppler Mapping (VDM) ----
        VDM = permute(get_velocity2doppler_mapping_matrix(A, wave_length, ...
            velocity_bin, freq_bin, rx_cnt), [2,3,1,4]);    % M×M×rx_cnt×F (example: 20×20×rx_cnt×121)

        % ---- Crop around the estimated torso velocity to reduce search space ----
        [VDM_sub, v_torso, bounds] = subtract_VDM_by_torso_speed(A, VDM, doppler_spectrum_seg_tgt, ...
            freq_bin, velocity_bin, wave_length, win_half);

        N = size(VDM_sub, 1);
        Ubound = U_bound(bounds(1):bounds(2), bounds(3):bounds(4));

        % ---- Run constrained optimization (fmincon) ----
        [P, fval, exitFlag, output] = fmincon( ...
            @(P) DVM_target_func(P, VDM_sub, lambda, ...
                                 doppler_spectrum_seg_tgt, ...
                                 size(doppler_spectrum_seg_tgt, 1), norm), ...
            zeros(N, N), ...   % Initial value
            [], [], ...        % Linear inequality constraints
            [], [], ...        % Linear equality constraints
            zeros(N, N), ...   % Lower bound
            Ubound, ...        % Upper bound
            [], ...            % Nonlinear constraints (unused)
            optimoptions('fmincon', 'Algorithm', 'sqp', ...
                         'MaxFunctionEvaluations', MaxFunctionEvaluations)); % Options

        % ---- Diagnostic: reconstruction error (RMSE / NRMSE) ----
        recon_pages = pagemtimes( ...
            permute(reshape(VDM_sub, N*N, rx_cnt, size(VDM_sub,4)), [2,1,3]), ...
            reshape(P, [N*N, 1, 1]) ...
        );
        recon = squeeze(recon_pages);   % recon: rx_cnt × F
        resid = recon - doppler_spectrum_seg_tgt;
        rmse = sqrt(mean(resid(:).^2));
        ref_norm = sqrt(mean(doppler_spectrum_seg_tgt(:).^2)); % Normalize by observation energy
        nrmse = rmse / ref_norm;
        fprintf('Segment %d RMSE: %.3e (NRMSE=%.2f%%)\n', ii, rmse, nrmse*100);

        % ---- Write the local window back into the full M×M grid ----
        G_full = zeros(M, M);
        G_full(bounds(1):bounds(2), bounds(3):bounds(4)) = P;

        % ---- Torso centering + direction alignment + energy normalization ----
        theta = ori_list(ii);  % Torso orientation (degrees)
        G_norm = torso_center_and_normalize(G_full, v_torso, velocity_bin, theta, M);

        % Save segment result
        velocity_spectrum(:,:,ii) = G_norm;

        % exitFlag can be inspected for debugging if needed
        exitFlag;
    end

    % ---- Resample along the time/segment axis to a fixed target length ----
    T_target = 30; % Could be made a function argument
    G_resampled = zeros(M, M, T_target);

    for i = 1:M
        for j = 1:M
            vec = squeeze(velocity_spectrum(i, j, :));
            G_resampled(i, j, :) = resample(vec, T_target, seg_number);
        end
    end

    res = G_resampled;
end
