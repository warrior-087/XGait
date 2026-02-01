function floss = DVM_target_func(P, ...
    VDM, lambda, doppler_spectrum_seg_tgt, rx_cnt, norm)

    if (size(P,1) ~= size(P,2))
        return;
    end

    % Initialize variables
    M = size(P,1);                         % Velocity grid size (P is M x M)
    F = size(doppler_spectrum_seg_tgt, 2); % Number of Doppler frequency bins

    % ------------------------------------------------------------
    % Construct the approximated Doppler spectrum from P and VDM
    %
    % Conceptually:
    %   - P(i,j) represents the energy/weight at velocity hypothesis (vx_i, vy_j).
    %   - VDM(i,j,r,f) indicates how that velocity hypothesis maps to Doppler bin f
    %     on receiver/link r (typically a one-hot mapping along f).
    %   - Summing P(i,j) over all (i,j) after weighting by VDM yields an estimated
    %     Doppler spectrum for each receiver: (rx_cnt x F).
    %
    % This implementation avoids explicitly expanding P to (M x M x rx_cnt x F),
    % and instead uses reshape + matrix multiplication for efficiency.
    % ------------------------------------------------------------
    doppler_spectrum_seg_approximate = zeros(rx_cnt, F);
    for r = 1:rx_cnt
        % Reshape VDM(:,:,r,:) to (M*M) x F
        Vr = reshape(VDM(:,:,r,:), [M*M, F]);

        % Reshape P to (M*M) x 1
        Pr = reshape(P, [M*M, 1]);

        % Estimated spectrum for receiver r: 1 x F
        doppler_spectrum_seg_approximate(r,:) = Pr.' * Vr;
    end

    % ------------------------------------------------------------
    % Loss function: EMD-like distance + optional sparsity regularization
    % ------------------------------------------------------------

    % EMD-like distance (cumulative L1 difference)
    % Multiply by an upper-triangular all-ones matrix to compute cumulative sums,
    % which approximates the 1D Earth Mover's Distance between two spectra.
    floss = 0;
    for ii = 1:rx_cnt
        if any(doppler_spectrum_seg_tgt(ii,:))  % Only evaluate if target spectrum is non-zero
            diff = doppler_spectrum_seg_approximate(ii,:) - doppler_spectrum_seg_tgt(ii,:);
            floss = floss + sum(abs(diff * triu(ones(F,F), 0)));
        end
    end

    % Regularization term
    if norm == 1
        % L1 regularization (encourages sparsity)
        floss = floss + lambda * sum(P(:));
    elseif norm == 0
        % L0-like regularization (counts non-zero entries; non-convex)
        floss = floss + lambda * sum(P(:) ~= 0);
    end

end
