function A = get_A_matrix(torso_pos, Tx_pos, Rx_pos, rxcnt)
% GET_A_MATRIX  Compute the geometric projection matrix A for velocity-to-Doppler mapping.
%
% Inputs:
%   torso_pos - Torso (body center) position (1 x 2), e.g., [x, y]
%   Tx_pos    - Transmitter position (1 x 2)
%   Rx_pos    - Receiver positions (N_rx x 2), one receiver per row
%   rxcnt     - Number of receivers actually used (<= size(Rx_pos,1))
%
% Output:
%   A         - Geometry matrix (rxcnt x 2) describing how a 2D velocity vector [vx, vy]
%               projects onto the bistatic Doppler (i.e., link-direction radial velocity)
%               for each Tx-Rx pair.

    if rxcnt > size(Rx_pos,1)
        error('Error Rx Count!')
    end

    A = zeros(rxcnt, 2);

    for ii = 1:rxcnt
        dis_torso_tx = sqrt((torso_pos - Tx_pos) * (torso_pos - Tx_pos)');         % Distance: Tx -> torso
        dis_torso_rx = sqrt((torso_pos - Rx_pos(ii,:)) * (torso_pos - Rx_pos(ii,:))'); % Distance: Rx -> torso

        % (torso_pos - Tx_pos) / dis_torso_tx gives the unit vector pointing from Tx to torso.
        % (torso_pos - Rx_pos(ii,:)) / dis_torso_rx gives the unit vector pointing from Rx to torso.
        % Their sum forms the bistatic projection direction used to map 2D velocity onto Doppler shift
        % along the Tx->torso and Rx->torso paths.
        A(ii,:) = (torso_pos - Tx_pos) / dis_torso_tx + ...
                  (torso_pos - Rx_pos(ii,:)) / dis_torso_rx;
    end
end
