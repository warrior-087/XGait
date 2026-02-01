function G_norm = torso_center_and_normalize(G_full, v_torso, velocity_bin, angle_ref, M)
% torso_center_and_normalize
%
% Inputs:
%   G_full       - Raw velocity spectrum (M x M)
%   v_torso      - Torso velocity vector [vx; vy]
%   velocity_bin - Discrete velocity grid (1 x M), assumed symmetric, e.g., [-vmax,...,0,...,vmax]
%   angle_ref    - Reference orientation (degrees), consistent with imrotate convention
%   M            - Grid size
%
% Output:
%   G_norm       - Centered, direction-aligned, and energy-normalized spectrum (M x M)

    % === Step 1: Find the velocity-bin indices corresponding to the torso velocity ===
    [~, i0] = min(abs(velocity_bin - v_torso(1))); % x-axis (vx)
    [~, j0] = min(abs(velocity_bin - v_torso(2))); % y-axis (vy)
    center_idx = ceil(M/2);

    shift_x = center_idx - i0;  % Column shift (x corresponds to columns)
    shift_y = center_idx - j0;  % Row shift (y corresponds to rows)

    % circshift note: MATLAB matrices are indexed as (row, col) = (y, x)
    G_shifted = circshift(G_full, [shift_y, shift_x]);

    % === Step 2: Rotate to a unified reference direction ===
    % Torso direction angle from velocity vector
    ang = atan2(v_torso(2), v_torso(1)); % radians
    ang_deg = rad2deg(ang);

    % Rotation angle to align torso direction with angle_ref
    theta = -(ang_deg - angle_ref);

    % imrotate: negative angle means clockwise rotation by default
    G_rot = imrotate(G_shifted, theta, 'bilinear', 'crop');

    % === Step 3: Energy normalization ===
    s = sum(G_rot(:));
    if s > 0
        G_norm = G_rot / s;
    else
        G_norm = G_rot;
    end

end
