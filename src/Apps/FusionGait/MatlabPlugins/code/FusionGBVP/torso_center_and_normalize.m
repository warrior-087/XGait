function G_norm = torso_center_and_normalize(G_full, v_torso, velocity_bin, angle_ref, M)
% torso_center_and_normalize
% 输入:
%   G_full       - 原始速度谱 (M×M)
%   v_torso      - 躯干速度 [vx; vy]
%   velocity_bin - 离散速度格点 (1×M)，假设对称，如 [-vmax,...,0,...,vmax]
%   angle_ref    - 躯干方向 (度数, 与 imrotate 保持一致)
%   M            - 网格大小
%
% 输出:
%   G_norm       - 标准化速度谱 (M×M)

    % === Step 1: 找到 torso 对应的速度格点 ===
    [~, i0] = min(abs(velocity_bin - v_torso(1))); % x方向
    [~, j0] = min(abs(velocity_bin - v_torso(2))); % y方向
    center_idx = ceil(M/2);

    shift_x = center_idx - i0;  % 列方向偏移
    shift_y = center_idx - j0;  % 行方向偏移

    % circshift (注意 MATLAB 矩阵：第一维是行，对应y；第二维是列，对应x)
    G_shifted = circshift(G_full, [shift_y, shift_x]);

    % === Step 2: 旋转到统一参考方向 ===
    % v_torso 方向角
    ang = atan2(v_torso(2), v_torso(1)); % 弧度
    ang_deg = rad2deg(ang);

    % 要旋转的角度（使 torso 方向对齐到 angle_ref）
    theta = -(ang_deg - angle_ref);  

    % imrotate 默认顺时针是负角度
    G_rot = imrotate(G_shifted, theta, 'bilinear', 'crop');

    % === Step 3: 能量归一化 ===
    s = sum(G_rot(:));
    if s > 0
        G_norm = G_rot / s;
    else
        G_norm = G_rot;
    end

end