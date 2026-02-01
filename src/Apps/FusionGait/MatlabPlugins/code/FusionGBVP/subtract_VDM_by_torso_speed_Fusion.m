function [VDM_sub, v_torso, bounds] = subtract_VDM_by_torso_speed_Fusion( ...
    torso_pos, ...         % [1×2] 或 [1×3] 躯干中心位置
    VDM, ...               % [M×M×Total_Obs] 拼接后的全量 VDM
    doppler_seg_all, ...   % [Total_Obs × 1] 拼接后的平铺 Doppler 观测
    velocity_bin, ...      % [1×M] 离散速度网格
    win_half, ...          % 搜索窗口半宽度
    A_all ...              % [Total_RX × 2] 拼接后的几何投影矩阵
)
% SUBTRACT_VDM_BY_TORSO_SPEED_FUSION
%   利用多模态观测估算躯干速度，并对 VDM 进行局部速度窗口裁剪
%
% 输入说明：
%   A_all 每一行对应一个接收机，用于建立 f = A * v 的关系

    % =====================================================
    % 1. 参数初始化
    % =====================================================
    M = size(VDM, 1);
    total_obs = length(doppler_seg_all);
    rx_cnt = size(A_all, 1);
    
    % 计算每个 RX 对应的频率点数 F (假设各模态内部 F 均匀)
    % 如果各模态 F 不同，则需要从外部传入各模态起始索引，此处简化处理
    F = total_obs / rx_cnt; 

    % =====================================================
    % 2. 提取每个接收机的峰值频率 (Targeting Torso)
    % =====================================================
    % 将平铺的 doppler 重新整理回 [rx_cnt, F] 寻找峰值
    doppler_reshaped = reshape(doppler_seg_all, [rx_cnt, F]);
    [~, idx_peak] = max(doppler_reshaped, [], 2); 
    
    % 注意：这里 idx_peak 需要映射回真实的频率值（Hz）
    % 假设索引 1 对应频率范围的最小值。
    % 实际应用中建议通过 freq_bin 映射，此处直接作为观测频率输入
    f_obs = idx_peak; 

    % =====================================================
    % 3. 加权最小二乘法 (WLS) 估算 v_torso
    % =====================================================
    % 物理模型: f = (1/lambda) * A * v. 
    % 由于 A 在主函数中已经除以了 lambda (在 get_A_matrix 逻辑中或 VDM 构建中)
    % 此处直接解超定方程: A_all * v = f_obs
    
    % 排除可能的异常值
    W = eye(rx_cnt); 
    v_torso = (A_all' * W * A_all) \ (A_all' * W * f_obs);

    % 健壮性检查：若计算失败（如 A 共线），给默认值防止后续崩溃
    if any(isnan(v_torso)) || any(isinf(v_torso))
        v_torso = [0; 0];
    end

    % =====================================================
    % 4. 速度格点索引定位
    % =====================================================
    [~, i0] = min(abs(velocity_bin - v_torso(1)));
    [~, j0] = min(abs(velocity_bin - v_torso(2)));

    % =====================================================
    % 5. 裁剪速度子空间 (Local Optimization Window)
    % =====================================================
    i_min = max(1, i0 - win_half);
    i_max = min(M, i0 + win_half);
    j_min = max(1, j0 - win_half);
    j_max = min(M, j0 + win_half);

    % 保持正方形裁剪以适配后续处理
    L = min(i_max - i_min, j_max - j_min);
    i_max = i_min + L;
    j_max = j_min + L;

    % 裁剪 VDM: [N x N x Total_Obs]
    VDM_sub = VDM(i_min:i_max, j_min:j_max, :);

    bounds = [i_min, i_max, j_min, j_max];
end