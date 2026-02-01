function res = segmentOptimizeGBVP_Fusion( ...
    doppler_spectrum_dict, ...   % struct: modality -> [rx × freq × time]
    pos_list, ...                % [2 × T] or [3 × T]
    ori_list, ...                % [1 × T]
    wave_length_dict, ...        % struct: modality -> lambda
    velocity_bin, ...            % velocity grid
    freq_bin_dict, ...           % struct: modality -> freq_bin
    Tx_pos_dict, ...             % struct: modality -> Tx_pos
    Rx_pos_dict, ...             % struct: modality -> Rx_pos
    rx_cnt_dict, ...             % struct: modality -> rx_cnt
    win_half, ...
    M, ...
    U_bound, ...
    MaxFunctionEvaluations, ...
    norm_type, ...
    lambda ...
)
    % ============================================================
    % Fusion GBVP Optimization (方案 A - 矢量化修复版)
    % ============================================================

    modalities = fieldnames(doppler_spectrum_dict);

    % -------------------------------
    % Segment 配置
    % -------------------------------
    seg_number = floor(size(pos_list, 2));
    ref_mod = modalities{1};
    total_time_len = size(doppler_spectrum_dict.(ref_mod), 3);
    seg_length = max(1, floor(total_time_len / seg_number));
    
    fprintf("seg_number = %d, seg_length = %d\n", seg_number, seg_length);

    % 速度谱容器
    velocity_spectrum = zeros(M, M, seg_number);

    % ===============================
    % Main optimization loop
    % ===============================
    for ii = 1:seg_number
        fprintf("---- Processing Segment %d / %d ----\n", ii, seg_number);

        doppler_seg_all = [];
        VDM_all = [];
        A_all = []; % 用于存储所有 RX 的几何投影

        % ===============================
        % 遍历各模态进行数据准备
        % ===============================
        for mm = 1:length(modalities)
            mod = modalities{mm};

            doppler = doppler_spectrum_dict.(mod);
            lambda_m = wave_length_dict.(mod);
            freq_bin_m = freq_bin_dict.(mod);
            Tx_pos = Tx_pos_dict.(mod);
            Rx_pos = Rx_pos_dict.(mod);
            rx_cnt = rx_cnt_dict.(mod);

            % 1. 获取当前时间段数据并平均
            t_idx = (ii - 1) * seg_length + 1 : min(ii * seg_length, size(doppler,3));
            doppler_seg = mean(doppler(:, :, t_idx), 3);

            % 2. RX 能量归一化并拉平
            doppler_seg_tgt = normalize_rx_energy(doppler_seg);
            % 重要：doppler_seg_tgt(:) 的展开顺序是先 rx 再 freq
            doppler_flat = doppler_seg_tgt(:); 

            % 3. 计算几何矩阵 A 并记录
            A_curr = get_A_matrix(pos_list(:, ii)', Tx_pos, Rx_pos, rx_cnt);
            A_all = [A_all; A_curr]; 

            % 4. 获取原始 VDM (rx_cnt x M x M x F)
            VDM_raw = get_velocity2doppler_mapping_matrix(A_curr, lambda_m, velocity_bin, freq_bin_m, rx_cnt);
            
            % 5. 调整 VDM 维度以匹配 doppler_flat 的拉平顺序
            % 目标顺序：[M, M, (rx_cnt * F)]
            % 原始 VDM 顺序是 [rx, M, M, F]，需 permute 为 [M, M, rx, F] 然后 reshape
            VDM_permuted = permute(VDM_raw, [2, 3, 1, 4]); 
            VDM_reshaped = reshape(VDM_permuted, M, M, []);

            % 6. 拼接
            doppler_seg_all = [doppler_seg_all; doppler_flat];    
            VDM_all = cat(3, VDM_all, VDM_reshaped);              
        end

        % ===============================
        % Torso 中心速度搜索 (传入真实的 A_all)
        % ===============================
        % 注意：此处需确保您的 subtract_VDM 函数已更新以接收 A_all
        [VDM_sub, v_torso, bounds] = subtract_VDM_by_torso_speed_Fusion( ...
            pos_list(:, ii)', VDM_all, doppler_seg_all, velocity_bin, win_half, A_all);

        % 健壮性检查：防止 imrotate 因 NaN 崩溃
        if any(isnan(v_torso)) || any(isinf(v_torso))
            v_torso = [0; 0]; 
        end

        N = size(VDM_sub, 1);
        Ubound = U_bound(bounds(1):bounds(2), bounds(3):bounds(4));

        % ===============================
        % Optimization (矢量化调用)
        % ===============================
        % 初始值微量扰动，避免全零进入死循环
        P0 = Ubound * 0.01; 
        
        [P, ~, exitFlag, ~] = fmincon( ...
            @(P) DVM_target_func_fusion(P, VDM_sub, lambda, doppler_seg_all, norm_type), ...
            P0, ...
            [], [], [], [], ...
            zeros(N, N), ...
            Ubound, ...
            [], ...
            optimoptions('fmincon', 'Algorithm', 'sqp', ...
                'MaxFunctionEvaluations', MaxFunctionEvaluations) ...
        );

        % ===============================
        % 坐标对齐与回写
        % ===============================
        G_full = zeros(M, M);
        G_full(bounds(1):bounds(2), bounds(3):bounds(4)) = P;

        theta = ori_list(ii);
        if isnan(theta), theta = 0; end
        
        % 核心：将 P 移回中心并旋转对齐
        G_norm = torso_center_and_normalize(G_full, v_torso, velocity_bin, theta, M);
        velocity_spectrum(:, :, ii) = G_norm;
    end

    %% ===============================
    %% Temporal resample (保持 30 帧)
    %% ===============================
    %T_target = 30;
    %res = zeros(M, M, T_target);
    %for i = 1:M
    %    for j = 1:M
    %        vec = squeeze(velocity_spectrum(i, j, :));
    %        res(i, j, :) = resample(vec, T_target, seg_number);
    %    end
    %end
    res = velocity_spectrum;
end
