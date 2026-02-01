function loss = DVM_target_func_fusion(P, VDM_sub, lambda, doppler_all, norm_type)
    N = size(P, 1);
    Obs = size(VDM_sub, 3);
    
    % 将 VDM 重塑为 [Obs x N*N]
    % 将 P 拉平为 [N*N x 1]
    V_matrix = reshape(VDM_sub, [N*N, Obs])'; 
    pred = V_matrix * P(:);
    
    % L2 残差
    resid = pred - doppler_all;
    loss_data = (resid' * resid) / Obs;

    % 正则化
    if norm_type == 0
        reg = lambda * sum(P(:) > 1e-4); % 软 L0
    else
        reg = lambda * sum(abs(P(:)));
    end
    loss = loss_data + reg;
end