function doppler_norm = normalize_rx_energy(doppler_rx)
% normalize_rx_energy
% ===================
% 对 Doppler 观测进行 RX 间能量归一化（path-loss 补偿）
%
% 输入：
%   doppler_rx : [rx_cnt × freq_bin]
%
% 输出：
%   doppler_norm : [rx_cnt × freq_bin]
%
% 归一化策略：
%   - 以第 1 路 RX 作为参考
%   - 其余 RX 的频谱整体缩放，使能量和第 1 路一致
%
% 注意：
%   - 不做跨模态归一化（fusion 在更高层完成）
%   - 保留频谱形状，仅做整体幅度缩放
%

    doppler_norm = doppler_rx;

    if isempty(doppler_rx)
        return;
    end

    rx_cnt = size(doppler_rx, 1);

    % 参考 RX 能量
    ref_energy = sum(doppler_rx(1, :));

    if ref_energy <= 0 || ~isfinite(ref_energy)
        % 无法归一化，直接返回原始数据
        return;
    end

    for r = 2:rx_cnt
        cur_energy = sum(doppler_rx(r, :));

        if cur_energy > 0 && isfinite(cur_energy)
            doppler_norm(r, :) = doppler_rx(r, :) * ...
                (ref_energy / cur_energy);
        else
            % 若当前 RX 无有效能量，保持原样（或可置零）
            doppler_norm(r, :) = doppler_rx(r, :);
        end
    end
end
