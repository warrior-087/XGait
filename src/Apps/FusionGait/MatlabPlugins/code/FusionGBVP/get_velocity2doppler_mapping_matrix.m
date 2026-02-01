% Input:(If rx_cnt=2)
% A: 2*2
% velocity_bin: 1*M
% freq_bin: 1*F
% Output: 2*M*M*F, component is ...
function VDM = get_velocity2doppler_mapping_matrix(A, wave_length,...
        velocity_bin, freq_bin, rx_cnt)

% 构建速度到多普勒频率的映射矩阵 (Velocity-to-Doppler Mapping, VDM)
%
% 输入：
%   A            - 几何矩阵 (rx_cnt x 2)，由 get_A_matrix 计算得到，
%                  描述速度向量在 Tx-Rx 路径上的投影方向
%   wave_length  - 信号波长 (c/f)
%   velocity_bin - 离散速度取值 (1 x M)，覆盖速度范围 [Vmin, Vmax]
%   freq_bin     - 离散的 Doppler 频率 bins (1 x F)，FFT 频率索引
%   rx_cnt       - 接收机数量
%
% 输出：
%   VDM - 维度为 (rx_cnt x M x M x F) 的 4D 张量
%         其中：
%         - 第一个维度对应接收机编号
%         - (i,j) 对应一个速度网格点 (vx=velocity_bin(i), vy=velocity_bin(j))
%         - 第四个维度对应频率 bins
%         - VDM(ii,i,j,f) = 1 表示该速度点投影到该频率 bin
%           否则为 0（或 1e10 作为非法惩罚值）

    if size(A,1) ~= rx_cnt
        error('Error Rx Count!');
    end
    F = size(freq_bin, 2);
    M = size(velocity_bin, 2);
    freq_min = min(freq_bin);
    freq_max = max(freq_bin);
    
    VDM = zeros(rx_cnt, M, M, F);

    % For Each Link
    for ii = 1:rx_cnt
        for i = 1:M %x方向速度
            for j = 1:M %y方向速度
                
                % 计算当前速度 (vx, vy) 在 Tx-Rx 链路上的 Doppler 投影
                plcr_hz = round(A(ii,:) * [velocity_bin(i) velocity_bin(j)]'...
                    / wave_length);

                % 如果投影的 Doppler 超出可观测频率范围，则认为无效
                if plcr_hz > freq_max || plcr_hz < freq_min
                    %VDM(ii,i,j,:) = 1e10*ones(1,size(VDM,4));
                    continue;
                end

                % 计算频率 bin 的索引
                % 因为 freq_bin 可能从负数开始，所以需要偏移 freq_min
                %idx = plcr_hz + 1 - freq_min;

                [~, idx] = min(abs(freq_bin - plcr_hz));
                VDM(ii,i,j,idx) = 1;
            end
        end
    end
end