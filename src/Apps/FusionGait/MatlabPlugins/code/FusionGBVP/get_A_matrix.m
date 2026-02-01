function A = get_A_matrix(torso_pos, Tx_pos, Rx_pos, rxcnt)
%GET_A_MATRIX 计算速度到多普勒频移的几何投影矩阵 A
%
% 输入：
%   torso_pos - 人体中心位置 (1x2)，例如 [x,y]
%   Tx_pos    - 发射机位置 (1x2)
%   Rx_pos    - 接收机位置集合 (N_rx x 2)，每行一个接收机坐标
%   rxcnt     - 实际使用的接收机数量 (≤ size(Rx_pos,1))
%
% 输出：
%   A - 几何矩阵 (rxcnt x 2)，描述“速度向量”如何映射到 Doppler 频移

    if rxcnt > size(Rx_pos,1)
        error('Error Rx Count!')
    end
    A = zeros(rxcnt,2);
    
    for ii = 1:rxcnt
        dis_torso_tx = sqrt((torso_pos-Tx_pos) * (torso_pos-Tx_pos)');%发射机到人体中心的距离
        dis_torso_rx = sqrt((torso_pos-Rx_pos(ii,:)) * (torso_pos-Rx_pos(ii,:))');%接收机到人体中心的距离
        
        %     - (torso_pos - Tx_pos)/dis_torso_tx  表示 发射机 → 人体 的单位向量
        %     - (torso_pos - Rx_pos(ii,:))/dis_torso_rx 表示 接收机 → 人体 的单位向量
        %     - 二者相加，得到“人体速度方向在 Tx/Rx 路径上的投影方向”
        A(ii,:) = (torso_pos - Tx_pos)/dis_torso_tx + ...
            (torso_pos - Rx_pos(ii,:))/dis_torso_rx;
    end
end