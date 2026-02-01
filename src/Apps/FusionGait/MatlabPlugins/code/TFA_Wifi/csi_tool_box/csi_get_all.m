function [cfr_array, timestamp] = csi_get_all(filename)
csi_trace = read_bf_file(filename);
timestamp = zeros(length(csi_trace), 1);
cfr_array = zeros(length(csi_trace), 90);

tx_select = 1; % select tx
for k = 1:length(csi_trace)
    csi_entry = csi_trace{k}; % for the k_{th} packet
    csi_raw = get_scaled_csi(csi_entry);

%     try
        if numel(size(csi_raw)) == 3
            csi_all = squeeze(csi_raw(tx_select, :, :)).';
        else
            csi_all = squeeze(csi_raw).'; % estimate channel matrix Hexp-figu
        end
%     catch err
%         disp(err)
%         continue
%     end
    
    csi = [csi_all(:,1); csi_all(:,2); csi_all(:, 3)].'; % select CSI for one antenna pair
    
    timestamp(k) = csi_entry.timestamp_low;
    cfr_array(k,:) = csi;
end