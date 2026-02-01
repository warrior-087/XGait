function res = successiveTrack_core(init_loc, tx_loc, rx_loc, plcr, direct, duration)
% res = SUCCESSIVE_TRACK(INIT_LOC,TX_LOC,RX_LOC,PLCR,DIRECT,DURATION)
% find the trace with PLCR and DIRECTION information.
% 
% INIT_LOC : Initial location of the target
% TX_LOC   : Location of the transmitter
% RX_LOC   : Location of the receivers
% PLCR     : Amplitudes of path length change rate
% DIRECT   : Opportunistic signs of path length change rate
% DURATION : time duration between samples.
%
% LS       : Location series
% VS       : Velocity series
% ES       : Error series
%

    ls = [init_loc, zeros(2, size(plcr,2))];
    vs = zeros(2, size(plcr,2));
    es = zeros(1, size(plcr,2));
    tls = ls;
    tvs = vs;
    tes = es;
    mls = tls;
    mvs = tvs;
    mes = tes;
    
    search_point_record = 1; % Start point records of possible segments.
    init_error_record = 1; % Initial error records of possible segments.
    init_error_list = [0.01,0.01;-0.01,-0.01].'; % Initial error candidates.
    first_trace = true; % Flag of first trace candidate.
    
    % Search trace segments with different initial errors.
    while ~isempty(search_point_record)
        search_point = search_point_record(end);
        jj = init_error_record(end);
        if jj > size(init_error_list,2)
            search_point_record = search_point_record(1:end-1);
            init_error_record = init_error_record(1:end-1);
            continue;
        end
        init_error_record(end) = jj + 1;
        
        % Antenna selection in two turns. In first turn, all antennas are
        % assigned with equal weights. In second turn, antennas are
        % weighted according to their contribution to tracking errors.
        antenna_weight = ones(size(plcr,1),1);
        for sel_turn = 1:2
            antenna_mse_err_sum = zeros(size(plcr,1),1);
            link_sel = true(size(plcr,1),1);
          
            % Successive update target locations.
            for ii = search_point:length(es)
                curr_loc = mls(:,ii);
                dist_mtx = distance_matrix_fusion(curr_loc, tx_loc, rx_loc);
                dist_mtx_sel = dist_mtx(link_sel, :);
                direct_sel = direct(link_sel, ii);
                unknown_sel = direct_sel == 0;
                mse_mtx = (dist_mtx_sel.' * diag(antenna_weight) * dist_mtx_sel)\(dist_mtx_sel.'*diag(antenna_weight));
                
                % Test all unknown directions.
                link_num = sum(unknown_sel);
                for kk = 0:((2^link_num)-1)
                    direct_cand_bin = dec2bin(kk, link_num);
                    unknown_cand = zeros(link_num, 1);
                    for ll = 1:link_num
                        unknown_cand(ll) = 2 * str2double(direct_cand_bin(ll)) - 1;
                    end
                    direct_cand = direct_sel;
                    direct_cand(unknown_sel) = unknown_cand;
                    plcr_signed = plcr(link_sel,ii) .* direct_cand;
                    tv = mse_mtx * plcr_signed;
                    
                    % The velocity deviation error.
                    if ii == search_point
                        lv_err = 0.1*norm(tv - init_error_list(:,jj));
                    else
                        lv_err = 0.1*norm(antenna_weight .* (dist_mtx_sel * tv) - antenna_weight .* (ld(link_sel,:) * mvs(:,ii-1)), 2);
                    end
                    
                    % The weighted least square mean error.
                    antenna_mse_err = antenna_weight .* (dist_mtx_sel * tv - plcr_signed).^2;
                    mse_err = norm(antenna_mse_err, 2);
                    all_err = lv_err + mse_err;
                    if kk == 0
                        all_err_m = all_err;
                        mse_err_m = mse_err;
                        antenna_mse_err_m = antenna_mse_err;
                        mvs(:,ii) = tv;
                    else
                        if all_err < all_err_m
                            all_err_m = all_err;
                            mse_err_m = mse_err;
                            antenna_mse_err_m = antenna_mse_err;
                            mvs(:,ii) = tv;                            
                        end
                    end
                end
                
                % Update temporal location.
                mls(:,ii+1) = curr_loc + mvs(:,ii) * duration(:,ii);
                if ii == search_point
                    mes(:,ii) = mse_err_m;
                else
                    mes(:,ii) = all_err_m;
                end
                antenna_mse_err_sum = antenna_mse_err_sum + antenna_mse_err_m;
                
                % Identify vulnerable point.
                if (sum(duration(search_point:ii-1)) > 1 && ...
                        (norm(mvs(:,ii),2) < 0.3 && norm(mvs(:,ii),2) > 2))
                    break;
                end
                
                % Record last distance matrix.
                ld = dist_mtx;
            end

            % Update temporal solution for segment.
            if sel_turn == 1
                segment_err_m = sum(mes(search_point:ii)) / (ii-search_point+1);
                tls(:,search_point:ii+1) = mls(:,search_point:ii+1);
                tvs(:,search_point:ii) = mvs(:,search_point:ii);
                tes(:,search_point:ii) = mes(:,search_point:ii);
                last_search_end = ii;
            else
                temp_err_m  = sum(mes(search_point:ii)) / (ii-search_point+1);
                if temp_err_m <= segment_err_m
                    tls(:,search_point:ii+1) = mls(:,search_point:ii+1);
                    tvs(:,search_point:ii) = mvs(:,search_point:ii);
                    tes(:,search_point:ii) = mes(:,search_point:ii);
                    last_search_end = ii;
                    segment_err_m = temp_err_m;
                end
            end
            
            % Update antenna weight.
            if size(plcr,1) > 3
                antenna_mse_err_mean = antenna_mse_err_sum / (ii - search_point + 1);
                antenna_mes_err_mean_sorted = sort(antenna_mse_err_mean, 'ascend');
                antenna_mes_err_mm = mean(antenna_mse_err_mean);
                antenna_weight((antenna_mse_err_mean > 1.2 * antenna_mes_err_mm) & (antenna_mse_err_mean > antenna_mes_err_mean_sorted(4))) = 0;
                antenna_weight((antenna_mse_err_mean <= 1.2 * antenna_mes_err_mm) & (antenna_mse_err_mean > antenna_mes_err_mean_sorted(4))) = exp(-20*(antenna_mse_err_mean((antenna_mse_err_mean <= 1.2 * antenna_mes_err_mm) & (antenna_mse_err_mean > antenna_mes_err_mean_sorted(4)))-antenna_mes_err_mean_sorted(1)));
                antenna_weight = antenna_weight / sum(antenna_weight) * size(plcr,1);
            end
        end
        
        %  Update global trace.
        if last_search_end == length(tes)
            if first_trace
                global_min_err = sum(tes);
                ls = tls;
                vs = tvs;
                es = tes;
                first_trace = false;
            else
                temp_err_m = sum(tes);
                if temp_err_m < global_min_err
                    ls = tls;
                    vs = tvs;
                    es = tes;
                    global_min_err = temp_err_m;
                end
            end
        else
            % Seach next segment.
            search_point_record = [search_point_record,last_search_end+1];
            init_error_record = [init_error_record,1];
        end
    end

    res.ls = ls;
    res.vs = vs;
    res.es = es;
end