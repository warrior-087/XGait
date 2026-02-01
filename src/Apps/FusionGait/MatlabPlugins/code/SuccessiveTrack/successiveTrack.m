%% -------------------------
function res = successiveTrack(init_loc, tx_loc, rx_loc, plcr, direct, duration)
% Core function: supports per-link independent TX and RX locations.
% The original successiveTrack logic is preserved; the only required change is that
% the distance_matrix computation must handle a distinct TX for each link.

ls  = [init_loc, zeros(2, size(plcr,2))];
vs  = zeros(2, size(plcr,2));
es  = zeros(1, size(plcr,2));
tls = ls;
tvs = vs;
tes = es;
mls = tls;
mvs = tvs;
mes = tes;

search_point_record = 1;
init_error_record   = 1;
init_error_list     = [0.01,0.01; -0.01,-0.01].';
first_trace         = true;

while ~isempty(search_point_record)
    search_point = search_point_record(end);
    jj = init_error_record(end);

    if jj > size(init_error_list,2)
        search_point_record = search_point_record(1:end-1);
        init_error_record = init_error_record(1:end-1);
        continue;
    end
    init_error_record(end) = jj + 1;

    antenna_weight = ones(size(plcr,1),1);

    for sel_turn = 1:2
        antenna_mse_err_sum = zeros(size(plcr,1),1);
        link_sel = true(size(plcr,1),1);

        for ii = search_point:length(es)
            curr_loc = mls(:,ii);

            % --- Key change: use per-link TX locations (each link has its own TX) ---
            dist_mtx = distance_matrix(curr_loc, tx_loc, rx_loc);

            dist_mtx_sel = dist_mtx(link_sel, :);
            direct_sel = direct(link_sel, ii);
            unknown_sel = direct_sel == 0;

            mse_mtx = (dist_mtx_sel.' * diag(antenna_weight) * dist_mtx_sel) \ ...
                      (dist_mtx_sel.' * diag(antenna_weight));

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

                if ii == search_point
                    % Initialization regularization: prefer velocities close to a small bias vector
                    lv_err = 0.1 * norm(tv - init_error_list(:,jj));
                else
                    % Temporal smoothness term: penalize changes in the predicted link measurements
                    lv_err = 0.1 * norm(antenna_weight .* (dist_mtx_sel * tv) - ...
                                        antenna_weight .* (ld(link_sel,:) * mvs(:,ii-1)), 2);
                end

                % Measurement fitting error per link (weighted squared residual)
                antenna_mse_err = antenna_weight .* (dist_mtx_sel * tv - plcr_signed).^2;
                mse_err = norm(antenna_mse_err, 2);

                % Total objective
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

            % Propagate location with the estimated velocity and per-step duration
            mls(:,ii+1) = curr_loc + mvs(:,ii) * duration(:,ii);

            % For the first step, store pure measurement error; otherwise store total error
            if ii == search_point
                mes(:,ii) = mse_err_m;
            else
                mes(:,ii) = all_err_m;
            end

            antenna_mse_err_sum = antenna_mse_err_sum + antenna_mse_err_m;

            % Early termination condition (kept as-is from original logic)
            if (sum(duration(search_point:ii-1)) > 1 && ...
                    (norm(mvs(:,ii),2) < 0.3 && norm(mvs(:,ii),2) > 2))
                break;
            end

            ld = dist_mtx;
        end

        if sel_turn == 1
            segment_err_m = sum(mes(search_point:ii)) / (ii-search_point+1);
            tls(:,search_point:ii+1) = mls(:,search_point:ii+1);
            tvs(:,search_point:ii)   = mvs(:,search_point:ii);
            tes(:,search_point:ii)   = mes(:,search_point:ii);
            last_search_end = ii;
        else
            temp_err_m = sum(mes(search_point:ii)) / (ii-search_point+1);
            if temp_err_m <= segment_err_m
                tls(:,search_point:ii+1) = mls(:,search_point:ii+1);
                tvs(:,search_point:ii)   = mvs(:,search_point:ii);
                tes(:,search_point:ii)   = mes(:,search_point:ii);
                last_search_end = ii;
                segment_err_m = temp_err_m;
            end
        end

        if size(plcr,1) > 3
            % Robust re-weighting: downweight links with consistently large residuals
            antenna_mse_err_mean = antenna_mse_err_sum / (ii - search_point + 1);
            antenna_mes_err_mean_sorted = sort(antenna_mse_err_mean, 'ascend');
            antenna_mes_err_mm = mean(antenna_mse_err_mean);

            antenna_weight((antenna_mse_err_mean > 1.2 * antenna_mes_err_mm) & ...
                           (antenna_mse_err_mean > antenna_mse_err_mean_sorted(4))) = 0;

            antenna_weight((antenna_mse_err_mean <= 1.2 * antenna_mes_err_mm) & ...
                           (antenna_mse_err_mean > antenna_mse_err_mean_sorted(4))) = ...
                exp(-20 * (antenna_mse_err_mean((antenna_mse_err_mean <= 1.2 * antenna_mes_err_mm) & ...
                                                (antenna_mse_err_mean > antenna_mse_err_mean_sorted(4))) - ...
                           antenna_mes_err_mean_sorted(1)));

            antenna_weight = antenna_weight / sum(antenna_weight) * size(plcr,1);
        end
    end

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
        search_point_record = [search_point_record, last_search_end + 1];
        init_error_record   = [init_error_record, 1];
    end
end

res.ls = ls;
res.vs = vs;
res.es = es;
end
