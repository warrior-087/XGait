function res = fusionSuccessiveTrack(init_loc, input_struct)
% fusionSuccessiveTrack
%   Support dynamic multi-modality input and automatically merge all links.
%
% Inputs:
%   init_loc     : 2x1 initial location
%   input_struct : struct whose field names are modality names (e.g., .wifi, .acous)
%                 Each sub-struct contains:
%                   tx_loc   - 2x1 transmitter location
%                   rx_loc   - 2xN receiver locations (one column per link)
%                   plcr     - NxT measured link changes (per-link per-time)
%                   direc    - NxT direction sign indicators (NOTE: named 'direc' from Python)
%                   duration - 2xT (or 1xT) time-step duration; assumed aligned across modalities

% 1) Initialize merged containers
tx_all = [];
rx_all = [];
plcr_all = [];
direct_all = [];
global_duration = [];

% 2) Enumerate all modalities and merge their links
mod_names = fieldnames(input_struct);

for i = 1:numel(mod_names)
    mod_name = mod_names{i};
    data = input_struct.(mod_name);

    % Number of links in this modality
    N = size(data.rx_loc, 2);

    % --- Expand and merge link geometry ---
    % tx_current: 2 x N (replicate tx_loc for each link)
    tx_current = repmat(data.tx_loc, 1, N);
    tx_all = [tx_all, tx_current];

    % rx_all: 2 x N (concatenate receivers)
    rx_all = [rx_all, data.rx_loc];

    % plcr_all: (sumN) x T (stack links vertically)
    plcr_all = [plcr_all; data.plcr];

    % direct_all: (sumN) x T
    % NOTE: Python passes the field name as 'direc', which corresponds to 'direct' here.
    direct_all = [direct_all; data.direc];

    % --- Extract duration ---
    % Since all modalities have already been time-aligned in Python,
    % we can take the duration from the first modality that provides it.
    if isempty(global_duration) && isfield(data, 'duration')
        global_duration = data.duration;
    end
end

% 3) Call the core tracking function
% NOTE: duration is assumed to be aligned and have the correct length T.
res = successiveTrack_core(init_loc, tx_all, rx_all, plcr_all, direct_all, global_duration);

end
