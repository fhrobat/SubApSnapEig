function [U_map, S_map, V_map, time_online, time_offline] = SubApSnap_SVD(Afull, Arows, m_rows, n_cols, opts)
% SUBAPSNAP_SVD Solves a parametric Singular Value Decomposition (SVD) 
% using Model Order Reduction (MOR) and row subsampling.
%
% This algorithm computes the singular values and vectors for a sequence 
% of parameter-dependent matrices A(p). It constructs orthogonal bases for 
% both the left and right singular subspaces from past solutions, projecting 
% the problem into a lower-dimensional space via a sketched approach. An update  
strategy for the subspace is implemented.
%
% INPUTS:
%   Afull  : Function handle returning the full m x n matrix for parameter p.
%   Arows  : Function handle returning A(p)(idx, :) * X. 
%   m_rows : Integer specifying the number of matrix rows.
%   n_cols : Integer specifying the number of matrix columns.
%   opts   : Configuration structure containing the following fields:
%       - param        : Array of parameters for the online phase (Required).
%       - nsv          : Number of target singular values/vectors (Required).
%       - k            : Number of snapshot points for offline training (Required).
%       - sampling     : Function handle or string for the sampling method (Required).
%       - s            : Oversampling parameter (default: round(nsv * 0.5)).
%       - update_freq  : Frequency of sliding window updates (default: 10).
%       - tol          : Sketched residual tolerance (default: 1e-4).
%       - tol_upd      : Exact solver tolerance (default: 1e-8).
%       - sketch_target: Subspace to sample from ('U' [default], 'AV', or 'U_AV').
%       - trunc_tol    : Tolerance for basis rank truncation (default: 1e-8). 
%
% OUTPUTS:
%   U_map        : containers.Map linking parameter p to left singular vectors.
%   S_map        : containers.Map linking parameter p to singular values.
%   V_map        : containers.Map linking parameter p to right singular vectors.
%   time_online  : Total execution time for the online phase (seconds).
%   time_offline : Total execution time for the offline phase (seconds).

%% --- 0. SETUP AND VALIDATION ---

% Validate mandatory parameters
if ~isfield(opts, 'param'), error('opts.param is mandatory.'); end
if ~isfield(opts, 'nsv'), error('opts.nsv is mandatory.'); end
if ~isfield(opts, 'k'), error('opts.k is mandatory.'); end
if ~isfield(opts, 'sampling'), error('opts.sampling is mandatory.'); end

% Set default values for optional parameters
if ~isfield(opts, 's'), opts.s = round(opts.nsv * 0.5); end 
if ~isfield(opts, 'update_freq'), opts.update_freq = 10; end
if ~isfield(opts, 'tol'), opts.tol = 1e-4; end
if ~isfield(opts, 'tol_upd'), opts.tol_upd = 1e-8; end
if ~isfield(opts, 'sketch_target'), opts.sketch_target = 'U'; end
if ~isfield(opts, 'trunc_tol'), opts.trunc_tol = 1e-8; end

% Initialize snapshot points if not provided
if ~isfield(opts, 'SnapPoints') || isempty(opts.SnapPoints)
    points = floor(linspace(1, length(opts.param), opts.k));
    opts.SnapPoints = opts.param(points);
end

tic

%% --- 1. OFFLINE PHASE ---

X_U = zeros(m_rows, (opts.s + opts.nsv) * opts.k); 
X_V = zeros(n_cols, (opts.s + opts.nsv) * opts.k); 

U_map = containers.Map('KeyType','double','ValueType','any');
S_map = containers.Map('KeyType','double','ValueType','any');
V_map = containers.Map('KeyType','double','ValueType','any');

cached_params = [];
cached_norms = [];

% Compute initial SVD snapshots to build the global basis
for i = 1:opts.k
    p = opts.SnapPoints(i);
    mat = Afull(p);
    
    [U_max, S_max, V_max] = svds(mat, opts.nsv + opts.s, 'largest', 'MaxIterations', 2000);
    
    [~, idx] = sort(diag(S_max), 'descend');
    X_U(:, (opts.s + opts.nsv)*(i-1)+1 : (opts.s + opts.nsv)*i) = U_max(:, idx);
    X_V(:, (opts.s + opts.nsv)*(i-1)+1 : (opts.s + opts.nsv)*i) = V_max(:, idx);
    
    U_map(p) = U_max(:, idx(1:opts.nsv));
    V_map(p) = V_max(:, idx(1:opts.nsv));
    S_map(p) = diag(S_max(idx(1:opts.nsv), idx(1:opts.nsv)));
    
    cached_params(end+1) = p;
    cached_norms(end+1) = norm(mat, 1);
end

% Orthonormalize the bases
Q_U = orth(X_U);
Q_V = orth(X_V);

time_offline = toc;

tic

%% --- 2. ONLINE PHASE ---

Pveclev = []; 

for i = 1:length(opts.param)
    p_now = opts.param(i);
    if isKey(U_map, p_now), continue; end
    
    mat_now = []; 
    
    % -- 2A. Row Subsampling --
    if isempty(Pveclev)
        switch opts.sketch_target
            case 'U'
                 [Pveclev,probs] = samp(Q_U, max(size(Q_U,2), size(Q_V,2))*4, opts.sampling);
            case 'AV'
                TargetMat = Afull(p_now) * Q_V;
                [V_target, ~] = qr(TargetMat, 0);
                Pveclev = samp(V_target, size(V_target,2)*4, opts.sampling);
            case 'U_AV'
                TargetMat = [Q_U, Afull(p_now) * Q_V];
                [V_target, ~] = qr(TargetMat, 0);
                Pveclev = samp(V_target, size(V_target,2)*4, opts.sampling);
        end
    end
    
    % -- 2B. Sketched SVD Problem --
    if strcmp(opts.sampling, 'leverage')
        w = sqrt(length(Pveclev) * probs(Pveclev));
        w = w(:);
        S_U = Q_U(Pveclev, :)./w;
        S_AV = Arows(p_now, Pveclev, Q_V)./w; 
    else
        S_U = Q_U(Pveclev, :);
        S_AV = Arows(p_now, Pveclev, Q_V); 
    end
    
    % Solve the reduced problem
    Mnow = lsqminnorm(S_U, S_AV);
    [u_m, s_m, v_m] = svd(Mnow, 'econ');
    
    % -- 2C. Update Trigger Logic --
    do_update = (mod(i, opts.update_freq) == 0);
    
    if ~do_update
        u_n = u_m(:, 1:opts.nsv);
        v_n = v_m(:, 1:opts.nsv);
        S_n = s_m(1:opts.nsv, 1:opts.nsv);
        
        % Evaluate sketched residual against tolerance
        res_nums = norm(S_AV * v_n - S_U * u_n * S_n, 'fro');
        [~, closest_idx] = min(abs(cached_params - p_now));
        res_dens = cached_norms(closest_idx);
        
        if (res_nums / res_dens) > opts.tol
            mat_now = Afull(p_now);
            U_approx = Q_U * u_n; 
            V_approx = Q_V * v_n;
            sigma_m  = diag(S_n);
            
            true_res = norm(mat_now * V_approx - U_approx * diag(sigma_m), 'fro') / ...
                      ((norm(mat_now, 1) + max(abs(sigma_m))) * norm(V_approx, 'fro') + eps);

            cached_params(end+1) = p_now;
            cached_norms(end+1) = norm(mat_now, 1);
            
            if true_res > opts.tol
                do_update = true;
                warning('Step %d (p=%g): True residual (%e) exceeds tolerance.', i, p_now, true_res);
            end
        end
    end
    
    % -- 2D. Subspace update --
    if do_update
        if isempty(mat_now)
            mat_now = Afull(p_now); 
            cached_params(end+1) = p_now;
            cached_norms(end+1) = norm(mat_now, 1);
        end
        
        num_to_seed = min(size(v_m, 2), opts.nsv + opts.s);
        v_start = Q_V * v_m(:, 1:num_to_seed);
        
        % Compute exact SVD to update the bases
        if opts.nsv + opts.s == 1
            [U_true, S_true, V_true] = svds(mat_now, opts.nsv + opts.s, 'largest', ...
                                        'Tolerance', opts.tol_upd, ...
                                        'RightStartVector', v_start);
        else
            [U_true, S_true, V_true] = my_svds(mat_now, opts.nsv + opts.s, ...
                                        'Tolerance', opts.tol_upd, ...
                                        'RightStartVector', v_start);
        end

        [~, idx] = sort(diag(S_true), 'descend');
        new_U_full = U_true(:, idx);
        new_V_full = V_true(:, idx);
        
        S_map(p_now) = diag(S_true(idx(1:opts.nsv), idx(1:opts.nsv)));
        U_map(p_now) = new_U_full(:, 1:opts.nsv);
        V_map(p_now) = new_V_full(:, 1:opts.nsv);
        
        % Memory Update: Slide windows with rank truncation
        vectors_to_add_U = [];
        Q_temp_U = Q_U; 
        for j = 1:size(new_U_full, 2)
            u = new_U_full(:, j);
            u_orth = u - Q_temp_U * (Q_temp_U' * u); 
            if opts.trunc_tol == 0 || norm(u_orth) / (norm(u) + eps) > opts.trunc_tol
                vectors_to_add_U = [vectors_to_add_U, u];
                Q_temp_U = [Q_temp_U, u_orth / (norm(u_orth) + eps)]; 
            end
        end
        
        vectors_to_add_V = [];
        Q_temp_V = Q_V; 
        for j = 1:size(new_V_full, 2)
            v = new_V_full(:, j);
            v_orth = v - Q_temp_V * (Q_temp_V' * v); 
            if opts.trunc_tol == 0 || norm(v_orth) / (norm(v) + eps) > opts.trunc_tol
                vectors_to_add_V = [vectors_to_add_V, v];
                Q_temp_V = [Q_temp_V, v_orth / (norm(v_orth) + eps)]; 
            end
        end
        
        if ~isempty(vectors_to_add_U)
            if size(X_U, 2) > 500
                X_U(:, 1:size(vectors_to_add_U, 2)) = [];
            end
            X_U = [X_U, vectors_to_add_U];
            Q_U = orth(X_U);
        end
        
        if ~isempty(vectors_to_add_V)
            if size(X_V,2) > 500
                X_V(:, 1:size(vectors_to_add_V, 2)) = [];
            end
            X_V = [X_V, vectors_to_add_V];
            Q_V = orth(X_V);
        end
        
        Pveclev = []; % Force resampling
        
    else
        % Lazy Projection: Use ROM approximation without full matrix call
        num_to_extract = min(size(u_m, 2), opts.nsv);
        U_map(p_now) = Q_U * u_m(:, 1:num_to_extract);
        V_map(p_now) = Q_V * v_m(:, 1:num_to_extract);
        S_map(p_now) = diag(s_m(1:opts.nsv, 1:opts.nsv));
    end
end
time_online = toc;
end