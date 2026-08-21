function [U_map, S_map, V_map, time_online, time_offline] = ApSnap_SVD(Afull, m_rows, n_cols, opts)
% APSNAP_SVD Solves a parametric Singular Value Decomposition A(p) ~ U*S*V'
% using an Model Order Reduction (MOR) framework.
%
% This algorithm computes the singular values and corresponding singular vectors
% of a sequence of matrices provided via a function handle Afull(p). It employs 
% a dual-projection onto reduced left and right subspaces (U and V). Accuracy 
% is maintained via an adaptive Sliding Window strategy that enriches the bases 
% at a specified frequency.
%
% =========================================================================
% INPUTS:
% =========================================================================
% Afull  : Function handle. Afull(p) returns the m x n matrix for parameter p.
% m_rows : Integer. Number of rows of the full-space matrix.
% n_cols : Integer. Number of columns of the full-space matrix.
% opts   : Struct containing algorithm configurations:
%
%   [Mandatory]
%   opts.param      : Array of parameters (p) for the online evaluation.
%   opts.nsv        : Number of target singular values/vectors.
%   opts.k          : Number of initial snapshot points for training.
%
%   [Optional]
%   opts.s          : Oversampling parameter (default: round(nsv * 0.5)).
%   opts.SnapPoints : Custom parameter values for the offline phase.
%   opts.update_freq: Frequency of sliding window updates (default: Inf).
%   opts.trunc_tol  : Tolerance for rank truncation (default: 1e-8).
%   opts.tol_upd    : Tolerance for the exact solver (default: 1e-8).
%
% =========================================================================

%% --- 0. SETUP AND OPTION VALIDATION ---
if ~isfield(opts, 'param'), error('opts.param is mandatory.'); end
if ~isfield(opts, 'nsv'), error('opts.nsv is mandatory.'); end
if ~isfield(opts, 'k'), error('opts.k is mandatory.'); end

% Set default values for optional parameters
if ~isfield(opts, 's'), opts.s = round(opts.nsv * 0.5); end 
if ~isfield(opts, 'update_freq'), opts.update_freq = Inf; end
if ~isfield(opts, 'trunc_tol'), opts.trunc_tol = 1e-8; end
if ~isfield(opts, 'tol_upd'), opts.tol_upd = 1e-8; end

% Generate default snapshot points if not provided
if ~isfield(opts, 'SnapPoints') || isempty(opts.SnapPoints)
    points = floor(linspace(1, length(opts.param), opts.k));
    opts.SnapPoints = opts.param(points);
end

tic
%% --- 1. OFFLINE PHASE: Snapshot points ---
% Build the initial bases Q_U and Q_V by collecting snapshots.
% -------------------------------------------------------------------------
X_U = zeros(m_rows, (opts.s + opts.nsv) * opts.k); 
X_V = zeros(n_cols, (opts.s + opts.nsv) * opts.k); 

U_map = containers.Map('KeyType','double','ValueType','any');
S_map = containers.Map('KeyType','double','ValueType','any');
V_map = containers.Map('KeyType','double','ValueType','any');

for i = 1:opts.k
    p = opts.SnapPoints(i);
    
    % Compute full-order solution for snapshots
    [U_full, S_full, V_full] = svds(Afull(p), opts.nsv + opts.s, 'largest', 'MaxIterations', 2000);
    
    % Sort results and store in the history matrices
    [~, idx] = sort(diag(S_full), 'descend');
    
    X_U(:, (opts.s + opts.nsv)*(i-1)+1 : (opts.s + opts.nsv)*i) = U_full(:, idx);
    X_V(:, (opts.s + opts.nsv)*(i-1)+1 : (opts.s + opts.nsv)*i) = V_full(:, idx);
    
    % Cache top target solutions
    U_map(p) = U_full(:, idx(1:opts.nsv));
    V_map(p) = V_full(:, idx(1:opts.nsv));
    S_map(p)  = diag(S_full(idx(1:opts.nsv), idx(1:opts.nsv)));
end

% Construct the initial orthonormal bases for U and V
Q_U = orth(X_U);
Q_V = orth(X_V);

time_offline = toc;
tic
%% --- 2. ONLINE PHASE ---
% -------------------------------------------------------------------------
for i = 1:length(opts.param)
    
    p_now = opts.param(i);
    
    % Skip evaluation if the result exists in the cache (from offline phase)
    if isKey(U_map, p_now), continue; end
    
    % --- 2A. Full Matrix Projection (Right Side) ---
    mat = Afull(p_now);
    A_QV = mat * Q_V; 
    
    % --- 2B. Reduced Rayleigh-Ritz Step (Left Side & SVD) ---
    Mnow = Q_U' * A_QV;
    [u_m, s_m, v_m] = svd(Mnow, 'econ'); 
    
    % --- 2C. Singular Vector Projection & Storage ---
    % Determine if an update step is required to enrich the basis
    is_update_step = (mod(i, opts.update_freq) == 0);
    
    if is_update_step
        num_to_extract = min([size(u_m, 2), size(v_m, 2), opts.nsv + opts.s]);
        new_U_full = Q_U * u_m(:, 1:num_to_extract);
        new_V_full = Q_V * v_m(:, 1:num_to_extract);
        
        safe_nsv = min(opts.nsv, num_to_extract);
    else
        safe_nsv = min([size(u_m, 2), size(v_m, 2), opts.nsv]);
        U_map(p_now) = Q_U * u_m(:, 1:safe_nsv);
        V_map(p_now) = Q_V * v_m(:, 1:safe_nsv);
        S_map(p_now) = diag(s_m(1:safe_nsv, 1:safe_nsv));
    end
    
    % --- 2D. Subspace update ---
    if is_update_step
        
        % Update Left Space (U)
        vectors_to_add_U = [];
        Q_temp_U = Q_U; 

        if opts.nsv + opts.s == 1
            [new_U_full, D, new_V_full] = svds(mat, opts.nsv+ opts.s, 'largest','RightStartVector', new_V_full,'Tolerance',opts.tol_upd);
        else
            [new_U_full, D, new_V_full] = my_svds(mat, opts.nsv+ opts.s,'RightStartVector', new_V_full,'Tolerance',opts.tol_upd);
        end
        [~, idx] = sort(abs(diag(D)), 'descend');
        new_U_full = new_U_full(:, idx);
        new_V_full = new_V_full(:, idx);
        S_map(p_now) = diag(D(idx(1:opts.nsv), idx(1:opts.nsv)));
        U_map(p_now) = new_U_full(:, 1:opts.nsv);
        V_map(p_now) = new_V_full(:, 1:opts.nsv);

        for j = 1:size(new_U_full, 2)
            u = new_U_full(:, j);
            % Orthogonalize and truncate based on the relative norm
            u_orth = u - Q_temp_U * (Q_temp_U' * u); 
            if norm(u_orth) / (norm(u) + eps) > opts.trunc_tol
                vectors_to_add_U = [vectors_to_add_U, u];
                Q_temp_U = [Q_temp_U, u_orth / (norm(u_orth) + eps)]; 
            end
        end
        
        % Update Right Space (V)
        vectors_to_add_V = [];
        Q_temp_V = Q_V; 
        for j = 1:size(new_V_full, 2)
            v = new_V_full(:, j);
            % Orthogonalize and truncate based on the relative norm
            v_orth = v - Q_temp_V * (Q_temp_V' * v); 
            if norm(v_orth) / (norm(v) + eps) > opts.trunc_tol
                vectors_to_add_V = [vectors_to_add_V, v];
                Q_temp_V = [Q_temp_V, v_orth / (norm(v_orth) + eps)]; 
            end
        end
        
        % Subspace update and re-orthogonalization for U
        if ~isempty(vectors_to_add_U)
            num_added_U = size(vectors_to_add_U, 2);
            if size(X_U, 2) > 500
            X_U(:, 1:num_added_U) = [];
            end
            X_U = [X_U, vectors_to_add_U];
            Q_U = orth(X_U);
        end
        
        % Subspace update and re-orthogonalization for V
        if ~isempty(vectors_to_add_V)
            num_added_V = size(vectors_to_add_V, 2);
            if size(X_V,2)>500
            X_V(:, 1:num_added_V) = [];
            end
            X_V = [X_V, vectors_to_add_V];
            Q_V = orth(X_V);
        end
        
    end
end
time_online = toc;
end