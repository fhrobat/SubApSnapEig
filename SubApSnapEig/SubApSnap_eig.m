function [eigenvectors, eigenvalues, time_online, time_offline] = SubApSnap_eig(Afull, Arows, n, opts)
% SUBAPSNAP_EIG Solves a parametric eigenvalue problem via Model Order Reduction and Row Subsampling.
%
% This algorithm computes the eigenvectors and eigenvalues for a sequence 
% of parameter-dependent matrices A(p). It builds a basis from past solutions 
% and projects the problem into a lower-dimensional space using a Sketched 
% Rayleigh-Ritz approach. Au updating strategy for the subspace is implemented.
%
% INPUTS:
%   Afull : Function handle returning the full n x n matrix for a given parameter p, or Amult(p, X) for matrix-free.
%   Arows : Function handle returning A(p)(idx, :) * X.
%   n     : Matrix dimension (n x n).
%   opts  : Configuration structure with the following fields:
%       - param      : Array of parameters for the online phase (Required).
%       - neig       : Number of target eigenvalues/eigenvectors (Required).
%       - k          : Number of snapshot points for offline training (Required).
%       - sampling   : Function handle or string for the sampling strategy (Required).
%       - eigs_sigma : Target eigenvalues ('largestabs' [default] or 'smallestabs').
%       - s          : Oversampling parameter (default: round(neig * 0.5)).
%       - update_freq: Frequency for triggering basis updates (default: 10).
%       - tol        : Sketched residual tolerance (default: 1e-4).
%       - tol_upd    : Exact solver tolerance (default: 1e-8).
%       - sketch_target: Subspace to sample from ('X', 'AX', or 'X_AX').
%       - trunc_tol  : Tolerance for basis rank truncation (default: 1e-8).
%
% OUTPUTS:
%   eigenvectors : containers.Map linking parameter p to its eigenvector matrix.
%   eigenvalues  : containers.Map linking parameter p to its eigenvalue vector.
%   time_online  : Total execution time for the online phase (seconds).
%   time_offline : Total execution time for the offline phase (seconds).

%% --- 0. SETUP AND VALIDATION ---

% Determine if the input uses a matrix-free approach
if nargin(Afull) == 2
    is_matrix_free = true;
    Amult = Afull;
else
    is_matrix_free = false;
end

% Validate mandatory parameters
if ~isfield(opts, 'param'), error('opts.param is mandatory.'); end
if ~isfield(opts, 'neig'), error('opts.neig is mandatory.'); end
if ~isfield(opts, 'k'), error('opts.k is mandatory.'); end
if ~isfield(opts, 'sampling'), error('opts.sampling is mandatory.'); end

% Set default values for optional parameters
if ~isfield(opts, 's'), opts.s = round(opts.neig * 0.5); end 
if ~isfield(opts, 'update_freq'), opts.update_freq = 10; end
if ~isfield(opts, 'tol'), opts.tol = 1e-4; end
if ~isfield(opts, 'tol_upd'), opts.tol_upd = 1e-8; end
if ~isfield(opts, 'sketch_target'), opts.sketch_target = 'X'; end
if ~isfield(opts, 'trunc_tol'), opts.trunc_tol = 1e-8; end
if ~isfield(opts, 'eigs_sigma'), opts.eigs_sigma = 'largestabs'; end
if ~isfield(opts, 'IsFunctionSymmetric'), opts.IsFunctionSymmetric = 0; end

% Initialize snapshot points if not provided
if ~isfield(opts, 'SnapPoints') || isempty(opts.SnapPoints)
    points = floor(linspace(1, length(opts.param), opts.k));
    opts.SnapPoints = opts.param(points);
end

% Define sorting direction based on target eigenvalues
is_smallest = isequal(opts.eigs_sigma, 'smallestabs') || isequal(opts.eigs_sigma, 'sm') || (isnumeric(opts.eigs_sigma) && opts.eigs_sigma == 0);
if is_smallest, sort_dir = 'ascend'; else, sort_dir = 'descend'; end

tic

%% --- 1. OFFLINE PHASE ---

X = zeros(n, (opts.s + opts.neig) * opts.k); 
eigenvectors = containers.Map('KeyType','double','ValueType','any');
eigenvalues = containers.Map('KeyType','double','ValueType','any');

cached_params = [];
cached_norms = [];

% Compute initial snapshots
for i = 1:opts.k
    p = opts.SnapPoints(i);
    
    if is_matrix_free
        [X_max, D_max] = eigs(@(X) Amult(p, X), n, opts.neig + opts.s, opts.eigs_sigma, 'MaxIterations', 2000, 'IsFunctionSymmetric', opts.IsFunctionSymmetric);
    else
        [X_max, D_max] = eigs(Afull(p), opts.neig + opts.s, opts.eigs_sigma, 'MaxIterations', 2000);
    end
    
    [~, idx] = sort(abs(diag(D_max)), sort_dir);
    X(:, (opts.s + opts.neig)*(i-1)+1 : (opts.s + opts.neig)*i) = X_max(:, idx);
    
    eigenvectors(p) = X_max(:, idx(1:opts.neig));
    eigenvalues(p)  = diag(D_max(idx(1:opts.neig), idx(1:opts.neig)));
    
    if ~is_matrix_free
        cached_params(end+1) = p;
        cached_norms(end+1) = norm(Afull(p), 1);
    end
end

% Orthonormalize the basis
Q = orth(X);
time_offline = toc;

tic

%% --- 2. ONLINE PHASE ---

Pveclev = []; % Sketching indices

for i = 1:length(opts.param)
    p_now = opts.param(i);
    if isKey(eigenvectors, p_now), continue; end
    
    mat_now = []; 
    
    % -- 2A. Sketched Rayleigh-Ritz Projection --
    if isempty(Pveclev)
        switch opts.sketch_target
            case 'X'
                [Pveclev,probs] = samp(Q, size(Q,2)*4, opts.sampling);
            case 'AX'
                TargetMat = Afull(p_now) * Q;
                [V_target, ~] = qr(TargetMat, 0);
                Pveclev = samp(V_target, size(V_target,2)*4, opts.sampling);
            case 'X_AX'
                TargetMat = [Q, Afull(p_now) * Q];
                [V_target, ~] = qr(TargetMat, 0);
                Pveclev = samp(V_target, size(V_target,2)*4, opts.sampling);
        end
    end
    
    if strcmp(opts.sampling, 'leverage')
        w = sqrt(length(Pveclev) * probs(Pveclev));
        w = w(:);
        SSnap = Q(Pveclev, :)./w;
        SAnowSnap = Arows(p_now, Pveclev, Q)./w; 
    else
        SSnap = Q(Pveclev, :);
        SAnowSnap = Arows(p_now, Pveclev, Q); 
    end

    % Solve the sketched generalized eigenvalue problem
    Mnow = lsqminnorm(SSnap, SAnowSnap);
    [x, eigen] = eig(Mnow);
    
    [~, ind] = sort(abs(diag(eigen)), sort_dir);
    x = x(:, ind);
    eigen_sorted = eigen(ind, ind);
    
    % -- 2B. Update Triggering Logic --
    do_update = (mod(i, opts.update_freq) == 0);
    
    if ~do_update
        X_n = x(:, 1:opts.neig);
        L_n = eigen_sorted(1:opts.neig, 1:opts.neig);
        
        AX_sketched = SAnowSnap * X_n;
        SX_sketched = SSnap * X_n;
        
        % Check sketched residual against tolerance
        res_nums = norm(AX_sketched - SX_sketched * L_n, 'fro');
        [~, closest_idx] = min(abs(cached_params - p_now));
        if ~is_matrix_free
            res_dens = cached_norms(closest_idx);
        else
            res_dens = Inf;
        end
        
        if (res_nums / res_dens) > opts.tol
            mat_now = Afull(p_now);
            V_m = Q * X_n;
            true_res = norm(mat_now * V_m - V_m * diag(L_n), 'fro') / norm(mat_now, 1);
            
            if ~is_matrix_free
                cached_params(end+1) = p_now;
                cached_norms(end+1) = norm(mat_now, 1);
            end
            
            if true_res > opts.tol
                do_update = true;
                warning('Step %d (p=%g): True residual (%e) exceeds tolerance.', i, p_now, true_res);
            end
        end
    end

    % -- 2C. Update--
    if do_update
        if ~is_matrix_free
            if isempty(mat_now)
                mat_now = Afull(p_now);
                cached_params(end+1) = p_now;
                cached_norms(end+1) = norm(mat_now, 1); 
            end
        end
        
        num_to_seed = min(size(x, 2), opts.neig + opts.s);
        start_vec = Q * x(:, 1:num_to_seed);
        
        % Compute exact eigenvalues to update the basis
        if is_matrix_free
            if opts.neig + opts.s == 1
                [X_true, D_true] = eigs(@(X) Amult(p_now, X), n, opts.neig + opts.s, opts.eigs_sigma, ...
                                    'Tolerance', opts.tol_upd, ...
                                    'StartVector', start_vec, 'IsFunctionSymmetric', opts.IsFunctionSymmetric);
            else
                [X_true, D_true] = my_eigs(@(X) Amult(p_now, X), n, opts.neig + opts.s, opts.eigs_sigma, ...
                                    'Tolerance', opts.tol_upd, ...
                                    'StartVector', start_vec, 'IsSymmetric', opts.IsFunctionSymmetric);
            end
        else
            if opts.neig + opts.s == 1
                [X_true, D_true] = eigs(mat_now, opts.neig + opts.s, opts.eigs_sigma, ...
                                    'Tolerance', opts.tol_upd, ...
                                    'StartVector', start_vec);
            else
                [X_true, D_true] = my_eigs(mat_now, opts.neig + opts.s, opts.eigs_sigma, ...
                                    'Tolerance', opts.tol_upd, ...
                                    'StartVector', start_vec);
            end
        end
        
        [~, idx] = sort(abs(diag(D_true)), sort_dir);
        new_eigenvecs_full = X_true(:, idx);
        
        eigenvalues(p_now) = diag(D_true(idx(1:opts.neig), idx(1:opts.neig)));
        eigenvectors(p_now) = new_eigenvecs_full(:, 1:opts.neig);
        
        % Subspace update
        vectors_to_add = [];
        Q_temp = Q; 
        for j = 1:size(new_eigenvecs_full, 2)
            v = new_eigenvecs_full(:, j);
            v_orth = v - Q_temp * (Q_temp' * v);
            if opts.trunc_tol == 0 || norm(v_orth) / (norm(v)) > opts.trunc_tol
                vectors_to_add = [vectors_to_add, v];
                Q_temp = [Q_temp, v_orth / (norm(v_orth))]; 
            end
        end
        
        if ~isempty(vectors_to_add)
            if size(X,2) > 500
                X(:, 1:size(vectors_to_add, 2)) = [];
            end
            X = [X, vectors_to_add];
            Q = orth(X);
        end
        Pveclev = []; % Force resampling
        
    else
        % Lazy Projection: Use ROM approximation without full matrix call
        num_to_extract = min(size(x, 2), opts.neig);
        eigenvectors(p_now) = Q * x(:, 1:num_to_extract);
        eigenvalues(p_now) = diag(eigen_sorted(1:opts.neig, 1:opts.neig));
    end
end
time_online = toc;
end