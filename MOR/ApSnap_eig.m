function [eigenvectors, eigenvalues, time_online, time_offline] = ApSnap_eig(Afull, n, opts)
% APSNAP_EIG Solves a parametric eigenvalue problem A(p)x = lambda*x
% using an Model Order Reduction (MOR) framework.
%
% This algorithm computes the eigenvalues/eigenvectors of a sequence of 
% matrices provided via a function handle Afull(p). It uses a Rayleigh-Ritz 
% projection onto a reduced subspace. An update strategy for the basis is implemented.
%
% =========================================================================
% INPUTS:
% =========================================================================
% Afull : Function handle. Afull(p) returns the n x n matrix for parameter p. Alternatively, for matrix-free operations, Afull can be a function handle that takes two arguments: Afull(p, X) returns A(p)*X.
% n     : Integer. Dimension of the full-space system.
% opts  : Struct containing algorithm configurations:
%
%   [Mandatory]
%   opts.param      : Array of parameters (p) for the online evaluation.
%   opts.neig       : Number of target eigenvalues/eigenvectors.
%   opts.k          : Number of initial snapshot points for training.
%
%   [Optional]
%   opts.eigs_sigma : Target eigenvalues (e.g., 'largestabs', 'smallestabs').
%   opts.s          : Oversampling (default: round(neig * 0.5)).
%   opts.SnapPoints : Custom parameter values for the offline phase.
%   opts.update_freq: Frequency of sliding window updates (default: Inf).
%   opts.trunc_tol  : Tolerance for rank truncation (default: 1e-8).
%
% =========================================================================

%% --- 0. SETUP AND OPTION VALIDATION ---

if nargin(Afull) == 2
    %  Matrix-Free -> Amult(p, X)
    is_matrix_free = true;
    Amult =Afull;
else
    % Full matrix -> Afull(p)
    is_matrix_free = false;
end


if ~isfield(opts, 'param'), error('opts.param is mandatory.'); end
if ~isfield(opts, 'neig'), error('opts.neig is mandatory.'); end
if ~isfield(opts, 'k'), error('opts.k is mandatory.'); end

% Set default values for optional parameters
if ~isfield(opts, 's'), opts.s = round(opts.neig * 0.5); end 
if ~isfield(opts, 'update_freq'), opts.update_freq = Inf; end
if ~isfield(opts, 'trunc_tol'), opts.trunc_tol = 1e-8; end
if ~isfield(opts, 'eigs_sigma'), opts.eigs_sigma = 'largestabs'; end
if ~isfield(opts, 'IsFunctionSymmetric'), opts.IsFunctionSymmetric = 0; end

% Generate default snapshot points if not provided
if ~isfield(opts, 'SnapPoints') || isempty(opts.SnapPoints)
    points = floor(linspace(1, length(opts.param), opts.k));
    opts.SnapPoints = opts.param(points);
end

% Determine sorting direction based on the eigenvalue target
is_smallest = isequal(opts.eigs_sigma, 'smallestabs') || isequal(opts.eigs_sigma, 'sm') ...
              || (isnumeric(opts.eigs_sigma) && opts.eigs_sigma == 0);
if is_smallest, sort_dir = 'ascend'; else, sort_dir = 'descend'; end

tic
%% --- 1. OFFLINE PHASE: Snapshot points ---
% Build the initial basis Q by collecting snapshots.
% -------------------------------------------------------------------------
X = zeros(n, (opts.s + opts.neig) * opts.k); 
eigenvectors = containers.Map('KeyType','double','ValueType','any');
eigenvalues = containers.Map('KeyType','double','ValueType','any');

for i = 1:opts.k
    p = opts.SnapPoints(i);
    % Compute full-order solution for snapshots
    if is_matrix_free
        [X_full, D_full] = eigs(@(X) Amult(p, X), n, opts.neig + opts.s, opts.eigs_sigma, 'MaxIterations', 2000, 'IsFunctionSymmetric', opts.IsFunctionSymmetric);
    else
        [X_full, D_full] = eigs(Afull(p), opts.neig + opts.s, opts.eigs_sigma, 'MaxIterations', 2000);
    end    
    % Sort results and store in the history matrix (X)
    [~, idx] = sort(abs(diag(D_full)), sort_dir);
    X(:, (opts.s + opts.neig)*(i-1)+1 : (opts.s + opts.neig)*i) = X_full(:, idx);
    
    % Cache solutions for the snapshot parameters
    eigenvectors(p) = X_full(:, idx(1:opts.neig));
    eigenvalues(p)  = diag(D_full(idx(1:opts.neig), idx(1:opts.neig)));
end

% Construct the initial orthonormal basis Q
Q = orth(X);
time_offline = toc;

tic
%% --- 2. ONLINE PHASE ---
% -------------------------------------------------------------------------
for i = 1:length(opts.param)
    
    p_now = opts.param(i);
    
    % Skip evaluation if the result exists in the cache (from offline phase)
    if isKey(eigenvectors, p_now), continue; end
    
    % --- 2A. Full Matrix Projection ---
    % Note: Without affine structure, Afull(p) must be evaluated and projected.
    % Complexity: O(Cost(Afull) + n * m^2) where m is the reduced dimension.
    if is_matrix_free
        % Matrix-free product
        A_Q = Amult(p_now, Q);
    else
        % Standard method: matrix assembly and multiplication
        mat = Afull(p_now);
        A_Q = mat * Q; 
    end    
    % --- 2B. Reduced Rayleigh-Ritz Step ---
    Mnow = Q' * A_Q;
    [x_red, eigen_mat] = eig(Mnow);
    
    % Sort reduced eigenvalues/eigenvectors
    [~, ind] = sort(abs(diag(eigen_mat)), sort_dir);
    x_red = x_red(:, ind);
    eigen_sorted = eigen_mat(ind, ind);
    
    % --- 2C. Eigenvector Projection & Storage ---
    % Determine if an update step is required to enrich the basis
    is_update_step = (mod(i, opts.update_freq) == 0);
    
    if is_update_step
        num_to_extract = min(size(x_red, 2), opts.neig + opts.s);
        new_eigenvecs_full = Q * x_red(:, 1:num_to_extract);
    else
        num_to_extract = min(size(x_red, 2), opts.neig);
        eigenvectors(p_now) = Q * x_red(:, 1:num_to_extract);
        eigenvalues(p_now) = diag(eigen_sorted(1:opts.neig, 1:opts.neig));
    end
    
    
    
    % --- 2D. Subspace update ---
    if is_update_step
        vectors_to_add = [];
        Q_temp = Q; % Temporary basis for orthogonalization

        if is_matrix_free
            if opts.neig + opts.s == 1
                [new_eigenvecs_full, D] = eigs(@(X) Amult(p_now, X), n, opts.neig + opts.s, opts.eigs_sigma, 'StartVector', new_eigenvecs_full, 'Tolerance', opts.tol_upd, 'IsFunctionSymmetric', opts.IsFunctionSymmetric);
            else
                [new_eigenvecs_full, D] = my_eigs(@(X) Amult(p_now, X), n, opts.neig + opts.s, opts.eigs_sigma, 'StartVector', new_eigenvecs_full, 'Tolerance', opts.tol_upd, 'IsSymmetric', opts.IsFunctionSymmetric);
            end
        else
            if opts.neig + opts.s == 1
                [new_eigenvecs_full, D] = eigs(mat, opts.neig + opts.s, opts.eigs_sigma, 'StartVector', new_eigenvecs_full, 'Tolerance', opts.tol_upd);
            else
                [new_eigenvecs_full, D] = my_eigs(mat, opts.neig + opts.s, opts.eigs_sigma, 'StartVector', new_eigenvecs_full, 'Tolerance', opts.tol_upd);
            end
        end        
        [~, idx] = sort(abs(diag(D)), sort_dir);
        new_eigenvecs_full = new_eigenvecs_full(:, idx);
        eigenvalues(p_now) = diag(D(idx(1:opts.neig), idx(1:opts.neig)));
        eigenvectors(p_now) = new_eigenvecs_full(:, 1:opts.neig);
        
        for j = 1:size(new_eigenvecs_full, 2)
            v = new_eigenvecs_full(:, j);
            
            % Orthogonalize against the current basis
            v_orth = v - Q_temp * (Q_temp' * v); 
            
            % Truncate based on the relative norm of the orthogonal component
            if norm(v_orth) / (norm(v)) > opts.trunc_tol
                vectors_to_add = [vectors_to_add, v];
                % Update local basis for subsequent vectors within this step
                Q_temp = [Q_temp, v_orth / (norm(v_orth))]; 
            end
        end
        
        if ~isempty(vectors_to_add)
            num_added = size(vectors_to_add, 2);
            % subspace update
            if size(X,2) > 500
            X(:, 1:num_added) = [];
            end
            X = [X, vectors_to_add];
            
            % Re-orthogonalize the basis
            Q = orth(X);
        end
    end
end
time_online = toc;

end