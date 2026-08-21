function [V, D] = my_eigs(A, varargin)
    % Block Eigenvalue Decomposition (Lanczos/Arnoldi).
    % Supports explicit matrices and matrix-free function handles.
    %
    % Syntax 1 (Matrix): [V, D] = my_eigs(A, k, sigma, 'Name', Value, ...)
    % Syntax 2 (Function): [V, D] = my_eigs(Afun, n, k, sigma, 'Name', Value, ...)

    % --- 1. Input Parsing & Validation ---
    is_func = isa(A, 'function_handle');
    
    if is_func
        % Matrix-free mode: n is expected as the first parameter.
        if isempty(varargin)
            error('Per una function handle devi specificare la dimensione n.');
        end
        n = varargin{1};
        k = varargin{2};
        args_start = 3;
    else
        % Explicit matrix mode.
        [n, cols] = size(A);
        if n ~= cols
            error('La matrice A deve essere quadrata.');
        end
        k = varargin{1};
        args_start = 2;
    end
    
    % Check for the optional 'sigma' parameter.
    if length(varargin) >= args_start && (ischar(varargin{args_start}) || isstring(varargin{args_start}))
        sigma = varargin{args_start};
        args_start = args_start + 1;
    else
        sigma = 'largestabs';
    end
    
    % Parse remaining Name-Value parameters.
    nv_pairs = varargin(args_start:end);
    p = inputParser;
    p.KeepUnmatched = true;
    addParameter(p, 'Tolerance', 1e-6);
    addParameter(p, 'MaxBlocks', 50);
    addParameter(p, 'StartVector', []);
    addParameter(p, 'IsSymmetric', []); % Useful for function handles.
    parse(p, nv_pairs{:});
    
    tol = p.Results.Tolerance;
    max_blocks = p.Results.MaxBlocks;
    V_guess = p.Results.StartVector;
    which_type = lower(sigma);
    
    if isempty(V_guess)
        V_guess = randn(n, k);
    end

    % Determine symmetry.
    if isempty(p.Results.IsSymmetric)
        if is_func
            is_sym = false; % Default for function handles if unspecified.
        else
            is_sym = isequal(A, A'); % Automatic calculation for matrices.
        end
    else
        is_sym = p.Results.IsSymmetric;
    end

    % Configure Shift-and-Invert for interior eigenvalues.
    if strcmp(which_type, 'smallestabs')
        invert_mode = true;
        if ~is_func
            dA = decomposition(A); % Precompute LU/Cholesky for explicit matrices.
        end
    elseif strcmp(which_type, 'largestabs')
        invert_mode = false;
    else
        error('Invalid sigma. Usa ''largestabs'' o ''smallestabs''.');
    end

    % --- 2. Preallocation & Initialization ---
    Q = zeros(n, k * max_blocks);
    H = zeros(k * max_blocks, k * max_blocks); 
    [Q(:, 1:k), ~] = qr(V_guess, 0);
    B_prev = zeros(k, k);
    actual_blocks = max_blocks; 

    % --- 3. Block Krylov Iteration ---
    for j = 1:max_blocks
        idx_curr = (j-1)*k + 1 : j*k;
        V_j = Q(:, idx_curr);
        
        Z = zeros(n, k);
        
        % --- Compute Z = A * V_j or Z = A \ V_j ---
        if invert_mode
            if is_func
                % Iterative resolution for function handles (column-by-column).
                for c = 1:k
                    [Z(:, c), ~] = gmres(A, V_j(:, c), [], tol, min(n, 100));
                end
            else
                % Vectorized forward/backward substitution.
                Z = dA \ V_j; 
            end
        else
            if is_func
                % Apply function column-by-column.
                for c = 1:k
                    Z(:, c) = A(V_j(:, c));
                end
            else
                % Standard matrix-vector product.
                Z = A * V_j; 
            end
        end
        
        % --- Orthogonalization strategy ---
        if is_sym
            % Block Lanczos
            if j > 1
                idx_prev = (j-2)*k + 1 : (j-1)*k;
                V_prev = Q(:, idx_prev);
                Z = Z - V_prev * B_prev'; 
            end
            A_j = V_j' * Z;
            A_j = (A_j + A_j') / 2;
            H(idx_curr, idx_curr) = A_j;
            Z = Z - V_j * A_j;
        else
            % Block Arnoldi
            for i = 1:j
                idx_i = (i-1)*k + 1 : i*k;
                V_i = Q(:, idx_i);
                H_ij = V_i' * Z;
                H(idx_i, idx_curr) = H_ij;
                Z = Z - V_i * H_ij;
            end
        end
        
        % Early QR factorization
        [V_next, B_next] = qr(Z, 0);
        
        % --- Convergence Check ---
        current_size = j * k;
        [Y_curr, D_curr] = eig(H(1:current_size, 1:current_size));
        
        [~, sort_idx] = sort(abs(diag(D_curr)), 'descend');
        Y_top_k = Y_curr(:, sort_idx(1:k));
        Y_bottom = Y_top_k(end-k+1:end, :);
        
        max_val = max(abs(diag(D_curr))); 
        res_norm = norm(B_next * Y_bottom, 'fro') / (max_val + eps);
        
        if res_norm < tol
            actual_blocks = j;
            break;
        end
        
        % Store basis
        if j < max_blocks
            idx_next = j*k + 1 : (j+1)*k;
            Q(:, idx_next) = V_next;
            H(idx_next, idx_curr) = B_next;
            if is_sym
                H(idx_curr, idx_next) = B_next'; 
                B_prev = B_next;
            end
        end
    end
    
    % --- 4. Rayleigh-Ritz Projection & Post-processing ---
    final_size = actual_blocks * k;
    Q = Q(:, 1:final_size);
    H = H(1:final_size, 1:final_size);
    
    [Y, D_small] = eig(H);
    all_evs = diag(D_small);
    [~, final_idx] = sort(abs(all_evs), 'descend');
    
    V = Q * Y(:, final_idx(1:k));
    
    if invert_mode
        D = diag(1 ./ all_evs(final_idx(1:k)));
    else
        D = diag(all_evs(final_idx(1:k)));
    end
end