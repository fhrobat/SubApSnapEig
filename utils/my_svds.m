function [U, S, V] = my_svds(A, k, varargin)
    % Block Singular Value Decomposition (Lanczos method).
    % Computes the k largest singular values and vectors.

    % --- 1. Input Parsing & Validation ---
    [rows, cols] = size(A);
    p = inputParser;
    addRequired(p, 'A');
    addRequired(p, 'k');
    addParameter(p, 'Tolerance', 1e-6);
    addParameter(p, 'MaxBlocks', 50);
    addParameter(p, 'RightStartVector', []);
    parse(p, A, k, varargin{:});
    
    tol = p.Results.Tolerance;
    max_blocks = p.Results.MaxBlocks;
    V_guess = p.Results.RightStartVector;
    
    % Default to random starting block if unspecified.
    if isempty(V_guess)
        V_guess = randn(cols, k);
    end

    % Check for strict symmetry to optimize products.
    is_sym = (rows == cols) && isequal(A, A');
    
    % --- 2. Preallocation & Initialization ---
    Q = zeros(cols, k * max_blocks);
    T = zeros(k * max_blocks, k * max_blocks);
    
    % Orthonormalize initial guess.
    [Q(:, 1:k), ~] = qr(V_guess, 0);
    B_prev = zeros(k, k);
    actual_blocks = max_blocks; 

    % --- 3. Block Lanczos Iteration ---
    for j = 1:max_blocks
        idx_curr = (j-1)*k + 1 : j*k;
        V_j = Q(:, idx_curr);
        
        % Apply linear operator (A or A'A).
        if is_sym
            Z = A * V_j;         
        else
            Z = A' * (A * V_j);  
        end
        
        % 3-term recurrence: orthogonalize against previous block.
        if j > 1
            idx_prev = (j-2)*k + 1 : (j-1)*k;
            V_prev = Q(:, idx_prev);
            Z = Z - V_prev * B_prev'; 
        end
        
        % Orthogonalize against current block and symmetrize projection.
        A_j = V_j' * Z;
        A_j = (A_j + A_j') / 2; 
        T(idx_curr, idx_curr) = A_j;
        Z = Z - V_j * A_j;
        
        % Early QR factorization for convergence check.
        [V_next, B_next] = qr(Z, 0);
        
        % --- Convergence Check (True Ritz Residual) ---
        % Solve projected eigenvalue problem.
        current_size = j * k;
        [Y_curr, D_curr] = eig(T(1:current_size, 1:current_size));
        
        % Extract bottom k components of top k eigenvectors.
        [~, sort_idx] = sort(abs(diag(D_curr)), 'descend');
        Y_top_k = Y_curr(:, sort_idx(1:k));
        Y_bottom = Y_top_k(end-k+1:end, :);
        
        % Compute orthogonal-invariant relative residual.
        max_val = max(abs(diag(D_curr))); 
        res_norm = norm(B_next * Y_bottom, 'fro') / (max_val + eps);
        
        if res_norm < tol
            actual_blocks = j;
            break; % Subspace converged.
        end
        
        % Store basis and off-diagonal blocks for next iteration.
        if j < max_blocks
            idx_next = j*k + 1 : (j+1)*k;
            Q(:, idx_next) = V_next;
            T(idx_next, idx_curr) = B_next;
            T(idx_curr, idx_next) = B_next';
            B_prev = B_next;
        end
    end
    
    % --- 4. Rayleigh-Ritz Projection & Post-processing ---
    % Truncate preallocated matrices to actual size.
    final_size = actual_blocks * k;
    Q = Q(:, 1:final_size);
    T = T(1:final_size, 1:final_size);
    
    % Final eigendecomposition of tridiagonal matrix.
    [Y, D] = eig(T);
    [autovalori, sort_idx] = sort(diag(D), 'descend');
    Y = Y(:, sort_idx);
    
    % Reconstruct right singular vectors.
    V = Q * Y(:, 1:k);
    
    % Compute left singular vectors and scaling.
    if is_sym
        S_vec = abs(autovalori(1:k));
        U = (A * V) ./ S_vec'; 
    else
        S_vec = sqrt(abs(autovalori(1:k))); 
        U_unscaled = A * V;
        U = U_unscaled ./ S_vec';
    end
    
    S = diag(S_vec);
end