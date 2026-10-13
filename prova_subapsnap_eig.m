% warning off;
clear

%% Exponential Problem Setup
% N = 2*10^6;                             % Problem dimension
% L = 1;                                  % Domain length/parameter
% Afull =@(p) -exp_full(p, L, N);   % Full matrix handle
% Arows =@(p,rows,M) -exp_rows(p, L, N, rows)*M; % Row-sampled matrix handle
% neig  = 1;                              % Number of eigenvalues to compute
% param = logspace(-1,0,3000)';           % Parameter space grid
% IsFunctionSymmetric = 1;                % Symmetry flag
% n = size(Afull(param(1)),2);            % Matrix size
%%


%% PageRank Problem Setup
% Load Notre Dame web graph dataset
name_file = 'web-NotreDame.mat'; 
data = load(name_file);
fields = fieldnames(data);
Problem = data.(fields{1});
A = Problem.A;

n = size(A, 1);
A = spones(A); % Convert to unweighted adjacency matrix
A = A - spdiags(diag(A), 0, n, n); % Remove self-loops

% Handle dangling nodes (nodes with out-degree 0)
out_degrees = sum(A, 1);
dangling = find(out_degrees == 0);
if ~isempty(dangling)
    A = A + sparse(dangling, dangling, 1, n, n);
    out_degrees = sum(A, 1);
end

% Compute Markov transition matrix
M = A * spdiags(1 ./ out_degrees', 0, n, n);
in_deg = sum(M > 0, 2); % Compute in-degrees
M_T = M';

% Define matrix handles for PageRank
Afull = @(t) pagerank_full(t, M);
Arows = @(t, rows, X) pagerank_rows(t, rows, X, M_T, in_deg);
Amult = @(t, X) pagerank_mult(t, X, M, in_deg);

neig  = 1;                                % Number of eigenvalues
param = linspace(0, 0.5, 1000)';          % Parameter range (teleportation/scaling)
IsFunctionSymmetric = 0;                  % PageRank matrix is non-symmetric

% Custom PageRank multiplication function
function mat = pagerank_mult(t, X, M, in_deg)
    persistent last_t w col_sums
    
    % Update persistent variables only if parameter 't' changes
    if isempty(last_t) || last_t ~= t
        last_t = t;
        w = (1 + in_deg) .^ t;
        col_sums = w' * M;
    end
    
    n = size(M, 1);
    p = 0.9; % Damping factor
    
    % Apply scaled transitions and teleportation
    X_scaled = X ./ col_sums';
    M_X = (M * X_scaled) .* w;
    
    teleport_X = (1 / n) * sum(X, 1);
    mat = p * M_X + (1 - p) * teleport_X;
end
%%

%% Differential Operator Setup
% N = 500;
% eps = 1e-4;
% L = 1;
% Afull =@(p)  diff_full(p, L, N,eps);
% Arows =@(p,rows,M)  diff_rows(p, L, N,eps, rows)*M;
% neig = 3;
% param = logspace(0,1,1000)';
%%

%% Kernel Problem Setup
% load('mnist_completo.mat', 'X'); 
% N = size(X, 1);
%
% K = 100; % Number of nearest neighbors
%
% [idx, D] = knnsearch(X, X, 'K', K);
% D2_knn = D.^2;
%
% D2_neighbors = D2_knn(:, 2:end); 
% p_start = median(D2_neighbors(:));
%
% N_total = size(X, 1);
% i_idx = repmat((1:N_total)', 1, K);
% j_idx = idx;
%
% % Build sparse symmetric distance matrix
% D2_sparse = sparse(i_idx, j_idx, D2_knn + eps, N_total, N_total);
% D2_sym = max(D2_sparse, D2_sparse');
%
% Afull = @(p) build_sparse_symmetric(D2_sym, p, N_total);
% Arows = @(p, r, M) row_symmetric(D2_sym(:, r), p, M);
%
% param = linspace(p_start * 0.5, p_start * 1.5, 1000)';
% neig = 1;
% IsFunctionSymmetric = 1;
% n = size(Afull(param(1)),2);
%
% function S = build_sparse_symmetric(D2_sym, p, N)
%     [i, j, v] = find(D2_sym);
%     v_aff = exp(-(v - eps) / (2*p^2));
%     S = sparse(i, j, v_aff, N, N);
% end
%
% function res = row_symmetric(D2_cols, p, M)
%     [i, j, v] = find(D2_cols); 
%     S_vals = exp(-(v - eps) / (2*p^2)); 
%     S_sub = sparse(i, j, S_vals, size(D2_cols, 1), size(D2_cols, 2));
%     res = S_sub' * M; 
% end
%%

%% Eigenvector Centrality Setup
% data = load('coAuthorsDBLP.mat'); 
% A = data.Problem.A; 
% N = size(A, 1);
% A_base = spones(A + A');
% rng(42);
% Phi = rand(N, 1) * 2 * pi;
% get_dp = @(p) 1 + 0.5 * sin(2 * pi * p / 1000 + Phi);
% Afull = @(p) spdiags(get_dp(p),0,N,N) * A_base * spdiags(get_dp(p),0,N,N);
% Amult = @(p,X) get_dp(p).* (A_base * (get_dp(p).*X));
% Arows = @(p, r, M) calcola_Arows(p, r, M, A_base, get_dp);
% 
% function res = calcola_Arows(p, r, M, A_base, get_dp)
%     dp = get_dp(p);
%     riga_sparsa = A_base(:, r)'; 
%     res = dp(r) .* (riga_sparsa * (dp .* M));
% end
% 
% neig  = 1;
% param = (1:1000)';
% IsFunctionSymmetric = 1;
% n = size(Afull(param(1)),2);
%%


% n = size(Afull(param(1)),2);
k = 5;
s = 0;
tol = Inf;
tol_upd = 1e-8;
update_freq = Inf;
sigma = 'largestabs';

%% Execution of the Three Methods (Save outputs with distinct names)
%% Settings and Parameters Configuration
opts = [];
opts.eigs_sigma = sigma;
opts.param = param;
opts.neig = neig;
opts.k = k;
opts.s = s;
opts.tol_upd = tol_upd;
opts.update_freq = update_freq;
opts.tol = tol; 
opts.IsFunctionSymmetric = IsFunctionSymmetric;
is_smallest = isequal(opts.eigs_sigma, 'smallestabs') || isequal(opts.eigs_sigma, 'sm') || (isnumeric(opts.eigs_sigma) && opts.eigs_sigma == 0);
if is_smallest, sort_dir = 'ascend'; else, sort_dir = 'descend'; end

% Snapshot configuration options (uncomment as needed)
% opts.SnapPoints = logspace(0,1,k)';
% opts.SnapPoints = param(1:k);
% opts.SnapPoints = [param(1:k);param(floor(linspace(k+1, length(param), k)))];

if ~isfield(opts, 'SnapPoints') || isempty(opts.SnapPoints)
    points = floor(linspace(1, length(opts.param), opts.k));
    opts.SnapPoints = opts.param(points);
end

param_test = param(round(linspace(1, length(param), 100)));
is_snapshot = ismembertol(param_test, opts.SnapPoints, 1e-10);
param_test = param_test(~is_snapshot);
% param_test = param; % Uncomment to execute on the full parameter space

n_test = length(param_test);

%% Preallocation for Errors and Plotting Data
err_res = zeros(n_test, 3);
err_lam = zeros(n_test, 3);
err_ang = zeros(n_test, 3); 
vals_ex = zeros(n_test, neig);
vals_approx = zeros(n_test, neig, 3);

warning off;

%% PHASE 0: Exact Solutions Calculation 
disp('--- PHASE 0: Exact Solutions ---');
V_ex_all = cell(n_test, 1);
lam_ex_all = cell(n_test, 1);

msg = '';
for i = 1:n_test
    p = param_test(i);
    % A_p = Afull(p);
    
    try 
        [V_ex, D_ex] = eigs(@(X) Amult(p, X), n, neig, opts.eigs_sigma);
    catch
        [V_ex, D_ex] = eigs(Afull(p), neig, opts.eigs_sigma); 
    end
    lam_ex = diag(D_ex);
    
    [~, idx_ex] = sort(abs(lam_ex), sort_dir);
    lam_ex_all{i} = lam_ex(idx_ex);
    V_ex_all{i} = V_ex(:, idx_ex);
    vals_ex(i, :) = lam_ex_all{i}.'; 
    
    fprintf(repmat('\b', 1, length(msg)));
    msg = sprintf('Exact Solutions: [%3.1f%%]', (i / n_test) * 100);
    fprintf('%s', msg);
end
fprintf('\n');
clear A_p V_ex D_ex lam_ex; % Free temporary full matrices

warning on;

%% PHASE 1: LU Method
disp('--- PHASE 1: LU Method ---');
opts.sampling = 'lu';
% [eigvec_lu, eigval_lu, t_on_lu, t_off_lu] = SubApSnap_eig(Afull, Arows, n, opts);
[eigvec_lu, eigval_lu, t_on_lu, t_off_lu] = SubApSnap_eig(Amult, Arows, n, opts); %%This is for PageRank only, otherwise uncomment previous line and comment this one
% [eigvec_lu, eigval_lu, t_on_lu, t_off_lu] = SubApSnap_Affine_eig(A0, A1, n, opts);
time_lu = t_on_lu + t_off_lu;

msg = '';
for i = 1:n_test
    p = param_test(i);
    % A_p = Afull(p); % Regenerated on the fly for residual computation
    
    V_m = eigvec_lu(p);
    lam_m = eigval_lu(p);
    if size(lam_m, 1) == size(lam_m, 2), lam_m = diag(lam_m); end
    
    lam_ex = lam_ex_all{i}; V_ex = V_ex_all{i};
    perm_idx = perms(1:neig); best_cost = Inf; best_perm = 1:neig;
    for pp = 1:size(perm_idx,1)
        cost = norm(lam_m(perm_idx(pp,:)) - lam_ex);
        if cost < best_cost, best_cost = cost; best_perm = perm_idx(pp,:); end
    end
    lam_m = lam_m(best_perm); V_m = V_m(:, best_perm);
    
    try
        err_res(i, 1) = norm(Amult(p,V_m) - V_m * diag(lam_m), 'fro');
    catch
        err_res(i, 1) = norm( Afull(p) * V_m - V_m * diag(lam_m), 'fro');
    end
    err_lam(i, 1) = mean(abs(lam_m - lam_ex) ./ abs(lam_ex), 'omitnan');
    err_ang(i, 1) = subspace(V_m, V_ex);
    vals_approx(i, :, 1) = lam_m.';
    
    fprintf(repmat('\b', 1, length(msg))); msg = sprintf('LU Evaluation: [%3.1f%%]', (i / n_test) * 100); fprintf('%s', msg);
end
fprintf('\n');
clear eigvec_lu eigval_lu V_m lam_m A_p; % Memory release for LU variant

%% PHASE 2: Leverage Method
disp('--- PHASE 2: Leverage Method ---');
opts.sampling = 'leverage';
% [eigvec_lev, eigval_lev, t_on_lev, t_off_lev] = SubApSnap_eig(Afull, Arows, n, opts);
[eigvec_lev, eigval_lev, t_on_lev, t_off_lev] = SubApSnap_eig(Amult, Arows, n, opts); %%This is for PageRank only, otherwise uncomment previous line and comment this one
% [eigvec_lev, eigval_lev, t_on_lev, t_off_lev] = SubApSnap_Affine_eig(A0, A1, n, opts);
time_lev = t_on_lev + t_off_lev;

msg = '';
for i = 1:n_test
    p = param_test(i);
    % A_p = Afull(p);
    
    V_m = eigvec_lev(p);
    lam_m = eigval_lev(p);
    if size(lam_m, 1) == size(lam_m, 2), lam_m = diag(lam_m); end
    
    lam_ex = lam_ex_all{i}; V_ex = V_ex_all{i};
    perm_idx = perms(1:neig); best_cost = Inf; best_perm = 1:neig;
    for pp = 1:size(perm_idx,1)
        cost = norm(lam_m(perm_idx(pp,:)) - lam_ex);
        if cost < best_cost, best_cost = cost; best_perm = perm_idx(pp,:); end
    end
    lam_m = lam_m(best_perm); V_m = V_m(:, best_perm);
    
    try
        err_res(i, 2) = norm(Amult(p,V_m) - V_m * diag(lam_m), 'fro');
    catch
        err_res(i, 2) = norm( Afull(p) * V_m - V_m * diag(lam_m), 'fro');
    end

    err_lam(i, 2) = mean(abs(lam_m - lam_ex) ./ abs(lam_ex), 'omitnan');
    err_ang(i, 2) = subspace(V_m, V_ex);
    vals_approx(i, :, 2) = lam_m.';
    
    fprintf(repmat('\b', 1, length(msg))); msg = sprintf('Leverage Evaluation: [%3.1f%%]', (i / n_test) * 100); fprintf('%s', msg);
end
fprintf('\n');
clear eigvec_lev eigval_lev V_m lam_m A_p; % Memory release for Leverage variant

%% PHASE 3: POD Method
disp('--- PHASE 3: POD Method ---');
% [eigevec_pod, eigeval_pod, t_on_pod, t_off_pod] = ApSnap_eig(Afull, n, opts);
[eigevec_pod, eigeval_pod, t_on_pod, t_off_pod] = ApSnap_eig(Amult, n, opts); %%This is for PageRank only, otherwise uncomment previous line and comment this one
% [eigevec_pod, eigeval_pod, t_on_pod, t_off_pod] = ApSnap_Affine_eig(A0, A1, n, opts);
time_pod = t_on_pod + t_off_pod;

msg = '';
for i = 1:n_test
    p = param_test(i);
    % A_p = Afull(p);
    
    V_m = eigevec_pod(p);
    lam_m = eigeval_pod(p);
    if size(lam_m, 1) == size(lam_m, 2), lam_m = diag(lam_m); end
    
    lam_ex = lam_ex_all{i}; V_ex = V_ex_all{i};
    perm_idx = perms(1:neig); best_cost = Inf; best_perm = 1:neig;
    for pp = 1:size(perm_idx,1)
        cost = norm(lam_m(perm_idx(pp,:)) - lam_ex);
        if cost < best_cost, best_cost = cost; best_perm = perm_idx(pp,:); end
    end
    lam_m = lam_m(best_perm); V_m = V_m(:, best_perm);
    
    try
        err_res(i, 3) = norm(Amult(p,V_m) - V_m * diag(lam_m), 'fro');
    catch
        err_res(i, 3) = norm( Afull(p) * V_m - V_m * diag(lam_m), 'fro');
    end

    err_lam(i, 3) = mean(abs(lam_m - lam_ex) ./ abs(lam_ex), 'omitnan');
    err_ang(i, 3) = subspace(V_m, V_ex);
    vals_approx(i, :, 3) = lam_m.';
    
    fprintf(repmat('\b', 1, length(msg))); msg = sprintf('POD Evaluation: [%3.1f%%]', (i / n_test) * 100); fprintf('%s', msg);
end
fprintf('\nComputation completed successfully!\n');

%% Final Global Cleanup
clear eigevec_pod eigeval_pod V_m lam_m A_p V_ex_all lam_ex_all;

%% Plot Results (NaNs Filtered) - Improved Readability
figure('Name', 'Subspace Error and Eigenvalue Analysis', ...
       'Position', [100, 100, 1200, 850], ...
       'Color', 'w');
t = tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% Define distinguishable colors for plotting
colors = [0.85 0.33 0.10;   % LU
          0.00 0.45 0.74;   % Leverage
          0.47 0.67 0.19];  % POD
labels = {'LU', 'Leverage', 'POD'};
markers = {'o', 's', '^'};
lineStyles = {'-', '-', '-'};
m_bg = 2;          % Leverage in background
fg_methods = [1 3]; % LU and POD in foreground

%% 1. Residual Plot
ax = nexttile;
set(ax, 'YScale', 'log');
hold on;
% Leverage (background)
m = m_bg;
valid_idx = ~isnan(err_res(:, m));
idx_plot = find(valid_idx);
plot(param_test(valid_idx), err_res(valid_idx, m), ...
    'Color', colors(m,:), ...
    'LineStyle', lineStyles{m}, ...
    'LineWidth', 1.0, ...
    'Marker', markers{m}, ...
    'MarkerSize', 3.5, ...
    'MarkerIndices', idx_plot(1:max(1,floor(numel(idx_plot)/12)):numel(idx_plot)), ...
    'DisplayName', labels{m});

% Foreground methods
for m = fg_methods
    valid_idx = ~isnan(err_res(:, m));
    idx_plot = find(valid_idx);
    plot(param_test(valid_idx), err_res(valid_idx, m), ...
        'Color', colors(m,:), ...
        'LineStyle', lineStyles{m}, ...
        'LineWidth', 1.8, ...
        'Marker', markers{m}, ...
        'MarkerSize', 5, ...
        'MarkerIndices', idx_plot(1:max(1,floor(numel(idx_plot)/12)):numel(idx_plot)), ...
        'DisplayName', labels{m});
end
title('Relative Residual Error', 'FontWeight', 'bold');
xlabel('Parameter p');
ylabel('Residual');
legend('Location', 'best');
grid on; grid minor; box on;

%% 2. Eigenvalue Error Plot
ax = nexttile;
set(ax, 'YScale', 'log');
hold on;
m = m_bg;
valid_idx = ~isnan(err_lam(:, m));
idx_plot = find(valid_idx);
plot(param_test(valid_idx), err_lam(valid_idx, m), ...
    'Color', colors(m,:), ...
    'LineStyle', lineStyles{m}, ...
    'LineWidth', 1.0, ...
    'Marker', markers{m}, ...
    'MarkerSize', 3.5, ...
    'MarkerIndices', idx_plot(1:max(1,floor(numel(idx_plot)/12)):numel(idx_plot)), ...
    'DisplayName', labels{m});

for m = fg_methods
    valid_idx = ~isnan(err_lam(:, m));
    idx_plot = find(valid_idx);
    plot(param_test(valid_idx), err_lam(valid_idx, m), ...
        'Color', colors(m,:), ...
        'LineStyle', lineStyles{m}, ...
        'LineWidth', 1.8, ...
        'Marker', markers{m}, ...
        'MarkerSize', 5, ...
        'MarkerIndices', idx_plot(1:max(1,floor(numel(idx_plot)/12)):numel(idx_plot)), ...
        'DisplayName', labels{m});
end
title('Relative Eigenvalue Error', 'FontWeight', 'bold');
xlabel('Parameter p');
ylabel('Error');
legend('Location', 'best');
grid on; grid minor; box on;

%% 3. Subspace Angle Plot
ax = nexttile;
set(ax, 'YScale', 'log');
hold on;
m = m_bg;
valid_idx = ~isnan(err_ang(:, m));
idx_plot = find(valid_idx);
plot(param_test(valid_idx), err_ang(valid_idx, m), ...
    'Color', colors(m,:), ...
    'LineStyle', lineStyles{m}, ...
    'LineWidth', 1.0, ...
    'Marker', markers{m}, ...
    'MarkerSize', 3.5, ...
    'MarkerIndices', idx_plot(1:max(1,floor(numel(idx_plot)/12)):numel(idx_plot)), ...
    'DisplayName', labels{m});

for m = fg_methods
    valid_idx = ~isnan(err_ang(:, m));
    idx_plot = find(valid_idx);
    plot(param_test(valid_idx), err_ang(valid_idx, m), ...
        'Color', colors(m,:), ...
        'LineStyle', lineStyles{m}, ...
        'LineWidth', 1.8, ...
        'Marker', markers{m}, ...
        'MarkerSize', 5, ...
        'MarkerIndices', idx_plot(1:max(1,floor(numel(idx_plot)/12)):numel(idx_plot)), ...
        'DisplayName', labels{m});
end
title('Subspace Similarity', 'FontWeight', 'bold');
xlabel('Parameter p');
legend('Location', 'best');
grid on; grid minor; box on;

%% 4. Eigenvalues Plot
ax = nexttile;
hold on;
for j = 1:neig
    valid_idx = ~isnan(vals_approx(:, j, m_bg));
    x = param_test(valid_idx);
    y = real(vals_approx(valid_idx, j, m_bg));
    if j == 1
        plot(x, y, 'Color', colors(m_bg,:), 'LineStyle', 'none', ...
            'Marker', markers{m_bg}, 'MarkerSize', 2.5, ...
            'DisplayName', labels{m_bg});
    else
        plot(x, y, 'Color', colors(m_bg,:), 'LineStyle', 'none', ...
            'Marker', markers{m_bg}, 'MarkerSize', 2.5, ...
            'HandleVisibility', 'off');
    end
end
for m = fg_methods
    for j = 1:neig
        valid_idx = ~isnan(vals_approx(:, j, m));
        x = param_test(valid_idx);
        y = real(vals_approx(valid_idx, j, m));
        if j == 1
            plot(x, y, 'Color', colors(m,:), 'LineStyle', 'none', ...
                'Marker', markers{m}, 'MarkerSize', 3.5, ...
                'DisplayName', labels{m});
        else
            plot(x, y, 'Color', colors(m,:), 'LineStyle', 'none', ...
                'Marker', markers{m}, 'MarkerSize', 3.5, ...
                'HandleVisibility', 'off');
        end
    end
end
for j = 1:neig
    valid_idx = ~isnan(vals_ex(:, j));
    x = param_test(valid_idx);
    y = real(vals_ex(valid_idx, j));
    if j == 1
        plot(x, y, 'k.', 'MarkerSize', 10, 'DisplayName', 'Exact');
    else
        plot(x, y, 'k.', 'MarkerSize', 10, 'HandleVisibility', 'off');
    end
end
title('Eigenvalues', 'FontWeight', 'bold');
xlabel('Parameter p');
ylabel('Real part');
legend('Location', 'eastoutside');
grid on; grid minor; box on;

%% Overall Title
title(t, 'Comparison of Residual, Eigenvalue, and Subspace Errors', ...
    'FontSize', 14, 'FontWeight', 'bold');

%% 5. Computational Times Table and Ratios (vs POD)
Metodo = {'LU'; 'Leverage'; 'POD'};
Fase_Offline = [t_off_lu; t_off_lev; t_off_pod];
Fase_Online  = [t_on_lu; t_on_lev; t_on_pod];
Tempo_Totale = Fase_Offline + Fase_Online;

% Compute ratios relative to POD
Rapporto_Offline = Fase_Offline ./ t_off_pod;
Rapporto_Online  = Fase_Online ./ t_on_pod;
Rapporto = Tempo_Totale ./ time_pod;

% --- Option 1: Print table in the Command Window ---
disp('--- Confronto Tempi di Esecuzione e Rapporti (vs POD) ---');
T_tempi = table(Metodo, Fase_Offline, Fase_Online, Tempo_Totale, Rapporto_Offline, Rapporto_Online, Rapporto, ...
    'VariableNames', {'Metodo', 'Offline_sec', 'Online_sec', 'Totale_sec', 'Ratio_Off_vs_POD', 'Ratio_On_vs_POD', 'Ratio_vs_POD'});
disp(T_tempi);

% --- Option 2: Create a dedicated figure with a Graphical UI Table ---
fig_tab = figure('Name', 'Tabella dei Tempi Computazionali', ...
                 'Position', [300, 300, 650, 120], ... 
                 'Color', 'w', ...
                 'MenuBar', 'none', ...
                 'ToolBar', 'none');

% Organize data matrix for uitable
dati_tabella = [Fase_Offline, Fase_Online, Tempo_Totale, Rapporto_Offline, Rapporto_Online];

% Create graphical table
uit = uitable(fig_tab, 'Data', dati_tabella, ...
              'RowName', Metodo, ...
              'ColumnName', {'Offline (s)', 'Online (s)', 'Totale (s)', 'Ratio Off (vs POD)', 'Ratio On (vs POD)'}, ...
              'ColumnFormat', {'shortEng', 'shortEng', 'shortEng', 'short', 'short'}, ...
              'Units', 'normalized', ...
              'Position', [0.02 0.05 0.96 0.9], ...
              'FontSize', 11);

%% --- DATA EXPORT TO .dat FILES ---
disp('--- Salvataggio dei risultati su file .dat ---');

% 1. Save Residuals (res.dat)
T_res = array2table([param_test, err_res], ...
    'VariableNames', {'p', 'Res_LU', 'Res_Leverage', 'Res_POD'});
writetable(T_res, 'res.dat', 'Delimiter', '\t');

% 2. Save Eigenvalue Errors (eigval.dat)
T_eigval = array2table([param_test, err_lam], ...
    'VariableNames', {'p', 'ErrLam_LU', 'ErrLam_Leverage', 'ErrLam_POD'});
writetable(T_eigval, 'eigval.dat', 'Delimiter', '\t');

% 3. Save Subspace Angles (eigvec.dat)
T_eigvec = array2table([param_test, err_ang], ...
    'VariableNames', {'p', 'ErrAng_LU', 'ErrAng_Leverage', 'ErrAng_POD'});
writetable(T_eigvec, 'eigvec.dat', 'Delimiter', '\t');

% 4. Save Eigenvalues for Plotting (plot.dat)
% Since 'neig' can vary, generate column names dynamically
varNames_plot = {'p'};
for j = 1:neig, varNames_plot{end+1} = sprintf('Ex_eig_%d', j); end
for j = 1:neig, varNames_plot{end+1} = sprintf('LU_eig_%d', j); end
for j = 1:neig, varNames_plot{end+1} = sprintf('Lev_eig_%d', j); end
for j = 1:neig, varNames_plot{end+1} = sprintf('POD_eig_%d', j); end

% Extract only the real part of the eigenvalues for plotting, matching the figure
vals_plot_matrix = [param_test, ...
                    real(vals_ex), ...
                    real(vals_approx(:,:,1)), ... % LU
                    real(vals_approx(:,:,2)), ... % Leverage
                    real(vals_approx(:,:,3))];    % POD

T_plot = array2table(vals_plot_matrix, 'VariableNames', varNames_plot);
writetable(T_plot, 'plot.dat', 'Delimiter', '\t');

% 5. Save Computational Times Table (times.dat)
% T_tempi is already generated in the previous section
writetable(T_tempi, 'times.dat', 'Delimiter', '\t');

disp('Salvataggio completato: res.dat, eigval.dat, eigvec.dat, plot.dat, times.dat.');
