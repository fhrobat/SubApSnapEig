warning off;
clear

%% Transformer (BERT) Setup 
nsv = 1;                                % Number of singular values to compute
step_iniziale = 1688;                   % Initial step index
param = (1:563)';                       % Parameter space grid
head_idx = 1;                           % Attention head index

Q_RAM = cell(length(param), 1);         % Preallocate cell array for Queries
K_RAM = cell(length(param), 1);         % Preallocate cell array for Keys

for i = 1:length(param)
    i
    step_corrente = step_iniziale + i;
    filename = sprintf('python/matrici_matlab_dirette_rope/step_%d_ROPE.mat', step_corrente);
    dati = load(filename, 'Q_rope', 'K_rope');
    Q_RAM{i} = dati.Q_rope;
    K_RAM{i} = dati.K_rope;
end

sqrt_d = sqrt(64);                      % Scaling factor for attention mechanism

% Matrix handles for attention score generation
Afull = @(p) apply_softmax_full(Q_RAM{p}, K_RAM{p}, sqrt_d);
Arows = @(p, r_idx, M) apply_softmax_rows(Q_RAM{p}, K_RAM{p}, r_idx, sqrt_d) * M;

function A = apply_softmax_full(Q_rope, K_rope, sd)
    W = (Q_rope * K_rope') / sd;
    W = W - max(W, [], 2);              % Subtract max for numerical stability
    expW = exp(W);
    A = expW ./ sum(expW, 2);           % Softmax normalization
end

function A_rows = apply_softmax_rows(Q_rope, K_rope, r_idx, sd)
    Q_sub = Q_rope(r_idx, :);
    W_sub = (Q_sub * K_rope') / sd;
    W_sub = W_sub - max(W_sub, [], 2);  % Subtract max for numerical stability
    expW = exp(W_sub);
    A_rows = expW ./ sum(expW, 2);      % Row-sampled softmax normalization
end
%%

[m_rows, n_cols] = size(Afull(param(1)));
k = 50;                                 % Number of snapshot points
s = 0;
tol = Inf;
update_freq = Inf;

%% Settings and Parameters Configuration
opts = [];
opts.param = param;
opts.nsv = nsv; 
opts.k = k;
opts.s = s;
opts.update_freq = update_freq;
opts.tol = tol;
opts.svd_sigma = 'largest';
opts.SnapPoints = param(1:k);

% Automatically distribute snapshot points if not explicitly provided
if ~isfield(opts, 'SnapPoints') || isempty(opts.SnapPoints)
    points = floor(linspace(1, length(opts.param), opts.k));
    opts.SnapPoints = opts.param(points);
end

% Define the test parameter grid
param_test = param(round(linspace(1, length(param), 100)));
is_snapshot = ismembertol(param_test, opts.SnapPoints, 1e-10);
param_test = param_test(~is_snapshot);
n_test = length(param_test);

%% Preallocation for Errors and Plotting Data
err_res = zeros(n_test, 3);
err_sval = zeros(n_test, 3);
err_ang = zeros(n_test, 3); 
vals_ex = zeros(n_test, nsv);
vals_approx = zeros(n_test, nsv, 3);

warning off;

%% PHASE 0: Exact Solutions Calculation 
disp('--- PHASE 0: Exact Solutions ---');
U_ex_all = cell(n_test, 1);
V_ex_all = cell(n_test, 1);
S_ex_all = cell(n_test, 1);
norm_A_all = zeros(n_test, 1);
msg = '';

for i = 1:n_test
    p = param_test(i);
    A_p = Afull(p);
    norm_A_all(i) = norm(A_p, 1); 
    
    [U_ex, S_ex_mat, V_ex] = svds(A_p, nsv, opts.svd_sigma); 
    S_ex = diag(S_ex_mat);
    
    [~, idx_ex] = sort(S_ex, 'descend');
    S_ex_all{i} = S_ex(idx_ex);
    U_ex_all{i} = U_ex(:, idx_ex);
    V_ex_all{i} = V_ex(:, idx_ex);
    vals_ex(i, :) = S_ex_all{i}.'; 
    
    fprintf(repmat('\b', 1, length(msg)));
    msg = sprintf('Exact Solutions: [%3.1f%%]', (i / n_test) * 100);
    fprintf('%s', msg);
end
fprintf('\n');
clear A_p U_ex S_ex_mat S_ex V_ex; % Free temporary full matrices
warning on;

%% PHASE 1: LU Method
disp('--- PHASE 1: LU Method ---');
opts.sampling = 'lu';
[U_lu, S_lu, V_lu, t_on_lu, t_off_lu] = SubApSnap_SVD(Afull, Arows, m_rows, n_cols, opts);
time_lu = t_on_lu + t_off_lu;
msg = '';

for i = 1:n_test
    p = param_test(i);
    A_p = Afull(p); 
    
    U_m = U_lu(p); V_m = V_lu(p); S_m = S_lu(p);
    if size(S_m, 1) == size(S_m, 2), S_m = diag(S_m); end
    
    S_ex = S_ex_all{i}; U_ex = U_ex_all{i}; V_ex = V_ex_all{i};
    perm_idx = perms(1:nsv); best_cost = Inf; best_perm = 1:nsv;
    for pp = 1:size(perm_idx,1)
        cost = norm(S_m(perm_idx(pp,:)) - S_ex);
        if cost < best_cost, best_cost = cost; best_perm = perm_idx(pp,:); end
    end
    S_m = S_m(best_perm); U_m = U_m(:, best_perm); V_m = V_m(:, best_perm);
    
    err_res(i, 1) = norm(A_p * V_m - U_m * diag(S_m), 'fro') / ((norm_A_all(i) + max(abs(S_m), [], 'omitnan')) * norm(V_m, 'fro'));
    err_sval(i, 1) = mean(abs(S_m - S_ex) ./ abs(S_ex), 'omitnan');
    err_ang(i, 1) = max(subspace(U_m, U_ex), subspace(V_m, V_ex));
    vals_approx(i, :, 1) = S_m.';
    
    fprintf(repmat('\b', 1, length(msg))); msg = sprintf('LU Evaluation: [%3.1f%%]', (i / n_test) * 100); fprintf('%s', msg);
end
fprintf('\n');
clear U_lu S_lu V_lu U_m V_m S_m A_p; % Memory release for LU variant

%% PHASE 2: Leverage Method
disp('--- PHASE 2: Leverage Method ---');
opts.sampling = 'leverage';
[U_lev, S_lev, V_lev, t_on_lev, t_off_lev] = SubApSnap_SVD(Afull, Arows, m_rows, n_cols, opts);
time_lev = t_on_lev + t_off_lev;
msg = '';

for i = 1:n_test
    p = param_test(i);
    A_p = Afull(p);
    
    U_m = U_lev(p); V_m = V_lev(p); S_m = S_lev(p);
    if size(S_m, 1) == size(S_m, 2), S_m = diag(S_m); end
    
    S_ex = S_ex_all{i}; U_ex = U_ex_all{i}; V_ex = V_ex_all{i};
    perm_idx = perms(1:nsv); best_cost = Inf; best_perm = 1:nsv;
    for pp = 1:size(perm_idx,1)
        cost = norm(S_m(perm_idx(pp,:)) - S_ex);
        if cost < best_cost, best_cost = cost; best_perm = perm_idx(pp,:); end
    end
    S_m = S_m(best_perm); U_m = U_m(:, best_perm); V_m = V_m(:, best_perm);
    
    err_res(i, 2) = norm(A_p * V_m - U_m * diag(S_m), 'fro') / ((norm_A_all(i) + max(abs(S_m), [], 'omitnan')) * norm(V_m, 'fro'));
    err_sval(i, 2) = mean(abs(S_m - S_ex) ./ abs(S_ex), 'omitnan');
    err_ang(i, 2) = max(subspace(U_m, U_ex), subspace(V_m, V_ex));
    vals_approx(i, :, 2) = S_m.';
    
    fprintf(repmat('\b', 1, length(msg))); msg = sprintf('Leverage Evaluation: [%3.1f%%]', (i / n_test) * 100); fprintf('%s', msg);
end
fprintf('\n');
clear U_lev S_lev V_lev U_m V_m S_m A_p; % Memory release for Leverage variant

%% PHASE 3: POD Method
disp('--- PHASE 3: POD Method ---');
[U_pod, S_pod, V_pod, t_on_pod, t_off_pod] = ApSnap_SVD(Afull, m_rows, n_cols, opts);
time_pod = t_on_pod + t_off_pod;
msg = '';

for i = 1:n_test
    p = param_test(i);
    A_p = Afull(p);
    
    U_m = U_pod(p); V_m = V_pod(p); S_m = S_pod(p);
    if size(S_m, 1) == size(S_m, 2), S_m = diag(S_m); end
    
    S_ex = S_ex_all{i}; U_ex = U_ex_all{i}; V_ex = V_ex_all{i};
    perm_idx = perms(1:nsv); best_cost = Inf; best_perm = 1:nsv;
    for pp = 1:size(perm_idx,1)
        cost = norm(S_m(perm_idx(pp,:)) - S_ex);
        if cost < best_cost, best_cost = cost; best_perm = perm_idx(pp,:); end
    end
    S_m = S_m(best_perm); U_m = U_m(:, best_perm); V_m = V_m(:, best_perm);
    
    err_res(i, 3) = norm(A_p * V_m - U_m * diag(S_m), 'fro') / ((norm_A_all(i) + max(abs(S_m), [], 'omitnan')) * norm(V_m, 'fro'));
    err_sval(i, 3) = mean(abs(S_m - S_ex) ./ abs(S_ex), 'omitnan');
    err_ang(i, 3) = max(subspace(U_m, U_ex), subspace(V_m, V_ex));
    vals_approx(i, :, 3) = S_m.';
    
    fprintf(repmat('\b', 1, length(msg))); msg = sprintf('POD Evaluation: [%3.1f%%]', (i / n_test) * 100); fprintf('%s', msg);
end
fprintf('\nComputation completed successfully!\n');

%% Final Global Cleanup
clear U_pod S_pod V_pod U_m V_m S_m A_p U_ex_all V_ex_all S_ex_all;

%% Plot Results (NaNs Filtered)
figure('Name', 'Subspace Error and Singular Value Analysis', ...
       'Position', [100, 100, 1200, 850], ...
       'Color', 'w');
t = tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
colors = [0.85 0.33 0.10;   
          0.00 0.45 0.74;   
          0.47 0.67 0.19];  
labels = {'LU', 'Leverage', 'POD'};
markers = {'o', 's', '^'};
lineStyles = {'-', '-', '-'};
m_bg = 2;          
fg_methods = [1 3]; 

%% 1. Residual Plot
ax = nexttile;
set(ax, 'YScale', 'log');
hold on;
m = m_bg;
valid_idx = ~isnan(err_res(:, m));
idx_plot = find(valid_idx);
plot(param_test(valid_idx), err_res(valid_idx, m), ...
    'Color', colors(m,:), 'LineStyle', lineStyles{m}, 'LineWidth', 1.0, ...
    'Marker', markers{m}, 'MarkerSize', 3.5, ...
    'MarkerIndices', idx_plot(1:max(1,floor(numel(idx_plot)/12)):numel(idx_plot)), ...
    'DisplayName', labels{m});
for m = fg_methods
    valid_idx = ~isnan(err_res(:, m));
    idx_plot = find(valid_idx);
    plot(param_test(valid_idx), err_res(valid_idx, m), ...
        'Color', colors(m,:), 'LineStyle', lineStyles{m}, 'LineWidth', 1.8, ...
        'Marker', markers{m}, 'MarkerSize', 5, ...
        'MarkerIndices', idx_plot(1:max(1,floor(numel(idx_plot)/12)):numel(idx_plot)), ...
        'DisplayName', labels{m});
end
title('Relative Residual Error (||AV - U\Sigma||)', 'FontWeight', 'bold');
xlabel('Parameter p'); ylabel('Residual');
legend('Location', 'best'); grid on; grid minor; box on;

%% 2. Singular Value Error Plot
ax = nexttile;
set(ax, 'YScale', 'log');
hold on;
m = m_bg;
valid_idx = ~isnan(err_sval(:, m));
idx_plot = find(valid_idx);
plot(param_test(valid_idx), err_sval(valid_idx, m), ...
    'Color', colors(m,:), 'LineStyle', lineStyles{m}, 'LineWidth', 1.0, ...
    'Marker', markers{m}, 'MarkerSize', 3.5, ...
    'MarkerIndices', idx_plot(1:max(1,floor(numel(idx_plot)/12)):numel(idx_plot)), ...
    'DisplayName', labels{m});
for m = fg_methods
    valid_idx = ~isnan(err_sval(:, m));
    idx_plot = find(valid_idx);
    plot(param_test(valid_idx), err_sval(valid_idx, m), ...
        'Color', colors(m,:), 'LineStyle', lineStyles{m}, 'LineWidth', 1.8, ...
        'Marker', markers{m}, 'MarkerSize', 5, ...
        'MarkerIndices', idx_plot(1:max(1,floor(numel(idx_plot)/12)):numel(idx_plot)), ...
        'DisplayName', labels{m});
end
title('Relative Singular Value Error', 'FontWeight', 'bold');
xlabel('Parameter p'); ylabel('Error');
legend('Location', 'best'); grid on; grid minor; box on;

%% 3. Subspace Angle Plot
ax = nexttile;
set(ax, 'YScale', 'log');
hold on;
m = m_bg;
valid_idx = ~isnan(err_ang(:, m));
idx_plot = find(valid_idx);
plot(param_test(valid_idx), err_ang(valid_idx, m), ...
    'Color', colors(m,:), 'LineStyle', lineStyles{m}, 'LineWidth', 1.0, ...
    'Marker', markers{m}, 'MarkerSize', 3.5, ...
    'MarkerIndices', idx_plot(1:max(1,floor(numel(idx_plot)/12)):numel(idx_plot)), ...
    'DisplayName', labels{m});
for m = fg_methods
    valid_idx = ~isnan(err_ang(:, m));
    idx_plot = find(valid_idx);
    plot(param_test(valid_idx), err_ang(valid_idx, m), ...
        'Color', colors(m,:), 'LineStyle', lineStyles{m}, 'LineWidth', 1.8, ...
        'Marker', markers{m}, 'MarkerSize', 5, ...
        'MarkerIndices', idx_plot(1:max(1,floor(numel(idx_plot)/12)):numel(idx_plot)), ...
        'DisplayName', labels{m});
end
title('Max Subspace Similarity (U and V)', 'FontWeight', 'bold');
xlabel('Parameter p'); ylabel('Angle');
legend('Location', 'best'); grid on; grid minor; box on;

%% 4. Singular Value Plot
ax = nexttile;
hold on;
for j = 1:nsv
    valid_idx = ~isnan(vals_approx(:, j, m_bg));
    x = param_test(valid_idx);
    y = real(vals_approx(valid_idx, j, m_bg));
    if j == 1
        plot(x, y, 'Color', colors(m_bg,:), 'LineStyle', 'none', ...
            'Marker', markers{m_bg}, 'MarkerSize', 2.5, 'DisplayName', labels{m_bg});
    else
        plot(x, y, 'Color', colors(m_bg,:), 'LineStyle', 'none', ...
            'Marker', markers{m_bg}, 'MarkerSize', 2.5, 'HandleVisibility', 'off');
    end
end
for m = fg_methods
    for j = 1:nsv
        valid_idx = ~isnan(vals_approx(:, j, m));
        x = param_test(valid_idx);
        y = real(vals_approx(valid_idx, j, m));
        if j == 1
            plot(x, y, 'Color', colors(m,:), 'LineStyle', 'none', ...
                'Marker', markers{m}, 'MarkerSize', 3.5, 'DisplayName', labels{m});
        else
            plot(x, y, 'Color', colors(m,:), 'LineStyle', 'none', ...
                'Marker', markers{m}, 'MarkerSize', 3.5, 'HandleVisibility', 'off');
        end
    end
end
for j = 1:nsv
    valid_idx = ~isnan(vals_ex(:, j));
    x = param_test(valid_idx);
    y = real(vals_ex(valid_idx, j));
    if j == 1
        plot(x, y, 'k.', 'MarkerSize', 10, 'DisplayName', 'Exact');
    else
        plot(x, y, 'k.', 'MarkerSize', 10, 'HandleVisibility', 'off');
    end
end
title('Singular Values', 'FontWeight', 'bold');
xlabel('Parameter p'); ylabel('Magnitude');
legend('Location', 'eastoutside'); grid on; grid minor; box on;

%% Overall Title
title(t, 'Comparison of Residual, Singular Value, and Subspace Errors', ...
    'FontSize', 14, 'FontWeight', 'bold');

%% 5. Computational Times Table and Ratios (vs POD)
Metodo = {'LU'; 'Leverage'; 'POD'};
Fase_Offline = [t_off_lu; t_off_lev; t_off_pod];
Fase_Online  = [t_on_lu; t_on_lev; t_on_pod];
Tempo_Totale = Fase_Offline + Fase_Online;

% Calculate computing time ratios relative to POD
Rapporto_Offline = Fase_Offline ./ t_off_pod;
Rapporto_Online  = Fase_Online ./ t_on_pod;
Rapporto = Tempo_Totale ./ time_pod;

disp('--- Confronto Tempi di Esecuzione e Rapporti (vs POD) ---');
T_tempi = table(Metodo, Fase_Offline, Fase_Online, Tempo_Totale, Rapporto_Offline, Rapporto_Online, Rapporto, ...
    'VariableNames', {'Metodo', 'Offline_sec', 'Online_sec', 'Totale_sec', 'Ratio_Off_vs_POD', 'Ratio_On_vs_POD', 'Ratio_vs_POD'});
disp(T_tempi);

% Graphical Table Setup
fig_tab = figure('Name', 'Tabella dei Tempi Computazionali', ...
                 'Position', [300, 300, 650, 120], ... 
                 'Color', 'w', 'MenuBar', 'none', 'ToolBar', 'none');

dati_tabella = [Fase_Offline, Fase_Online, Tempo_Totale, Rapporto_Offline, Rapporto_Online];

uit = uitable(fig_tab, 'Data', dati_tabella, ...
              'RowName', Metodo, ...
              'ColumnName', {'Offline (s)', 'Online (s)', 'Totale (s)', 'Ratio Off (vs POD)', 'Ratio On (vs POD)'}, ...
              'ColumnFormat', {'shortEng', 'shortEng', 'shortEng', 'short', 'short'}, ...
              'Units', 'normalized', ...
              'Position', [0.02 0.05 0.96 0.9], 'FontSize', 11);


%% --- DATA EXPORT TO .dat FILES ---
disp('--- Salvataggio dei risultati su file .dat ---');

% 1. Save Residuals (res.dat)
T_res = array2table([param_test, err_res], ...
    'VariableNames', {'p', 'Res_LU', 'Res_Leverage', 'Res_POD'});
writetable(T_res, 'res.dat', 'Delimiter', '\t');

% 2. Save Singular Value Errors to original file (eigval.dat)
T_eigval = array2table([param_test, err_sval], ...
    'VariableNames', {'p', 'ErrLam_LU', 'ErrLam_Leverage', 'ErrLam_POD'});
writetable(T_eigval, 'eigval.dat', 'Delimiter', '\t');

% 3. Save Subspace Angles to original file (eigvec.dat)
T_eigvec = array2table([param_test, err_ang], ...
    'VariableNames', {'p', 'ErrAng_LU', 'ErrAng_Leverage', 'ErrAng_POD'});
writetable(T_eigvec, 'eigvec.dat', 'Delimiter', '\t');

% 4. Save Values for Plotting (plot.dat)
% Dynamically generate column names using 'nsv'
varNames_plot = {'p'};
for j = 1:nsv, varNames_plot{end+1} = sprintf('Ex_eig_%d', j); end
for j = 1:nsv, varNames_plot{end+1} = sprintf('LU_eig_%d', j); end
for j = 1:nsv, varNames_plot{end+1} = sprintf('Lev_eig_%d', j); end
for j = 1:nsv, varNames_plot{end+1} = sprintf('POD_eig_%d', j); end

% Extract only the real part of the singular values for plotting
vals_plot_matrix = [param_test, ...
                    real(vals_ex), ...
                    real(vals_approx(:,:,1)), ... % LU
                    real(vals_approx(:,:,2)), ... % Leverage
                    real(vals_approx(:,:,3))];    % POD

T_plot = array2table(vals_plot_matrix, 'VariableNames', varNames_plot);
writetable(T_plot, 'plot.dat', 'Delimiter', '\t');

% 5. Save Computational Times Table (times.dat)
writetable(T_tempi, 'times.dat', 'Delimiter', '\t');

disp('Salvataggio completato: res.dat, eigval.dat, eigvec.dat, plot.dat, times.dat.');