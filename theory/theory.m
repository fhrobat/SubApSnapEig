clear; clc; close all;

Nd = 1000;
rho_max = 3.1;

M_max = 30;      % Maximum number of snapshots (subspace dimension)
%t_out = 1.00008765;
%t_out = 0.08765;

% 1. Problem Setup and Matrix Generation
rng(42); 
M1 = randn(Nd); M1 = (M1 + M1') / 2;
M2 = randn(Nd); M2 = (M2 + M2') / 2;
M2 = 0.4 * M2;

[V, D] = eig(M1); [~, ind] = sort(diag(D), 'ComparisonMethod','real'); 
ref_vec = V(:, ind(end)); 

% Compute the target vector at the new point
x_true = get_eigenvector(t_out, M1, M2, ref_vec);

M_values = 4:M_max;
err_equi = zeros(length(M_values), 1);
err_cheb = zeros(length(M_values), 1);
sing_equi = zeros(length(M_values), 1);
sing_cheb = zeros(length(M_values), 1);

omega_equi = zeros(length(M_values), 1);
omega_cheb = zeros(length(M_values), 1);

for idx = 1:length(M_values)
    idx
    M = M_values(idx);
    
    % ---------------------------------------------------------
    % 1. Equispaced Nodal Subspace
    % ---------------------------------------------------------
    t_equi = linspace(-1, 1, M)';
    X_equi = zeros(Nd, M);
    for j = 1:M
        X_equi(:, j) = get_eigenvector(t_equi(j), M1, M2, ref_vec);
    end
    
    [Q_eq, ~] = qr(X_equi, 0);
    err_equi(idx) = norm(x_true - Q_eq * (Q_eq' * x_true));
    sing_equi(idx) = min(svd(X_equi));
    
    % Compute the discrete nodal polynomial value: exp(-M * U_M(t_out)) = |Prod(t_out - t_k)|
    omega_equi(idx) = abs(prod(t_out - t_equi));
    
    % ---------------------------------------------------------
    % 2. Chebyshev Nodal Subspace
    % ---------------------------------------------------------
    k = 1:M;
    t_cheb = cos((k - 1)*pi/ M)';
    X_cheb = zeros(Nd, M);
    for j = 1:M
        X_cheb(:, j) = get_eigenvector(t_cheb(j), M1, M2, ref_vec);
    end
    
    [Q_ch, ~] = qr(X_cheb, 0);
    err_cheb(idx) = norm(x_true - Q_ch * (Q_ch' * x_true));
    sing_cheb(idx) = min(svd(X_cheb));
    
    % Compute the discrete nodal polynomial value: exp(-M * U_M(t_out)) = |Prod(t_out - t_k)|
    omega_cheb(idx) = abs(prod(t_out - t_cheb));
end

% semilogy(svd(X_cheb, 'econ'));
% p_cheb_sing = polyfit((1:M_max)', log(svd(X_cheb, 'econ')), 1);
% rate_sing_cheb = exp(p_cheb_sing(2)) * exp(p_cheb_sing(1)).^M_values;
% exp(-p_cheb_sing(1))

valid_cheb = find(err_cheb > 1e-14);
p_cheb_sing = polyfit(M_values(valid_cheb)', log(sing_cheb(valid_cheb)), 1);
rate_sing_cheb = exp(p_cheb_sing(2)) * exp(p_cheb_sing(1)).^M_values;
% rho_max = -p_cheb_sing(1)

% ---------------------------------------------------------
% Convergence Rate Fitting
% ---------------------------------------------------------
valid_cheb = find(err_cheb > 1e-14);
valid_equi = find(err_equi > 1e-14);

% Chebyshev fit (Pure Exponential Decay O(r^M))
late_cheb = valid_cheb(max(1, end-10): (end-6));
p_cheb = polyfit(M_values(late_cheb)', log(err_cheb(late_cheb)), 1);
rate_cheb = exp(p_cheb(2)) * exp(p_cheb(1)).^M_values;

% Equispaced fit - Phase 1: Latency/Stagnation O(exp(-c*sqrt(M)))
early_equi = valid_equi(1:min(10, length(valid_equi)));
p_sqr = polyfit(sqrt(M_values(early_equi))', log(err_equi(early_equi)), 1);
rate_equi_sqr = exp(p_sqr(2)) * exp(p_sqr(1)).^sqrt(M_values);

% Equispaced fit - Phase 2: Asymptotic Exponential Decay O(r_2^M)
late_equi = valid_equi(max(1, end-6):end);
p_eq_asym = polyfit(M_values(late_equi)', log(err_equi(late_equi)), 1);
rate_equi_asym = exp(p_eq_asym(2)) * exp(p_eq_asym(1)).^M_values;

% ---------------------------------------------------------
% THEORETICAL CALCULATIONS 1: Nodal Polynomials (Discrete Bounds)
% ---------------------------------------------------------
denom_factor = (rho_max/2) .^ M_values';
full_th_rate_cheb = omega_cheb ./ denom_factor;
full_th_rate_equi = omega_equi ./ denom_factor;

% Scaling factors to align theoretical bounds with empirical error plots
th_cheb_scaled_discrete = full_th_rate_cheb * (err_cheb(10) / full_th_rate_cheb(10));
th_equi_scaled_discrete = full_th_rate_equi * (err_equi(20) / full_th_rate_equi(20));

% ---------------------------------------------------------
% THEORETICAL CALCULATIONS 2: A Priori Bounds (Continuous Theorem)
% ---------------------------------------------------------
if t_out >= -1 && t_out <= 1
    % --- INTERPOLATION Formulas ---
    
    base_cheb = 1 / rho_max;
    
    % Equispaced: Exact spatially varying formula (based on custom theorem)
    % U_unif = 1 - 0.5*[(1+x)log(1+x) + (1-x)log(1-x)]
    term_1 = (t_out + 1)^((t_out + 1)/2);
    term_2 = (1 - t_out)^((1 - t_out)/2);
    base_equi = (2 / (exp(1) * rho_max)) * term_1 * term_2;
    
else
    % --- EXTRAPOLATION Formulas ---
    
    base_cheb = abs(t_out + sqrt(t_out^2 - 1)) / rho_max;
    
    t_abs = abs(t_out); % Extended via symmetry for t_out < -1
    term_num = (t_abs + 1)^((t_abs + 1)/2);
    term_den = (t_abs - 1)^((t_abs - 1)/2);
    base_equi = (2 / (exp(1) * rho_max)) * (term_num / term_den);
end

apriori_rate_cheb = base_cheb .^ M_values';
apriori_rate_equi = base_equi .^ M_values';
% Scaling factors to align theoretical bounds with empirical error plots
th_cheb_scaled_apriori = apriori_rate_cheb * (err_cheb(10) / apriori_rate_cheb(10));
th_equi_scaled_apriori = apriori_rate_equi * (err_equi(max(1, end-5)) / apriori_rate_equi(max(1, end-5))); % Aligns to the tail


% ---------------------------------------------------------
% Plot 1: Subspace Error vs Discrete Nodal Polynomial
% ---------------------------------------------------------
figure(2)
semilogy(M_values, err_cheb, 's-b', 'LineWidth', 2, 'MarkerFaceColor', 'b', 'DisplayName', 'Errore X_{cheb}'); hold on;
semilogy(M_values, err_equi, 'o-r', 'LineWidth', 2, 'MarkerFaceColor', 'r', 'DisplayName', 'Errore X_{equi}');

semilogy(M_values, rate_cheb, '--k', 'LineWidth', 1.5, 'DisplayName', 'Fit Cheb O(e^{-c_1 M})');
semilogy(M_values, rate_equi_sqr, '--m', 'LineWidth', 2, 'DisplayName', 'Fit Equi Latenza O(e^{-c_2 \surd{M}})');
semilogy(M_values, rate_equi_asym, '-.k', 'LineWidth', 1.5, 'DisplayName', 'Fit Equi Asintotico O(e^{-c_3 M})');

semilogy(M_values, th_cheb_scaled_discrete, ':c', 'LineWidth', 2.5, 'DisplayName', 'Bound Discreto \Omega_M Cheb');
semilogy(M_values, th_equi_scaled_discrete, ':y', 'LineWidth', 2.5, 'DisplayName', 'Bound Discreto \Omega_M Equi');

title(sprintf('Verifica Discreta: Errore vs Polinomio Nodale (t_{out} = %.4f)', t_out));
xlabel('Dimensione del Sottospazio (M)');
ylabel('Errore ||(I - XX^\dagger) x_{true}||_2');
ylim([1e-16, 10]);
grid on; 
legend('Location', 'southwest', 'FontSize', 10);

% ---------------------------------------------------------
% Plot 2 (NEW): Subspace Error vs Continuous A Priori Bounds
% ---------------------------------------------------------
figure(3)
semilogy(M_values, err_cheb, 's-b', 'LineWidth', 2, 'MarkerFaceColor', 'b', 'DisplayName', 'Errore X_{cheb}'); hold on;
semilogy(M_values, err_equi, 'o-r', 'LineWidth', 2, 'MarkerFaceColor', 'r', 'DisplayName', 'Errore X_{equi}');

semilogy(M_values, th_cheb_scaled_apriori, '--c', 'LineWidth', 2.5, 'DisplayName', 'Teorema A Priori Cheb');
semilogy(M_values, th_equi_scaled_apriori, '--y', 'LineWidth', 2.5, 'DisplayName', 'Teorema A Priori Equi');

title(sprintf('Verifica Continua: Errore vs Limiti A Priori del Teorema (t_{out} = %.4f)', t_out));
xlabel('Dimensione del Sottospazio (M)');
ylabel('Errore ||(I - XX^\dagger) x_{true}||_2');
ylim([1e-16, 10]);
grid on; 
legend('Location', 'southwest', 'FontSize', 10);

% ---------------------------------------------------------
% Plot 3: Conditioning / Minimum Singular Values Decay
% ---------------------------------------------------------
figure(4)
semilogy(M_values, sing_cheb, 's-b', 'LineWidth', 1.5, 'DisplayName', '\sigma_{min} X_{cheb}'); hold on;
semilogy(M_values, sing_equi, 'o-r', 'LineWidth', 1.5, 'DisplayName', '\sigma_{min} X_{equi}');
%semilogy(M_values, rate_sing_cheb, '--k', 'LineWidth', 1.5, 'DisplayName', 'rate \sigma_{min} X_{cheb}');
semilogy(M_values, 2*rho_max/(1- rho_max^(-1)) .* sqrt(M_values).*rho_max.^(-M_values), '--k', 'LineWidth', 1.5, 'DisplayName', 'rate \sigma_{min} X_{cheb}');
title('Decadimento del Valore Singolare Minimo (Condizionamento Base)');
xlabel('Numero di Snapshot (M)');
ylabel('\sigma_{min}');
grid on; legend('Location', 'southwest');

% --- Auxiliary Function ---
function eigvec = get_eigenvector(t, M1, M2, ref_vec)
    [V, D] = eig(M1 + t * M2);
    [~, ind] = sort(diag(D), 'ComparisonMethod','real'); 
    eigvec = V(:, ind(end));
    
    % Enforce sign consistency with the reference vector
    if dot(eigvec, ref_vec) < 0, eigvec = -eigvec; end
end