function theory(t_out)
%THEORY  Experiments of Section 3 (Figures 1-3): snapshot matrix and error bounds.
%
%   theory(t_out) considers A(p) = M1 + 0.4 p M2, with M1, M2 the symmetric
%   parts of two 1000 x 1000 random matrices, p in [-1,1], and its rightmost
%   eigenvector x(p). For r = 4, ..., 30 snapshot points, equispaced or
%   Chebyshev (p_i = cos(i pi/(r-1)), i = 0..r-1), it computes
%     - the smallest singular value of the snapshot matrix X and the fit
%       C sqrt(r) rho^(-r) of its decay (Figure 1);
%     - the error ||(I - X X^+) x(t_out)||_2 at the new point t_out;
%     - fits of the error of the form e^{-c r} and e^{-c sqrt(r)};
%     - the bounds with the discrete measure, |prod_i (t_out - p_i)| / (rho/2)^r
%       (Theorem 3.3), and with the limit measure (Theorem 3.4), where
%       rho is estimated by fitting sigma_min(X_cheb) ~ C sqrt(r) rho^(-r)
%       (Theorem 2.2).
%   The constants in front of the fits and bounds are chosen to match the
%   error. The results are written in theory/1000_<t_out>.tex, read by
%   the paper: t_out = 1.008765 (outside [-1,1], Figure 2) and
%   t_out = 0.8765 (inside, Figures 1 and 3).

n = 1000;
r_values = (4:30)';

rng(42);
M1 = randn(n); M1 = (M1 + M1') / 2;
M2 = randn(n); M2 = 0.4 * (M2 + M2') / 2;
[V, D] = eig(M1);
[~, imax] = max(diag(D));
ref = V(:, imax);                          % to fix the sign of the eigenvectors
x_out = eigvec(t_out, M1, M2, ref);

nr = numel(r_values);
err_equi = zeros(nr, 1); err_cheb = zeros(nr, 1);
sing_equi = zeros(nr, 1); sing_cheb = zeros(nr, 1);
omega_equi = zeros(nr, 1); omega_cheb = zeros(nr, 1);
for j = 1:nr
    r = r_values(j);
    for type = ["equi", "cheb"]
        if type == "equi", pts = linspace(-1, 1, r)'; else, pts = cos((0:r-1)' * pi / (r-1)); end
        X = zeros(n, r);
        for i = 1:r
            X(:, i) = eigvec(pts(i), M1, M2, ref);
        end
        [Q, ~] = qr(X, 0);
        err = norm(x_out - Q * (Q' * x_out));   % ||(I - X X^+) x(t_out)||
        smin = min(svd(X));
        omega = abs(prod(t_out - pts));         % nodal polynomial at t_out
        if type == "equi"
            err_equi(j) = err; sing_equi(j) = smin; omega_equi(j) = omega;
        else
            err_cheb(j) = err; sing_cheb(j) = smin; omega_cheb(j) = omega;
        end
    end
end

% Decay of the singular values: fit sigma_min ~ C sqrt(r) rho^(-r) (Theorem 2.2)
c = polyfit(r_values, log(sing_cheb ./ sqrt(r_values)), 1);
rho = exp(-c(1));
rate_sing_cheb = sqrt(r_values) .* exp(polyval(c, r_values));
fprintf('Estimated rho = %.4f\n', rho);

% Fits of the error (only where it is above machine precision)
ok_cheb = find(err_cheb > 1e-14);
ok_equi = find(err_equi > 1e-14);
rate_cheb      = fit_exp(r_values, err_cheb, ok_cheb(max(1, end-10):end-6), r_values);
rate_equi_sqr  = fit_exp(sqrt(r_values), err_equi, ok_equi(1:min(10, end)), sqrt(r_values));
rate_equi_asym = fit_exp(r_values, err_equi, ok_equi(max(1, end-6):end), r_values);

% Bounds with the discrete measure (Theorem 3.3), scaled to the error
th_cheb_scaled_discrete = scale(omega_cheb ./ (rho/2).^r_values, err_cheb, 10);
th_equi_scaled_discrete = scale(omega_equi ./ (rho/2).^r_values, err_equi, 20);

% Asymptotic rates with the limit measure (Theorem 3.4), scaled to the error
if abs(t_out) <= 1
    base_cheb = 1 / rho;
    base_equi = 2 / (exp(1) * rho) * (1 + t_out)^((1 + t_out)/2) * (1 - t_out)^((1 - t_out)/2);
else
    t = abs(t_out);
    base_cheb = (t + sqrt(t^2 - 1)) / rho;
    base_equi = 2 / (exp(1) * rho) * (t + 1)^((t + 1)/2) / (t - 1)^((t - 1)/2);
end
th_cheb_scaled_apriori = scale(base_cheb.^r_values, err_cheb, 10);
th_equi_scaled_apriori = scale(base_equi.^r_values, err_equi, nr - 5);

T = table(err_equi, err_cheb, sing_equi, sing_cheb, rate_cheb, rate_equi_sqr, ...
          rate_equi_asym, th_cheb_scaled_discrete, th_equi_scaled_discrete, ...
          th_cheb_scaled_apriori, th_equi_scaled_apriori, rate_sing_cheb);
file = fullfile(fileparts(mfilename('fullpath')), sprintf('%d_%.10g.tex', n, t_out));
writetable(T, file, 'Delimiter', '\t', 'FileType', 'text');
fprintf('Written %s\n', file);
end


function x = eigvec(t, M1, M2, ref)
% Rightmost eigenvector of M1 + t M2, with the sign fixed by ref.
[V, D] = eig(M1 + t * M2);
[~, imax] = max(diag(D));
x = V(:, imax);
if x' * ref < 0, x = -x; end
end


function y = fit_exp(s, err, sel, s_eval)
% Least-squares fit log(err) ~ a + b s on the entries sel, evaluated at s_eval.
c = polyfit(s(sel), log(err(sel)), 1);
y = exp(polyval(c, s_eval));
end


function y = scale(bound, err, j)
% Multiply the bound by the constant that makes it equal to err at index j.
y = bound * (err(j) / bound(j));
end
