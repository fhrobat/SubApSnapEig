function [sig, Ql, Yl, Qr, Yr, t_off, t_on] = ApSnap_SVD(prob, opts)
%APSNAP_SVD  Baseline MOR for sequences of SVD problems (no subsampling).
%
%   [sig, Ql, Yl, Qr, Yr, t_off, t_on] = ApSnap_SVD(prob, opts) has the same offline
%   phase as SubApSnap_SVD (bases Q_l and Q_r of the left and right
%   snapshot matrices), but in the online phase it uses the whole matrix:
%       M = Q_l' * A(p) * Q_r,   M = Y_l Sigma Y_r',
%   and returns (sigma, Q_l y_l, Q_r y_r).
%
%   prob, opts, outputs: as in SubApSnap_SVD (opts.sampling is not used).

N = numel(prob.param);
if ~isfield(opts, 'snap'),        opts.snap = floor(linspace(1, N, opts.k)); end
if ~isfield(opts, 'buffer'),      opts.buffer = 0; end
if ~isfield(opts, 'update_freq'), opts.update_freq = Inf; end
if ~isfield(opts, 'tol_upd'),     opts.tol_upd = 1e-8; end

m = prob.nsv;
sig = nan(m, N);
Cl = cell(N, 1); Cr = cell(N, 1);       % coefficients in the current bases
is_snap = false(N, 1);
is_snap(opts.snap) = true;

%% Offline phase
tic
nb = m + opts.buffer;
Xl = zeros(prob.m, nb * numel(opts.snap));
Xr = zeros(prob.n, nb * numel(opts.snap));
for j = 1:numel(opts.snap)
    i = opts.snap(j);
    [Ul, S, Vr] = svds(prob.A(prob.param(i)), nb, 'largest', 'MaxIterations', 2000);
    Xl(:, (j-1)*nb+1 : j*nb) = Ul;
    Xr(:, (j-1)*nb+1 : j*nb) = Vr;
    sig(:, i) = diag(S(1:m, 1:m));
end
Ql = orth(Xl);
Qr = orth(Xr);
for j = 1:numel(opts.snap)              % exact singular vectors at the snapshots
    Cl{opts.snap(j)} = Ql' * Xl(:, (j-1)*nb+1 : (j-1)*nb+m);
    Cr{opts.snap(j)} = Qr' * Xr(:, (j-1)*nb+1 : (j-1)*nb+m);
end
t_off = toc;

%% Online phase
tic
for i = 1:N
    if is_snap(i), continue; end
    p = prob.param(i);
    Ap = prob.A(p);

    [Yl, S, Yr] = svd(Ql' * (Ap * Qr));
    sig(:, i) = diag(S(1:m, 1:m));

    if mod(i, opts.update_freq) == 0
        [Ul, S, Vr] = svds(Ap, nb, 'largest', 'RightStartVector', ...
                           Qr*Yr(:, 1), 'Tolerance', opts.tol_upd);
        sig(:, i) = diag(S(1:m, 1:m));
        Ql = add_to_basis(Ql, Ul);
        Qr = add_to_basis(Qr, Vr);
        Cl{i} = Ql' * Ul(:, 1:m);
        Cr{i} = Qr' * Vr(:, 1:m);
    else
        Cl{i} = Yl(:, 1:m);
        Cr{i} = Yr(:, 1:m);
    end
end
t_on = toc;

% Coefficients in the final bases (zero padding, see SubApSnap_eig)
Yl = zeros(size(Ql, 2), m, N);
Yr = zeros(size(Qr, 2), m, N);
for i = 1:N
    Yl(1:size(Cl{i}, 1), :, i) = Cl{i};
    Yr(1:size(Cr{i}, 1), :, i) = Cr{i};
end
end
