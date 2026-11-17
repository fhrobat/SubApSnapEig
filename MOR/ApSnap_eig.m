function [lam, Q, Y, t_off, t_on] = ApSnap_eig(prob, opts)
%APSNAP_EIG  Baseline MOR: snapshots and full Rayleigh-Ritz (no subsampling).
%
%   [lam, Q, Y, t_off, t_on] = ApSnap_eig(prob, opts) is the method called MOR
%   in the paper, the extension of Guido, Kressner and Ricci (2024) to
%   eigenvalue problems. The offline phase is the same as in SubApSnap_eig
%   (snapshot matrix X and orthonormal basis Q), but in the online phase the
%   projected matrix is computed with the whole matrix:
%       M = Q' * A(p) * Q,
%   and its eigenpairs (lambda, y) give the approximations (lambda, Q*y).
%   The snapshot matrix is updated every opts.update_freq parameters, as in
%   SubApSnap_eig.
%
%   prob, opts, outputs: as in SubApSnap_eig (opts.sampling and opts.tol
%   are not used).

if ~isfield(opts, 'snap'),        opts.snap = floor(linspace(1, numel(prob.param), opts.k)); end
if ~isfield(opts, 'buffer'),      opts.buffer = 0; end
if ~isfield(opts, 'update_freq'), opts.update_freq = Inf; end
if ~isfield(opts, 'tol_upd'),     opts.tol_upd = 1e-8; end
if ~isfield(prob, 'Amult'),       prob.Amult = @(p, X) prob.A(p) * X; end

m = prob.neig;
N = numel(prob.param);
lam = nan(m, N);
C = cell(N, 1);                         % coefficients y in the current basis
is_snap = false(N, 1);
is_snap(opts.snap) = true;

%% Offline phase: snapshot matrix and its orthonormal basis
tic
nb = m + opts.buffer;
X = zeros(prob.n, nb * numel(opts.snap));
for j = 1:numel(opts.snap)
    i = opts.snap(j);
    [Xj, lj] = full_eigs(prob, prob.param(i), nb, 'MaxIterations', 2000);
    X(:, (j-1)*nb+1 : j*nb) = Xj;
    lam(:, i) = lj(1:m);
end
Q = orth(X);
for j = 1:numel(opts.snap)              % exact eigenvectors at the snapshots
    C{opts.snap(j)} = Q' * X(:, (j-1)*nb+1 : (j-1)*nb+m);
end
t_off = toc;

%% Online phase: Rayleigh-Ritz with the full matrix
tic
for i = 1:N
    if is_snap(i), continue; end
    p = prob.param(i);

    M = Q' * prob.Amult(p, Q);
    if prob.sym, M = (M + M') / 2; end
    [Y, D] = eig(M);
    [li, ord] = sort_eig(diag(D), prob.sigma);
    Y = Y(:, ord(1:m));
    lam(:, i) = li(1:m);

    if mod(i, opts.update_freq) == 0
        [Xi, li] = full_eigs(prob, p, nb, 'StartVector', Q*Y(:, 1), ...
                             'Tolerance', opts.tol_upd);
        lam(:, i) = li(1:m);
        Q = add_to_basis(Q, Xi);
        C{i} = Q' * Xi(:, 1:m);
    else
        C{i} = Y;
    end
end
t_on = toc;

% Coefficients in the final basis (zero padding, see SubApSnap_eig)
Y = zeros(size(Q, 2), m, N);
for i = 1:N
    Y(1:size(C{i}, 1), :, i) = C{i};
end
end
