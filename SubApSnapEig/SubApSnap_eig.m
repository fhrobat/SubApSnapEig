function [lam, Q, Y, t_off, t_on] = SubApSnap_eig(prob, opts)
%SUBAPSNAP_EIG  Sequence of eigenvalue problems with snapshots and row subsampling.
%
%   [lam, Q, Y, t_off, t_on] = SubApSnap_eig(prob, opts) computes m eigenpairs
%   of A(p) for every p in prob.param (Algorithm 1 of the paper).
%
%   Offline phase
%     1. Compute m (+ buffer) eigenpairs of A(p) at the snapshot points with
%        eigs and collect the eigenvectors in the snapshot matrix X.
%     2. Q = orthonormal basis of span(X).
%     3. Select rows S of Q (select_rows: LU or leverage scores).
%   Online phase, for each new parameter p
%     4. Solve min_M ||S(A(p)Q - QM)||_F, i.e. M = (SQ) \ (S A(p) Q). Only
%        the rows S of A(p) are needed (prob.Arows), and SQ is factorized
%        once in the offline phase.
%     5. Compute the m eigenpairs (lambda, y) of the small matrix M; the
%        approximate eigenvector is Q*y.
%     6. Every opts.update_freq parameters, or when the estimated residual
%        exceeds opts.tol, refine the eigenvectors with eigs (started from
%        Q*y), add them to the basis and select the rows again.
%
%   prob: problem definition (see the files in problems/)
%     n       size of A(p)
%     param   the N parameter values, in the order in which they are processed
%     neig    number m of wanted eigenpairs
%     sigma   'largestabs' or 'smallestabs' (as in eigs)
%     sym     true if A(p) is real symmetric (eigenvalues are then taken real)
%     A       @(p) A(p) as a sparse matrix   [optional, needed for 'smallestabs']
%     Amult   @(p,X) A(p)*X                   [optional, default A(p)*X]
%     Arows   @(p,idx,X) A(p)(idx,:)*X
%
%   opts:
%     k           number of snapshot points, equispaced in the index of prob.param
%     snap        indices into prob.param of the snapshot points (overrides k)
%     sampling    'lu' or 'leverage' (see select_rows)
%     buffer      extra eigenpairs per snapshot point                 (default 0)
%     update_freq update X and S every update_freq parameters         (default Inf)
%     tol         update X and S if the estimated residual is > tol   (default Inf)
%     tol_upd     tolerance of eigs when refining the eigenvectors    (default 1e-8)
%
%   Output:
%     lam    m x N matrix of eigenvalues, lam(:,i) for prob.param(i)
%     Q, Y   the eigenvectors in factored form: the eigenvectors for
%            prob.param(i) are Q*Y(:,:,i), with Q the final n x r basis and
%            Y an r x m x N array. The basis only grows during the updates
%            (columns are appended), so the coefficients computed with an
%            earlier basis are simply padded with zeros. Forming Q*y costs
%            O(n r) per parameter and is not part of the algorithm: the
%            eigenvectors that are needed can be formed at the end, e.g. all
%            at once with Q * reshape(Y, size(Q,2), []).
%     t_off  time of the offline phase (seconds)
%     t_on   time of the online phase (seconds)

opts = set_defaults(opts, numel(prob.param));
m = prob.neig;
N = numel(prob.param);
lam = nan(m, N);
C = cell(N, 1);                         % coefficients y in the current basis
is_snap = false(N, 1);
is_snap(opts.snap) = true;

%% Offline phase
tic
nb = m + opts.buffer;                   % eigenvectors per snapshot point
X = zeros(prob.n, nb * numel(opts.snap));
normA = zeros(numel(opts.snap), 1);     % ||A(p)|| at the snapshots, for the residual estimate
for j = 1:numel(opts.snap)
    i = opts.snap(j);
    [Xj, lj] = full_eigs(prob, prob.param(i), nb, 'MaxIterations', 2000);
    X(:, (j-1)*nb+1 : j*nb) = Xj;
    lam(:, i) = lj(1:m);
    if isfinite(opts.tol)
        normA(j) = norm_estimate(prob, prob.param(i), lj);
    end
end
Q = orth(X);
for j = 1:numel(opts.snap)              % exact eigenvectors at the snapshots
    C{opts.snap(j)} = Q' * X(:, (j-1)*nb+1 : (j-1)*nb+m);
end
[idx, w, dSQ] = subsample(Q, opts.sampling);
t_off = toc;

%% Online phase
tic
for i = 1:N
    if is_snap(i), continue; end
    p = prob.param(i);

    % Subsampled Rayleigh-Ritz: M = (SQ) \ (S A(p) Q)
    SAQ = w .* prob.Arows(p, idx, Q);
    [Y, D] = eig(dSQ \ SAQ);
    li = diag(D);
    if prob.sym, li = real(li); end
    [li, ord] = sort_eig(li, prob.sigma);
    Y = Y(:, ord(1:m));
    lam(:, i) = li(1:m);

    % Update of the snapshot matrix (by frequency or by estimated residual)
    update = mod(i, opts.update_freq) == 0;
    if ~update && isfinite(opts.tol)
        SQY = w .* Q(idx, :) * Y;                % S Q y
        [~, near] = min(abs(prob.param(opts.snap) - p));
        res = norm(SAQ*Y - SQY .* li(1:m).', 'fro') / normA(near);
        update = res > opts.tol;
    end

    if update
        [Xi, li] = full_eigs(prob, p, nb, 'StartVector', Q*Y(:, 1), ...
                             'Tolerance', opts.tol_upd);
        lam(:, i) = li(1:m);
        Q = add_to_basis(Q, Xi);
        C{i} = Q' * Xi(:, 1:m);
        [idx, w, dSQ] = subsample(Q, opts.sampling);
    else
        C{i} = Y;
    end
end
t_on = toc;
Y = pad_coefficients(C, size(Q, 2), m);
end


function Y = pad_coefficients(C, r, m)
% Collect the coefficients in an r x m x N array, padding with zeros those
% computed when the basis had fewer columns.
Y = zeros(r, m, numel(C));
for i = 1:numel(C)
    Y(1:size(C{i}, 1), :, i) = C{i};
end
end


function [idx, w, dSQ] = subsample(Q, method)
% Rows S of Q and factorization of the (weighted) subsampled basis SQ.
[idx, w] = select_rows(Q, method);
dSQ = decomposition(w .* Q(idx, :));
end


function nA = norm_estimate(prob, p, lam)
% ||A(p)||, used to make the estimated residual relative: the 1-norm if
% A(p) is available as a matrix, otherwise the largest computed |eigenvalue|.
if isfield(prob, 'A')
    nA = norm(prob.A(p), 1);
else
    nA = max(abs(lam));
end
end


function opts = set_defaults(opts, N)
if ~isfield(opts, 'snap'),        opts.snap = floor(linspace(1, N, opts.k)); end
if ~isfield(opts, 'buffer'),      opts.buffer = 0; end
if ~isfield(opts, 'update_freq'), opts.update_freq = Inf; end
if ~isfield(opts, 'tol'),         opts.tol = Inf; end
if ~isfield(opts, 'tol_upd'),     opts.tol_upd = 1e-8; end
end
