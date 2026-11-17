function [sig, Ql, Yl, Qr, Yr, t_off, t_on] = SubApSnap_SVD(prob, opts)
%SUBAPSNAP_SVD  Sequence of SVD problems with snapshots and row subsampling.
%
%   [sig, Ql, Yl, Qr, Yr, t_off, t_on] = SubApSnap_SVD(prob, opts) computes the m
%   largest singular triplets of A(p) for every p in prob.param (Section 7
%   of the paper).
%
%   Offline phase
%     1. Compute m (+ buffer) singular triplets of A(p) at the snapshot
%        points with svds; collect the left singular vectors in X_l and the
%        right ones in X_r.
%     2. Q_l, Q_r = orthonormal bases of span(X_l), span(X_r).
%     3. Select rows S of Q_l (select_rows: LU or leverage scores).
%   Online phase, for each new parameter p
%     4. Solve min_M ||S(Q_l M - A(p) Q_r)||_F, i.e. M = (S Q_l) \ (S A(p) Q_r).
%        Only the rows S of A(p) are needed (prob.Arows).
%     5. The SVD M = Y_l Sigma Y_r' gives the approximations
%        (sigma, Q_l y_l, Q_r y_r).
%     6. Every opts.update_freq parameters, refine the singular vectors with
%        svds (started from Q_r y_r), add them to the bases and select the
%        rows again.
%
%   prob: problem definition (see problems/problem_transformer.m)
%     m, n    size of A(p)
%     param   the N parameter values
%     nsv     number m of wanted (largest) singular values
%     A       @(p) A(p) as a matrix
%     Arows   @(p,idx,X) A(p)(idx,:)*X
%
%   opts: k, snap, sampling, buffer, update_freq, tol_upd, as in
%   SubApSnap_eig.
%
%   Output:
%     sig    m x N matrix of singular values
%     Ql, Yl the left singular vectors in factored form, Ql*Yl(:,:,i) for
%            prob.param(i) (see the output Q, Y of SubApSnap_eig)
%     Qr, Yr the right singular vectors, Qr*Yr(:,:,i)
%     t_off, t_on  offline/online time (seconds)

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
[idx, w, dSQ] = subsample(Ql, opts.sampling);
t_off = toc;

%% Online phase
tic
for i = 1:N
    if is_snap(i), continue; end
    p = prob.param(i);

    % Subsampled least-squares problem M = (S Q_l) \ (S A(p) Q_r)
    [Yl, S, Yr] = svd(dSQ \ (w .* prob.Arows(p, idx, Qr)));
    sig(:, i) = diag(S(1:m, 1:m));

    if mod(i, opts.update_freq) == 0
        [Ul, S, Vr] = svds(prob.A(p), nb, 'largest', 'RightStartVector', ...
                           Qr*Yr(:, 1), 'Tolerance', opts.tol_upd);
        sig(:, i) = diag(S(1:m, 1:m));
        Ql = add_to_basis(Ql, Ul);
        Qr = add_to_basis(Qr, Vr);
        Cl{i} = Ql' * Ul(:, 1:m);
        Cr{i} = Qr' * Vr(:, 1:m);
        [idx, w, dSQ] = subsample(Ql, opts.sampling);
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


function [idx, w, dSQ] = subsample(Q, method)
% Rows S of Q and factorization of the (weighted) subsampled basis SQ.
[idx, w] = select_rows(Q, method);
dSQ = decomposition(w .* Q(idx, :));
end
