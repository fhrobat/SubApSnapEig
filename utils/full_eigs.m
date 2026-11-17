function [V, lam] = full_eigs(prob, p, k, varargin)
%FULL_EIGS  k eigenpairs of the full matrix A(p) with eigs.
%
%   [V, lam] = full_eigs(prob, p, k, ...) returns the k eigenpairs of A(p)
%   selected by prob.sigma, sorted with sort_eig. Extra arguments are
%   passed to eigs (e.g. 'StartVector', 'Tolerance').
%   If prob.A is given, eigs works on the sparse matrix A(p) (required for
%   'smallestabs'); otherwise it uses the products prob.Amult(p, x).

if isfield(prob, 'A')
    [V, D] = eigs(prob.A(p), k, prob.sigma, varargin{:});
else
    [V, D] = eigs(@(x) prob.Amult(p, x), prob.n, k, prob.sigma, ...
                  'IsFunctionSymmetric', prob.sym, varargin{:});
end
[lam, ord] = sort_eig(diag(D), prob.sigma);
V = V(:, ord);
end
