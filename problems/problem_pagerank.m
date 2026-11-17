function [prob, opts] = problem_pagerank()
%PROBLEM_PAGERANK  Example 3 (Section 8.3): degree-biased parametric PageRank.
%
%   Leading eigenpair of
%       A(p) = d W(p) + (1-d)/n * 1 1',   d = 0.9,
%   on the web-NotreDame graph (n = 325729, data/web-NotreDame.mat).
%   M = A0 D^{-1} is the column-stochastic transition matrix (self-loops
%   removed, then added on dangling nodes), and W(p) is M with entry (i,j)
%   weighted by (1 + indeg_i)^p and columns normalized again:
%       W(p) = diag(w) M diag(1./c),   w = (1 + indeg).^p,   c = w' * M.
%   A(p) is dense and never formed: products with A(p) and rows of A(p)
%   are computed from the sparse M.
%
%   p: 1000 equispaced values in [0, 0.5]; 5 equispaced snapshot points.

data = load('web-NotreDame.mat');
A0 = spones(data.Problem.A);
n = size(A0, 1);
A0 = A0 - spdiags(diag(A0), 0, n, n);          % remove self-loops
dangling = find(sum(A0, 1) == 0);
A0 = A0 + sparse(dangling, dangling, 1, n, n);  % self-loops on dangling nodes
M = A0 * spdiags(1 ./ sum(A0, 1)', 0, n, n);   % column-stochastic
indeg = full(sum(M > 0, 2));
MT = M';                                        % rows of M = columns of MT
d = 0.9;

prob.n = n;
prob.param = linspace(0, 0.5, 1000)';
prob.neig = 1;
prob.sigma = 'largestabs';
prob.sym = false;
prob.Amult = @(p, X) mult(p, X, M, indeg, d);
prob.Arows = @(p, idx, X) rows(p, idx, X, M, MT, indeg, d);

opts.k = 5;
end


function Y = mult(p, X, M, indeg, d)
% Y = A(p) * X. The weights w and the column sums c only depend on p, so
% they are kept from the previous call (eigs calls this many times with
% the same p).
persistent p_last w c
if isempty(p_last) || p ~= p_last
    p_last = p;
    w = (1 + indeg).^p;
    c = (w' * M)';                              % column sums of diag(w) M
end
Y = d * (w .* (M * (X ./ c))) + (1 - d) / size(M, 1) * sum(X, 1);
end


function Y = rows(p, idx, X, M, MT, indeg, d)
% Y = A(p)(idx,:) * X. Only the columns j of M that have a nonzero in the
% rows idx are touched, together with their normalization c_j, so the
% cost does not depend on n (apart from the teleportation term sum(X,1)).
[j, k, v] = find(MT(:, idx));                   % M(idx(k), j) = v
[uj, ~, pos] = unique(j);
[ri, cj, mv] = find(M(:, uj));
c = accumarray(cj, mv .* (1 + indeg(ri)).^p);   % c_j = sum_i w_i M_ij
v = v .* (1 + indeg(idx(k))).^p ./ c(pos);      % entries of W(p)(idx,:)
Y = d * (sparse(k, 1:numel(j), v, numel(idx), numel(j)) * X(j, :)) ...
    + (1 - d) / size(M, 1) * sum(X, 1);
end
