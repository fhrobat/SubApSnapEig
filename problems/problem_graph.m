function [prob, opts] = problem_graph()
%PROBLEM_GRAPH  Example 5 (Section 8.5): eigenvector centrality of a varying graph.
%
%   Leading eigenpair of A(p) = D(p) A0 D(p), where A0 is the adjacency
%   matrix of coAuthorsDBLP (n = 299067, data/coAuthorsDBLP.mat) and D(p)
%   is diagonal with entries 1 + 0.5 sin(2 pi p / 1000 + Phi_i), with random
%   phases Phi_i uniform in [0, 2 pi] (fixed seed). A(p) is never formed.
%
%   p = 1, ..., 1000 (time-dependent case). The 20 snapshot points are the
%   first 20 parameters, and the snapshot matrix and the rows are updated
%   every 100 parameters.

data = load('coAuthorsDBLP.mat');
A0 = spones(data.Problem.A + data.Problem.A');
n = size(A0, 1);
rng(42);
Phi = 2 * pi * rand(n, 1);
dg = @(p, phi) 1 + 0.5 * sin(2 * pi * p / 1000 + phi);   % diagonal entries of D(p)

prob.n = n;
prob.param = (1:1000)';
prob.neig = 1;
prob.sigma = 'largestabs';
prob.sym = true;
prob.Amult = @(p, X) mult(dg(p, Phi), X, A0);
prob.Arows = @(p, idx, X) rows_times(A0, idx, X, ...  % A0 is symmetric
    @(v, i, k) v .* dg(p, Phi(i)) .* dg(p, Phi(idx(k))));

opts.snap = 1:20;
opts.update_freq = 100;
end


function Y = mult(d, X, A0)
% Y = D A0 D X with D = diag(d).
Y = d .* (A0 * (d .* X));
end
