function [prob, opts] = problem_kernel(file)
%PROBLEM_KERNEL  Example 4 (Section 8.4): sparse Gaussian kernel on MNIST.
%
%   Leading eigenpair of the sparse Gaussian kernel matrix
%       A_ij(p) = exp(-||x_i - x_j||^2 / (2 p^2))  if x_j is among the K = 100
%                 nearest neighbours of x_i (x_i included) or vice versa,
%       A_ij(p) = 0 otherwise,
%   built from the n = 60000 MNIST training images (rows of X).
%   The squared distances are computed once (knnsearch) and stored in the
%   sparse symmetric matrix D2; A(p) is obtained by applying the exponential
%   to the nonzeros of D2 only. A tiny shift delta is added to the stored
%   squared distances so that the zero diagonal is kept as a nonzero.
%
%   p: 1000 equispaced values in [0.5 pbar, 1.5 pbar], where pbar is the
%   median distance between each point and its K-1 nearest neighbours
%   (the point itself excluded); 10 equispaced snapshot points.
%
%   file (optional): MAT-file with the n x 784 data matrix X (not included
%   in the repository), default 'mnist_completo.mat'.

if nargin < 1, file = 'mnist_completo.mat'; end
X = double(getfield(load(file, 'X'), 'X')); %#ok<GFLD>
n = size(X, 1);
K = 100;
delta = eps;

[nbr, dist] = knnsearch(X, X, 'K', K);
D2 = sparse(repmat((1:n)', 1, K), nbr, dist.^2 + delta, n, n);
D2 = max(D2, D2');                         % symmetric neighbourhood graph
pbar = median(dist(:, 2:end), 'all');

kern = @(v, p) exp(-(v - delta) / (2 * p^2));
prob.n = n;
prob.param = linspace(0.5 * pbar, 1.5 * pbar, 1000)';
prob.neig = 1;
prob.sigma = 'largestabs';
prob.sym = true;
prob.A = @(p) spfun(@(v) kern(v, p), D2);
prob.Arows = @(p, idx, X) rows_times(D2, idx, X, @(v, ~, ~) kern(v, p));

opts.k = 10;
end
