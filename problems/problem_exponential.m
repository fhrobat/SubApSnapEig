function [prob, opts] = problem_exponential(N)
%PROBLEM_EXPONENTIAL  Example 1 (Section 8.1): exponential differential operator.
%
%   Smallest eigenvalue of the 3-point finite-difference discretization of
%       -(e^{p x} u'(x))'   on [-1,1],   u(-1) = u(1) = 0,
%   on a uniform grid with N points (n = N-2 unknowns). The factor 1/h^2 is
%   omitted, so that ||A(p)||_2 = O(1). Row i of A(p) is
%       -a_{i-1/2} u_{i-1} + (a_{i-1/2} + a_{i+1/2}) u_i - a_{i+1/2} u_{i+1},
%   with a_{i+1/2} = exp(p x_{i+1/2}) evaluated at the midpoints.
%
%   p: 3000 log-spaced values in [0.1, 1]; 10 log-spaced snapshot points.
%   N (optional) is the number of grid points, default 2e6.

if nargin < 1, N = 2e6; end
n = N - 2;
h = 2 / (N - 1);
xm = -1 + ((1:N-1)' - 0.5) * h;           % midpoints x_{i+1/2}, i = 1..N-1

prob.n = n;
prob.param = logspace(-1, 0, 3000)';
prob.neig = 1;
prob.sigma = 'smallestabs';
prob.sym = true;
prob.A = @(p) assemble(exp(p * xm), n);
prob.Arows = @(p, idx, X) rows(p, idx, X, h, n);

opts.k = 10;                               % equispaced indices = log-spaced p
end


function A = assemble(a, n)
% a(i) = a_{i-1/2} for i = 1..n+1 (coefficients at the n+1 midpoints).
off = -a(2:n);
A = spdiags([[off; 0], a(1:n) + a(2:n+1), [0; off]], -1:1, n, n);
end


function Y = rows(p, idx, X, h, n)
% Y = A(p)(idx,:) * X, computing only the coefficients that are needed.
am = exp(p * (-1 + (idx - 0.5) * h));      % a_{i-1/2}
ap = exp(p * (-1 + (idx + 0.5) * h));      % a_{i+1/2}
left = max(idx - 1, 1);                    % at the boundary the neighbour is
right = min(idx + 1, n);                   % missing: its coefficient is set to 0
Y = (am + ap) .* X(idx, :) - (am .* (idx > 1)) .* X(left, :) ...
                           - (ap .* (idx < n)) .* X(right, :);
end
