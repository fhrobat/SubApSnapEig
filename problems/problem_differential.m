function [prob, opts] = problem_differential(Nx)
%PROBLEM_DIFFERENTIAL  Example 2 (Section 8.2): operator with multiple eigenvalues.
%
%   The 3 smallest eigenvalues of the 5-point finite-difference
%   discretization of
%       -p Laplacian(u) - grad(div(u))   on [-1,1]^2,   u = 0 on the boundary,
%   for a vector field u = (u_1, u_2). The second smallest eigenvalue has
%   multiplicity 2. Each component lives on a uniform grid with Nx points
%   per direction, i.e. (Nx-2)^2 interior points, so n = 2 (Nx-2)^2. The
%   factor 1/h^2 is omitted, so that ||A(p)||_2 = O(1). The matrix is affine
%   in p:
%       A(p) = -p L - G,   L = blkdiag(Lap, Lap),   G = [Dx; Dy] [Dx, Dy],
%   where Lap is the 5-point Laplacian and Dx, Dy are central differences.
%   L and G are assembled once; both are symmetric, so the rows of A(p) are
%   extracted as columns of L and G.
%
%   p: 1000 log-spaced values in [1e-4, 1e-3]; 10 log-spaced snapshot points.
%   Nx (optional) is the number of grid points per direction, default 500.

if nargin < 1, Nx = 500; end
nx = Nx - 2;
e = ones(nx, 1);
I = speye(nx);
T = spdiags([e, -2*e, e], -1:1, nx, nx);   % 1D second difference
D = spdiags([-e/2, e/2], [-1, 1], nx, nx); % 1D central difference
Lap = kron(I, T) + kron(T, I);
L = blkdiag(Lap, Lap);
Dx = kron(I, D);                           % grid index i + (j-1)*nx
Dy = kron(D, I);
G = [Dx; Dy] * [Dx, Dy];

prob.n = 2 * nx^2;
prob.param = 1e-4 * logspace(0, 1, 1000)';
prob.neig = 3;
prob.sigma = 'smallestabs';
prob.sym = true;
prob.A = @(p) -p * L - G;
prob.Arows = @(p, idx, X) -p * rows_times(L, idx, X) - rows_times(G, idx, X);

opts.k = 10;
end
