function Lpen = diff_full(nu, L, Nx, eps)

if nargin<4
    error('Usage: Lpen = diff_full(nu, L, Nx, eps)');
end

x = linspace(-L, L, Nx);
hx = x(2)-x(1);
nx = Nx-2;
ny = nx;
N = nx * ny;

e = ones(nx,1);
T = spdiags([e -2*e e], [-1 0 1], nx, nx); 
I = speye(nx);

Lap2 = kron(speye(ny), T) + kron(T, speye(nx));
Lap_vec = blkdiag(Lap2, Lap2);

rows = []; cols = []; vals = [];
for j = 1:ny
    for i = 1:nx
        p = i + (j-1)*nx;
        if i>1,  rows(end+1)=p; cols(end+1)=p-1; vals(end+1)=-0.5; end
        if i<nx, rows(end+1)=p; cols(end+1)=p+1; vals(end+1)=0.5; end
    end
end
Dx_h = sparse(rows, cols, vals, N, N);

rows = []; cols = []; vals = [];
for j = 1:ny
    for i = 1:nx
        p = i + (j-1)*nx;
        if j>1,  rows(end+1)=p; cols(end+1)=p-nx; vals(end+1)=-0.5; end
        if j<ny, rows(end+1)=p; cols(end+1)=p+nx; vals(end+1)=0.5; end
    end
end
Dy_h = sparse(rows, cols, vals, N, N);


Grad_h = [Dx_h; Dy_h];
Div_h = [Dx_h, Dy_h];
Gdiv_scaled = Grad_h * Div_h;

Lpen = -eps * nu * Lap_vec - Gdiv_scaled;

end