function R = diff_rows(nu, L, Nx, eps_val, rows)

x = linspace(-L, L, Nx);
hx = x(2)-x(1);
nx = Nx-2;
ny = nx;
N = nx * ny;

c_center = -4.0;
c_xy = 1.0;

rows = rows(:);
m = numel(rows);
Ri = []; Rj = []; Rv = [];

for k = 1:m
    r = rows(k);
    if r <= N
        comp = 1; p = r;
    else
        comp = 2; p = r - N;
    end

    j = floor((p-1)/nx) + 1;
    i = p - (j-1)*nx;
    col_offset = (comp-1)*N;

    Ri(end+1)=k; Rj(end+1)=col_offset + p;    Rv(end+1)= -eps_val * nu * c_center;
    if i > 1,  Ri(end+1)=k; Rj(end+1)=col_offset + p-1;  Rv(end+1)= -eps_val * nu * c_xy; end
    if i < nx, Ri(end+1)=k; Rj(end+1)=col_offset + p+1;  Rv(end+1)= -eps_val * nu * c_xy; end
    if j > 1,  Ri(end+1)=k; Rj(end+1)=col_offset + p-nx; Rv(end+1)= -eps_val * nu * c_xy; end
    if j < ny, Ri(end+1)=k; Rj(end+1)=col_offset + p+nx; Rv(end+1)= -eps_val * nu * c_xy; end


    rowGrad_idx = []; rowGrad_val = [];
    if comp == 1
        if i > 1,  rowGrad_idx(end+1) = p-1; rowGrad_val(end+1) = -0.5; end
        if i < nx, rowGrad_idx(end+1) = p+1; rowGrad_val(end+1) = 0.5; end
    else
        if j > 1,  rowGrad_idx(end+1) = p-nx; rowGrad_val(end+1) = -0.5; end
        if j < ny, rowGrad_idx(end+1) = p+nx; rowGrad_val(end+1) = 0.5; end
    end

    for t = 1:numel(rowGrad_idx)
        s = rowGrad_idx(t);
        gv = rowGrad_val(t);
        js_s = floor((s-1)/nx) + 1;
        is_s = s - (js_s-1)*nx;

        if is_s > 1
            Ri(end+1)=k; Rj(end+1)=s-1;    Rv(end+1)= -(gv * (-0.5));
        end
        if is_s < nx
            Ri(end+1)=k; Rj(end+1)=s+1;    Rv(end+1)= -(gv * (0.5));
        end
        if js_s > 1
            Ri(end+1)=k; Rj(end+1)=N+s-nx; Rv(end+1)= -(gv * (-0.5));
        end
        if js_s < ny
            Ri(end+1)=k; Rj(end+1)=N+s+nx; Rv(end+1)= -(gv * (0.5));
        end
    end
end
R = sparse(Ri, Rj, Rv, m, 2*N);
end