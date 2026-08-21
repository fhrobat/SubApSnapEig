function R = exp_rows(mu, L, N, rows)

rows = rows(:);
m = numel(rows);
n = N - 2; % Number of interior nodes

if any(rows < 1 | rows > n)
    error('Row indices must be between 1 and %d', n);
end

h = (2 * L) / (N - 1); 
am = exp(mu * (-L + (rows - 0.5) * h)); % Left faces: a_{i-1/2}
ap = exp(mu * (-L + (rows + 0.5) * h)); % Right faces: a_{i+1/2}

main_diag = -(ap + am); 

Ri = repmat((1:m)', 3, 1);
Rj = [rows - 1; rows; rows + 1]; 
Rv = [am; main_diag; ap];

mask = (Rj >= 1) & (Rj <= n);

R = sparse(Ri(mask), Rj(mask), Rv(mask), m, n);

end