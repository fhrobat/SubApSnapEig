function A = pagerank_full(t, M)
    n = size(M, 1);
    p = 0.9;
    in_deg = sum(M > 0, 2);
    w = (1 + in_deg) .^ t;
    col_sums = w' * M;
    M_mod = spdiags(w, 0, n, n) * M * spdiags(1 ./ col_sums', 0, n, n);
    E = ones(n, n) / n;
    A = p * M_mod + (1 - p) * E;
end