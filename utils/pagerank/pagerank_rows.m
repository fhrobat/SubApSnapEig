function SAnowSnap = pagerank_rows(t, idx, X, M_T, in_deg)

persistent last_t w col_sums
    
    if isempty(last_t) || last_t ~= t
        last_t = t;
        w = (1 + in_deg) .^ t;
        col_sums = (M_T * w)';
    end

    n = size(M_T, 1);
    p = 0.9;
    
    X_scaled = X ./ col_sums';
    
    M_rows_X = (M_T(:, idx))' * X_scaled;
    M_rows_X = M_rows_X .* w(idx);
    
    teleport_X = (1 / n) * sum(X, 1);
    SAnowSnap = p * M_rows_X + (1 - p) * teleport_X;
end