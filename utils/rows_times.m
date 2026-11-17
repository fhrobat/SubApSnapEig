function Y = rows_times(AT, idx, X, f)
%ROWS_TIMES  Y = A(idx,:) * X, touching only the nonzeros of the rows idx.
%
%   AT is A' (a sparse matrix, so that the rows of A are columns of AT and
%   can be extracted quickly). The cost is O(nnz(A(idx,:)) * size(X,2)),
%   independent of n.
%
%   Y = rows_times(AT, idx, X, f) first applies f to the extracted entries:
%   f(v, i, k) receives the values v, their column indices i in A and their
%   row numbers k (1..numel(idx)), and returns the new values. This is used
%   when the entries of A(p) are a function of those of a fixed matrix.

[i, k, v] = find(AT(:, idx));
if nargin > 3
    v = f(v, i, k);
end
Y = sparse(k, 1:numel(i), v, numel(idx), numel(i)) * X(i, :);
end
