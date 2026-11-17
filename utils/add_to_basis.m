function Q = add_to_basis(Q, W)
%ADD_TO_BASIS  Enlarge the orthonormal basis Q with the columns of W.
%
%   Each column of W is orthogonalized against Q (classical Gram-Schmidt,
%   applied twice for stability) and appended to Q, unless it already lies
%   in span(Q) up to a relative tolerance of 1e-8. This is the update of
%   the snapshot matrix X <- [X, W] of the paper, done directly on its
%   orthonormal basis Q.

for j = 1:size(W, 2)
    v = W(:, j);
    nv = norm(v);
    v = v - Q*(Q'*v);
    v = v - Q*(Q'*v);
    if norm(v) > 1e-8 * nv
        Q = [Q, v / norm(v)]; %#ok<AGROW>
    end
end
end
