function V = form_vectors(Q, Y, idx)
%FORM_VECTORS  Eigenvectors (or singular vectors) from the factored form Q*Y.
%
%   V = form_vectors(Q, Y, idx) returns the cell array V with
%   V{j} = Q * Y(:,:,idx(j)), computed with a single matrix-matrix product.

[r, m, ~] = size(Y);
W = Q * reshape(Y(:, :, idx), r, m * numel(idx));
V = mat2cell(W, size(Q, 1), repmat(m, 1, numel(idx)));
end
