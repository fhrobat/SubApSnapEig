function [idx, w] = select_rows(Q, method)
%SELECT_ROWS  Choose the rows S used to subsample A(p)Q - QM.
%
%   [idx, w] = select_rows(Q, method) selects rows of the n x r matrix Q
%   with orthonormal columns. The subsampled least-squares problem is then
%       min_M || diag(w) * ( A(p)(idx,:)*Q - Q(idx,:)*M ) ||_F.
%
%   method = 'lu'       : the r pivot rows of LU with partial pivoting
%                         (square subsampled problem), w = 1.
%   method = 'leverage' : 4r rows drawn with replacement with probabilities
%                         pi_i = ||Q(i,:)||^2 / r (the leverage scores), and
%                         weights w_i = 1/sqrt(4r*pi_i).

r = size(Q, 2);
switch method
    case 'lu'
        [~, ~, perm] = lu(Q, 'vector');
        idx = perm(1:r).';
        w = ones(r, 1);
    case 'leverage'
        s = 4*r;
        prob = sum(abs(Q).^2, 2) / r;
        idx = randsample(size(Q, 1), s, true, prob);
        w = 1 ./ sqrt(s * prob(idx));
    otherwise
        error('Unknown sampling method ''%s''.', method);
end
end
