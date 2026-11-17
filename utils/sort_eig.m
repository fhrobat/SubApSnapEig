function [lam, ord] = sort_eig(lam, sigma)
%SORT_EIG  Sort eigenvalues as requested by sigma.
%
%   'largestabs'  : by decreasing absolute value;
%   'smallestabs' : by increasing absolute value.
%   ord is the permutation, so that the eigenvectors are V(:, ord).

if strcmp(sigma, 'smallestabs')
    [~, ord] = sort(abs(lam), 'ascend');
else
    [~, ord] = sort(abs(lam), 'descend');
end
lam = lam(ord);
end
