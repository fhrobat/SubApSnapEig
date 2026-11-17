function C = ci_median(E)
%CI_MEDIAN  Distribution-free confidence interval for the median of each row.
%
%   C = ci_median(E) returns [lo, hi] for every row of E (one sample per
%   column): lo and hi are the k-th smallest and k-th largest sample, with
%   the largest k such that P(Bin(n, 1/2) < k) <= 2.5%, so that [lo, hi]
%   contains the median with probability at least 95%. For n = 10 samples,
%   k = 2 (coverage 97.9%).

n = size(E, 2);
k = max(1, find(cumsum(arrayfun(@(i) nchoosek(n, i), 0:n)) / 2^n <= 0.025, 1, 'last'));
E = sort(E, 2);
C = [E(:, k), E(:, n - k + 1)];
end
