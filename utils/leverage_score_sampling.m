function [indices, probs] = leverage_score_sampling(V, s)
    % Leverage score sampling
    % Input: 
    %   V - (n x r) matrix with orthonormal columns
    %   s - number of samples to draw (with replacement)
    % Output:
    %   indices - sampled row indices (1-based)
    %   probs   - leverage score probabilities (length n)

    [n, r] = size(V);
    leverage_scores = sum(abs(V).^2, 2);    % n x 1
    probs = leverage_scores / r;

    indices = randsample(n, s, true, probs);
end