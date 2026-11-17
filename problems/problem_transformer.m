function [prob, opts] = problem_transformer(folder)
%PROBLEM_TRANSFORMER  Example 6 (Section 8.6): attention matrix during fine-tuning.
%
%   Leading singular triplet of the attention matrix
%       A(p) = softmax( Q(p) K(p)' / sqrt(64) )   (softmax applied row-wise),
%   where Q(p), K(p) are the 5000 x 64 query and key matrices (after the
%   rotary embedding) of the 4th head of the last layer of ModernBERT-base,
%   for a fixed 5000-token document, at training step p of the last epoch
%   of the fine-tuning on LexGLUE/SCOTUS. They are produced by
%   train_modernbert.py and saved in folder/step_<s>_ROPE.mat (variables
%   Q_rope and K_rope), with s = 1689, ..., 2251 for p = 1, ..., 563. All
%   of them are loaded in memory at the beginning.
%
%   A row of A(p) only needs the corresponding row of Q(p), so the rows of
%   A(p) are computed without forming A(p).
%
%   p = 1, ..., 563. The 50 snapshot points are the first 50 steps.
%
%   folder (optional), default 'python/matrici_matlab_dirette_rope'.

if nargin < 1, folder = fullfile('python', 'matrici_matlab_dirette_rope'); end
param = (1:563)';
Qs = cell(numel(param), 1);
Ks = cell(numel(param), 1);
for p = param'
    s = load(fullfile(folder, sprintf('step_%d_ROPE.mat', 1688 + p)), 'Q_rope', 'K_rope');
    Qs{p} = s.Q_rope;
    Ks{p} = s.K_rope;
end

prob.m = size(Qs{1}, 1);
prob.n = size(Ks{1}, 1);
prob.param = param;
prob.nsv = 1;
prob.A = @(p) attention(Qs{p}, Ks{p});
prob.Arows = @(p, idx, X) attention(Qs{p}(idx, :), Ks{p}) * X;

opts.snap = 1:50;
end


function A = attention(Q, K)
% Row-wise softmax of Q K' / sqrt(64) (the row maximum is subtracted for
% numerical stability).
W = (Q * K') / sqrt(64);
W = exp(W - max(W, [], 2));
A = W ./ sum(W, 2);
end
