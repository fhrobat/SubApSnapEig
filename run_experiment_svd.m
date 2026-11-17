function run_experiment_svd(varargin)
%RUN_EXPERIMENT_SVD  Run the singular value experiment (Example 6) and save the results.
%
%   run_experiment_svd() loads the attention matrices with
%   problems/problem_transformer.m (extra arguments are passed to it) and
%   runs SubApSnap_SVD with LU and leverage-score row selection and the
%   baseline ApSnap_SVD (MOR). The results are compared with svds at 100
%   parameters (snapshot points excluded):
%       residual            ||A(p) V - U Sigma||_F / ((||A(p)||_1 + max sigma) ||V||_F)
%       singular value err. mean_i |sigma_i - sigma_i^exact| / sigma_i^exact
%       singular vector err. max(subspace(U, U_exact), subspace(V, V_exact))
%   and written in results/transformer/ with the same file names and
%   columns as run_experiment. As there, the Leverage variant is run with
%   NSEEDS = 10 random seeds and the files contain the median over the seeds
%   and a 95% confidence interval for it; LU is also run NSEEDS times, and the times are
%   the median over the runs.

NSEEDS = 10;
addpath(genpath(fileparts(mfilename('fullpath'))));
[prob, opts] = problem_transformer(varargin{:});
outdir = fullfile('results', 'transformer');
if ~isfolder(outdir), mkdir(outdir); end

% Test parameters and reference solutions (not timed)
N = numel(prob.param);
test = round(linspace(1, N, 100));
test = test(~ismember(test, opts.snap));
m = prob.nsv;
fprintf('transformer: %d x %d, %d parameters, reference solutions...\n', prob.m, prob.n, N);
sig_ex = zeros(m, numel(test));
U_ex = cell(numel(test), 1); V_ex = U_ex; normA = zeros(numel(test), 1);
for j = 1:numel(test)
    Ap = prob.A(prob.param(test(j)));
    [U_ex{j}, S, V_ex{j}] = svds(Ap, m);
    sig_ex(:, j) = diag(S);
    normA(j) = norm(Ap, 1);
end

% The runs: LU and Leverage NSEEDS times each, alternated (LU is
% deterministic and is repeated only for the timings), then MOR
seeds = num2cell(repelem(42 + (0:NSEEDS-1)', 2));
runs = [[repmat({'LU'; 'Leverage'}, NSEEDS, 1), seeds]; {'MOR', 0}];
nr = size(runs, 1);
res = zeros(numel(test), nr); err_sig = res; err_vec = res;
sig = zeros(numel(test), m, nr);
t = zeros(nr, 2);
for k = 1:nr
    [method, seed] = runs{k, :};
    fprintf('%s (seed %d)...\n', method, seed);
    rng(seed);
    switch method
        case 'LU',       [sk, Ql, Yl, Qr, Yr, t(k,1), t(k,2)] = SubApSnap_SVD(prob, setfield(opts, 'sampling', 'lu'));       %#ok<SFLD>
        case 'Leverage', [sk, Ql, Yl, Qr, Yr, t(k,1), t(k,2)] = SubApSnap_SVD(prob, setfield(opts, 'sampling', 'leverage')); %#ok<SFLD>
        case 'MOR',      [sk, Ql, Yl, Qr, Yr, t(k,1), t(k,2)] = ApSnap_SVD(prob, opts);
    end
    fprintf('   offline %.2f s, online %.2f s\n', t(k,1), t(k,2));
    Uk = form_vectors(Ql, Yl, test);   % singular vectors at the test points
    Vk = form_vectors(Qr, Yr, test);
    for j = 1:numel(test)
        sj = sk(:, test(j));
        Ap = prob.A(prob.param(test(j)));
        res(j, k) = norm(Ap * Vk{j} - Uk{j} .* sj.', 'fro') / ...
                    ((normA(j) + max(abs(sj))) * norm(Vk{j}, 'fro'));
        err_sig(j, k) = mean(abs(sj - sig_ex(:, j)) ./ sig_ex(:, j));
        err_vec(j, k) = max(subspace(Uk{j}, U_ex{j}), subspace(Vk{j}, V_ex{j}));
        sig(j, :, k) = sj;
    end
end

% Output files (Leverage: median over the seeds and confidence interval)
lu = 1:2:2*NSEEDS; lev = 2:2:2*NSEEDS;
p = prob.param(test(:));
summary = @(E) [E(:,1), median(E(:,lev), 2), E(:,end), ci_median(E(:,lev))];
cols = {'LU', 'Leverage', 'POD', 'Leverage_lo', 'Leverage_hi'};
write_dat(fullfile(outdir, 'res.dat'),    [p, summary(res)],     ['p', strcat('Res_', cols)]);
write_dat(fullfile(outdir, 'eigval.dat'), [p, summary(err_sig)], ['p', strcat('ErrLam_', cols)]);
write_dat(fullfile(outdir, 'eigvec.dat'), [p, summary(err_vec)], ['p', strcat('ErrAng_', cols)]);
j = string(1:m);
write_dat(fullfile(outdir, 'plot.dat'), [p, sig_ex.', sig(:,:,1), median(sig(:,:,lev), 3), sig(:,:,end)], ...
    ['p', strcat('Ex_eig_', j), strcat('LU_eig_', j), strcat('Lev_eig_', j), strcat('POD_eig_', j)]);
write_times(fullfile(outdir, 'times.dat'), [median(t(lu,:), 1); median(t(lev,:), 1); t(end,:)]);
fprintf('Results written in %s\n', outdir);
end
