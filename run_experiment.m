function run_experiment(name, varargin)
%RUN_EXPERIMENT  Run one eigenvalue experiment of Section 8 and save the results.
%
%   run_experiment(name) with name = 'exponential', 'differential',
%   'pagerank', 'kernel' or 'graph' (Examples 1-5). The problem is defined
%   by problems/problem_<name>.m; extra arguments are passed to it (e.g.
%   run_experiment('exponential', 2e4) for a small test).
%
%   The three methods of the paper are run one after the other:
%       LU        SubApSnap_eig with LU row selection
%       Leverage  SubApSnap_eig with leverage-score sampling, run NSEEDS = 10
%                 times with different random seeds (rng(42), ..., rng(51))
%   LU is also run NSEEDS times (alternated with Leverage), to take the
%   median of its computational times.
%       MOR       ApSnap_eig (no subsampling)
%   Their eigenpairs are compared with eigs at 100 parameters equispaced
%   in the index of prob.param (snapshot points excluded):
%       residual         ||A(p) V - V Lambda||_F   (unit eigenvectors)
%       eigenvalue error mean_i |lambda_i - lambda_i^exact| / |lambda_i^exact|
%       eigenvector err. subspace(V, V_exact) (largest principal angle)
%   For Leverage, the files contain the median over the seeds and, in the
%   columns *_Leverage_lo and *_Leverage_hi, a 95% confidence interval for
%   the median (see utils/ci_median.m).
%   The results are written in results/<name>/ as tab-separated files read
%   by the LaTeX source of the paper:
%       res.dat, eigval.dat, eigvec.dat   errors (columns p, *_LU, *_Leverage, *_POD)
%       plot.dat                          eigenvalues (Ex_eig_j, LU_eig_j, ...)
%       times.dat                         offline/online times and ratios to MOR
%                                         (median over the NSEEDS runs for LU and Leverage)
%   (MOR is called POD in the column names, as in the original files.)

NSEEDS = 10;
addpath(genpath(fileparts(mfilename('fullpath'))));
name = char(name);
[prob, opts] = feval(['problem_' name], varargin{:});
if ~isfield(prob, 'Amult'), prob.Amult = @(p, X) prob.A(p) * X; end
outdir = fullfile('results', name);
if ~isfolder(outdir), mkdir(outdir); end

% Test parameters and reference solutions (not timed)
N = numel(prob.param);
if ~isfield(opts, 'snap'), opts.snap = floor(linspace(1, N, opts.k)); end
test = round(linspace(1, N, 100));
test = test(~ismember(test, opts.snap));
nt = numel(test);
m = prob.neig;
fprintf('%s: n = %d, %d parameters, reference solutions...\n', name, prob.n, N);
lam_ex = zeros(m, nt);
V_ex = cell(nt, 1);
for j = 1:nt
    [V_ex{j}, lj] = full_eigs(prob, prob.param(test(j)), m);
    lam_ex(:, j) = lj;
end

% The runs: LU and Leverage NSEEDS times each, alternated (LU is
% deterministic and is repeated only for the timings), then MOR
seeds = num2cell(repelem(42 + (0:NSEEDS-1)', 2));
runs = [[repmat({'LU'; 'Leverage'}, NSEEDS, 1), seeds]; {'MOR', 0}];
nr = size(runs, 1);
res = zeros(nt, nr); err_lam = res; err_vec = res;
lam = zeros(nt, m, nr);
t = zeros(nr, 2);                          % [offline, online]
for k = 1:nr
    [method, seed] = runs{k, :};
    fprintf('%s (seed %d)...\n', method, seed);
    rng(seed);
    switch method
        case 'LU',       [lk, Q, Y, t(k,1), t(k,2)] = SubApSnap_eig(prob, setfield(opts, 'sampling', 'lu'));       %#ok<SFLD>
        case 'Leverage', [lk, Q, Y, t(k,1), t(k,2)] = SubApSnap_eig(prob, setfield(opts, 'sampling', 'leverage')); %#ok<SFLD>
        case 'MOR',      [lk, Q, Y, t(k,1), t(k,2)] = ApSnap_eig(prob, opts);
    end
    fprintf('   offline %.2f s, online %.2f s\n', t(k,1), t(k,2));
    Vk = form_vectors(Q, Y, test);     % eigenvectors Q*y at the test points
    for j = 1:nt
        p = prob.param(test(j));
        [lj, Vj] = match(lk(:, test(j)), Vk{j}, lam_ex(:, j));
        res(j, k) = norm(prob.Amult(p, Vj) - Vj .* lj.', 'fro');
        err_lam(j, k) = mean(abs(lj - lam_ex(:, j)) ./ abs(lam_ex(:, j)));
        err_vec(j, k) = subspace(Vj, V_ex{j});
        lam(j, :, k) = lj;
    end
end

% Output files (Leverage: median over the seeds and confidence interval)
lu = 1:2:2*NSEEDS; lev = 2:2:2*NSEEDS;
p = prob.param(test(:));
summary = @(E) [E(:,1), median(E(:,lev), 2), E(:,end), ci_median(E(:,lev))];
cols = {'LU', 'Leverage', 'POD', 'Leverage_lo', 'Leverage_hi'};
write_dat(fullfile(outdir, 'res.dat'),    [p, summary(res)],     ['p', strcat('Res_', cols)]);
write_dat(fullfile(outdir, 'eigval.dat'), [p, summary(err_lam)], ['p', strcat('ErrLam_', cols)]);
write_dat(fullfile(outdir, 'eigvec.dat'), [p, summary(err_vec)], ['p', strcat('ErrAng_', cols)]);
j = string(1:m);
write_dat(fullfile(outdir, 'plot.dat'), ...
    [p, real(lam_ex.'), real(lam(:,:,1)), real(median(lam(:,:,lev), 3)), real(lam(:,:,end))], ...
    ['p', strcat('Ex_eig_', j), strcat('LU_eig_', j), strcat('Lev_eig_', j), strcat('POD_eig_', j)]);
write_times(fullfile(outdir, 'times.dat'), [median(t(lu,:), 1); median(t(lev,:), 1); t(end,:)]);
fprintf('Results written in %s\n', outdir);
end


function [l, V] = match(l, V, l_ex)
% Order the computed eigenpairs as the reference ones (needed when some
% eigenvalues are (nearly) multiple and the sorting can swap them).
P = perms(1:numel(l));
[~, best] = min(vecnorm(l(P) - l_ex.', 2, 2));
l = l(P(best, :));
V = V(:, P(best, :));
end
