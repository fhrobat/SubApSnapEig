# SubApSnapEig

MATLAB code accompanying the paper

> F. Hrobat and Y. Nakatsukasa, *Solving a sequence of eigenvalue problems using snapshot and row subsampling*.

**SubApSnapEig** solves a sequence of eigenvalue problems $A(p)\,x(p) = \lambda(p)\,x(p)$ for many values of a parameter $p$. It combines a snapshot (reduced-order) subspace with row subsampling.

- **Offline phase.** Compute eigenvectors at a few snapshot points, collect them in the snapshot matrix $X$, compute an orthonormal basis $Q$ of $\mathrm{span}(X)$ and select a set of rows $S$ of $Q$ (LU with partial pivoting or leverage-score sampling).
- **Online phase.** For each new parameter $p$, solve the small subsampled least-squares problem $M = \arg\min_M \|S(A(p)Q - QM)\|_F$ and compute the eigenpairs $(\lambda, y)$ of $M$; the approximate eigenvectors are $Qy$. Only the selected rows of $A(p)$ are needed. The eigenvectors are returned in factored form (the basis $Q$ and the coefficients $y$ for every $p$), since forming $Qy$ costs $O(nr)$ per parameter; they can be formed at the end where needed, with a single product $QY$.
- **Updates (optional).** Every `update_freq` parameters, or when the estimated residual is larger than `tol`, the eigenvectors are refined with `eigs` and added to the basis, and the rows are selected again.

The method is compared with `MOR`, which uses the same snapshot subspace with a full (non-subsampled) Rayleigh–Ritz projection $M = Q^* A(p) Q$.

## Repository structure

| Path | Content |
|---|---|
| `SubApSnapEig/SubApSnap_eig.m`, `SubApSnapEig/SubApSnap_SVD.m` | The proposed method, for eigenvalue problems and for the SVD |
| `MOR/ApSnap_eig.m`, `MOR/ApSnap_SVD.m` | Baseline `MOR` method (snapshots + full Rayleigh–Ritz) |
| `problems/problem_*.m` | Definition of the six test problems (one file per example of Section 8) |
| `run_experiment.m` | Runs one eigenvalue experiment (Examples 1–5) and writes the `.dat` files used in the paper |
| `run_experiment_svd.m` | The same for the singular value experiment (Example 6) |
| `run_all.sh` | Runs all experiments one after the other, waiting for the machine to be idle and logging its load |
| `utils/` | Row selection, update of the basis, `eigs` wrapper, sparse row extraction, output files |
| `data/` | `web-NotreDame` and `coAuthorsDBLP` graphs (SuiteSparse Matrix Collection) |
| `train_modernbert.py` | Fine-tunes ModernBERT and extracts the query/key matrices used in Example 6 |
| `theory/theory.m` | Experiments illustrating the theoretical bounds (singular value decay and error bounds) |

## How a problem is described

Each file `problems/problem_<name>.m` returns a struct `prob` and the options `opts`:

- `prob.n`, `prob.param` (the $N$ parameter values, in the order in which they are processed), `prob.neig` (number of wanted eigenpairs), `prob.sigma` (`'largestabs'` or `'smallestabs'`), `prob.sym` (true for real symmetric $A(p)$);
- `prob.A(p)`: the sparse matrix $A(p)$, when it can be formed (needed for `'smallestabs'`);
- `prob.Amult(p, X)`: the product $A(p)X$, when $A(p)$ is not formed;
- `prob.Arows(p, idx, X)`: the product $A(p)(\mathrm{idx},:)\,X$, computed from the rows `idx` only.

The main options are `opts.k` or `opts.snap` (snapshot points), `opts.sampling` (`'lu'` or `'leverage'`), `opts.update_freq` and `opts.tol` (updates); see the help of `SubApSnap_eig`.
LU selects $r$ rows ($r$ = dimension of the basis), leverage-score sampling selects $4r$ rows with replacement.

## Experiments

```matlab
run_experiment('exponential')     % Example 1
run_experiment('differential')    % Example 2
run_experiment('pagerank')        % Example 3
run_experiment('kernel')          % Example 4 (needs the MNIST data, see below)
run_experiment('graph')           % Example 5
run_experiment_svd                % Example 6 (needs the matrices from train_modernbert.py)
```

Each call runs `SubApSnapEig` with LU (`LU`) and leverage-score (`Leverage`) row selection and `MOR`, compares them with `eigs` (or `svds`) at 100 parameter values, and writes in `results/<name>/` the residuals (`res.dat`), eigenvalue and eigenvector errors (`eigval.dat`, `eigvec.dat`), eigenvalues (`plot.dat`) and offline/online times (`times.dat`). Since leverage-score sampling is random, the `Leverage` variant is run with 10 seeds (42, ..., 51): the files contain the median over the seeds and, in the `*_Leverage_lo` and `*_Leverage_hi` columns, a distribution-free 95% confidence interval for the median (the 2nd smallest and 2nd largest of the 10 values). `LU` is also run 10 times (alternated with `Leverage`), and `times.dat` reports the median times of both variants. Extra arguments are passed to the problem, e.g. `run_experiment('exponential', 2e4)` runs Example 1 on a smaller grid. To time all experiments on an idle machine:

```bash
NTHREADS=8 MATLAB=/path/to/matlab ./run_all.sh
```

**Example 1: Exponential differential operator.** Smallest eigenvalue of the 3-point finite-difference discretization of $-\frac{d}{dx}\big(e^{px}\frac{du}{dx}\big)$ on $[-1,1]$ with zero boundary conditions (without the factor $1/h^2$). $n \approx 2\times 10^6$; 3000 log-spaced parameters in $[0.1, 1]$; 10 log-spaced snapshot points.

**Example 2: Differential operator with multiple eigenvalues.** The 3 smallest eigenvalues of the 5-point finite-difference discretization of $-p\,\Delta u - \nabla(\nabla\cdot u)$ on $[-1,1]^2$ with zero boundary conditions (without the factor $1/h^2$). The second smallest eigenvalue has multiplicity 2. A $498\times 498$ interior grid for each component gives $n = 496\,008$; 1000 log-spaced parameters in $[10^{-4}, 10^{-3}]$; 10 log-spaced snapshot points.

**Example 3: PageRank.** Leading eigenpair of a degree-biased parametric PageRank matrix on the `web-NotreDame` graph ($n = 325\,729$). The damping factor is $d = 0.9$, and link weights are scaled by $(1 + \text{in-degree}_i)^p$. 1000 equispaced parameters in $[0, 0.5]$; 5 equispaced snapshot points. Products with the dense $A(p)$ and its rows are computed implicitly.

**Example 4: Gaussian kernel.** Leading eigenpair of a sparse Gaussian kernel matrix built from the MNIST training set ($n = 60\,000$): $A_{ij}(p) = \exp(-\|x_i - x_j\|^2 / 2p^2)$ if $x_j$ is among the 100 nearest neighbors of $x_i$ or vice versa, and $0$ otherwise. 1000 equispaced parameters in $[0.5\,\bar p, 1.5\,\bar p]$, where $\bar p$ is the median distance between each point and its nearest neighbors; 10 equispaced snapshot points. The MNIST data (`mnist_completo.mat`, variable `X`, one image per row) is not included in the repository: put it in `data/` or pass the file name, `run_experiment('kernel', file)`.

**Example 5: Graph eigenvector centrality.** Leading eigenpair of $A(p) = D(p)\,A_0\,D(p)$, where $A_0$ is the adjacency matrix of `coAuthorsDBLP` ($n = 299\,067$). $D(p)$ is diagonal with entries $1 + 0.5\sin(2\pi p/1000 + \Phi_i)$ and random phases $\Phi_i$. Parameters $p = 1,\dots,1000$; 20 snapshot points at the start of the parameter range. This is the time-dependent case: the snapshot matrix and the row selection are updated every 100 steps.

**Example 6: Transformer attention (SVD).** Leading singular value and vectors of the attention matrix $\mathrm{softmax}\big(Q(p)K(p)^T/\sqrt{64}\big)$, where $Q(p), K(p)$ are the query and key matrices (after rotary embedding) of the 4th head of the 22nd (last) layer of ModernBERT-base (indices `3` and `21` in `train_modernbert.py`, which are 0-based), during fine-tuning on the SCOTUS subset of LexGLUE. The parameter $p$ is the training step in the last (fourth) epoch, with 563 steps; the first 50 steps are used as snapshots. To reproduce:

1. Run `train_modernbert.py` (requires `torch`, `transformers`, `datasets`, `evaluate`, `scipy`). It fine-tunes the model and saves the projected query/key matrices for a fixed 5000-token document at every step in `matrici_matlab_dirette_rope/`.
2. Move that folder to `python/` and run `run_experiment_svd`.

## Requirements

MATLAB with the Statistics and Machine Learning Toolbox (`randsample`, `knnsearch`). Python 3 with the packages listed above for the fine-tuning step of Example 6.
