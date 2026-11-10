# SubApSnapEig

MATLAB code accompanying the paper

> F. Hrobat and Y. Nakatsukasa, *Solving a sequence of eigenvalue problems using snapshot and row subsampling*.

**SubApSnapEig** solves a sequence of eigenvalue problems $A(p)\,x(p) = \lambda(p)\,x(p)$ for many values of a parameter $p$. It combines a snapshot (reduced-order) subspace with row subsampling.

- **Offline phase.** Compute eigenvectors at a few snapshot points, form (and orthogonalize) the snapshot matrix $X$, then select a set of rows $S$ from $X$ (LU with partial pivoting or leverage-score sampling).
- **Online phase.** For each new parameter $p_*$, solve the small subsampled least-squares problem $\min_M \|S(A(p_*)X - XM)\|_F$ and compute the eigenpairs of $M$. Only the selected rows of $A(p_*)$ are needed.

The method is compared with `MOR`, which uses the same snapshot subspace with a full (non-subsampled) Rayleigh–Ritz projection.

## Repository structure

| Path | Content |
|---|---|
| `SubApSnapEig/SubApSnap_eig.m`, `SubApSnapEig/SubApSnap_SVD.m` | The proposed method, for eigenvalue problems and for the SVD |
| `MOR/ApSnap_eig.m`, `MOR/ApSnap_SVD.m` | Baseline `MOR` method (snapshots + full Rayleigh–Ritz) |
| `utils/` | Row-sampling strategies (`lu_sampling.m`, `leverage_score_sampling.m`), `eigs`/`svds` wrappers, and full/row-wise matrix constructors for each test problem |
| `prova_subapsnap_eig.m` | Driver script for the eigenvalue experiments (Examples 1–5) |
| `prova_subapsnap_svd.m` | Driver script for the singular value experiment (Example 6) |
| `train_modernbert.py` | Fine-tunes ModernBERT and extracts the attention matrices used in Example 6 |
| `theory/theory.m` | Experiments illustrating the theoretical bounds (singular value decay and error bounds) |

## Experiments

Each experiment in `prova_subapsnap_eig.m` has its own setup block. To run one, uncomment its block and comment out the others. The three methods compared are `SubApSnapEig` with LU sampling (`LU`), `SubApSnapEig` with leverage-score sampling (`Leverage`), and `MOR`. The script reports the residual, the eigenvalue and eigenvector errors, and the offline/online timings.

**Theory (`theory/theory.m`).** $A(p) = M_1 + p\,M_2$, with $M_1, M_2$ the symmetric parts of two $1000\times 1000$ random matrices, and $p \in [-1,1]$. The script compares equispaced and Chebyshev snapshot points: the decay of the smallest singular value of the snapshot matrix, and the error $\|(I - XX^\dagger)x(p_*)\|_2$ against the non-asymptotic and asymptotic bounds, for $p_*$ inside and outside $[-1,1]$.

**Example 1: Exponential differential operator.** Smallest eigenvalue of the 3-point finite-difference discretization of $\frac{d}{dx}\big(e^{px}\frac{du}{dx}\big)$ with zero boundary conditions (a Sturm–Liouville problem). $n = 2\times 10^6$; 3000 log-spaced parameters in $[0.1, 1]$; 10 log-spaced snapshot points.

**Example 2: Differential operator with multiple eigenvalues.** The 3 smallest eigenvalues of the 5-point finite-difference discretization of $-p\,\Delta u - \nabla(\nabla\cdot u)$ on $[-1,1]^2$ with zero boundary conditions. The second smallest eigenvalue has multiplicity 2. $n = 250\,000$; log-spaced parameters; 10 log-spaced snapshot points.

**Example 3: PageRank.** Leading eigenpair of a degree-biased parametric PageRank matrix on the `web-NotreDame` graph ($n = 325\,729$, included in `utils/pagerank/`). The damping factor is $d = 0.9$, and link weights are scaled by $(1 + \text{in-degree}_i)^p$. 1000 equispaced parameters in $[0, 0.5]$; 5 equispaced snapshot points. Products with the dense $A(p)$ are computed implicitly.

**Example 4: Gaussian kernel.** Leading eigenpair of the Gaussian kernel matrix $A_{ij}(p) = \exp(-\|x_i - x_j\|^2 / 2p^2)$ built from the MNIST training set ($n = 60\,000$). 1000 equispaced parameters in $[0.5\,p_*, 1.5\,p_*]$, where $p_*$ is the median pairwise distance; 10 equispaced snapshot points. The MNIST data (`mnist_completo.mat`, variable `X`) is not included in the repository.

**Example 5: Graph eigenvector centrality.** Leading eigenpair of $A(p) = D(p)\,A_0\,D(p)$, where $A_0$ is the adjacency matrix of `coAuthorsDBLP` ($n = 299\,067$, included in `utils/`). $D(p)$ is diagonal with entries $1 + 0.5\sin(2\pi p/1000 + \Phi_i)$ and random phases $\Phi_i$. Parameters $p = 1,\dots,1000$; 20 snapshot points at the start of the parameter range. This is the time-dependent case: the snapshot matrix and the row selection are updated every 100 steps.

**Example 6: Transformer attention (SVD).** Leading singular value and vector of the attention matrix $\mathrm{softmax}\big(XW_Q (XW_K)^T/\sqrt{64}\big)$ (layer 21, head 3) of ModernBERT-base, during fine-tuning on the SCOTUS subset of LexGLUE. The parameter $p$ is the training step in the last (fourth) epoch, with 563 steps; the first 50 steps are used as snapshots. To reproduce:

1. Run `train_modernbert.py` (requires `torch`, `transformers`, `datasets`, `evaluate`, `scipy`). It fine-tunes the model and saves the projected query/key matrices for a fixed 5000-token document at every step.
2. Run `prova_subapsnap_svd.m`, which loads these matrices (from `python/matrici_matlab_dirette_rope/`) and compares `SubApSnapEig` with `MOR`.

## Requirements

MATLAB (with `eigs`/`svds`) for all experiments. Python 3 with the packages listed above for the fine-tuning step of Example 6.
