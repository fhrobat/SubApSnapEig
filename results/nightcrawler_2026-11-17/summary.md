# Results reported in the paper

Machine, MATLAB version and settings: see `machine.txt`. The log of the whole run is `run_all.log`; for every example, `run.log` is the MATLAB output and `load.log` the load average of the machine during the run.

`LU` and `Leverage` times are the median over 10 runs (`Leverage` with seeds 42, ..., 51); `MOR` is run once. Speed-up = online time of `MOR` / online time of the method.

| Example | | Offline (s) | Online (s) | Online speed-up vs MOR |
|---|---|---:|---:|---:|
| 1 (exponential) | LU | 10.09 | 0.083 | 6044x |
| 1 (exponential) | Leverage | 9.19 | 0.111 | 4523x |
| 1 (exponential) | MOR | 7.92 | 501.102 |  |
| 2 (differential) | LU | 75.21 | 0.111 | 1006x |
| 2 (differential) | Leverage | 73.84 | 0.278 | 403x |
| 2 (differential) | MOR | 73.13 | 111.918 |  |
| 3 (pagerank) | LU | 9.86 | 3.714 | 7x |
| 3 (pagerank) | Leverage | 9.94 | 5.386 | 5x |
| 3 (pagerank) | MOR | 10.09 | 25.159 |  |
| 4 (kernel) | LU | 4.32 | 0.146 | 1820x |
| 4 (kernel) | Leverage | 4.32 | 0.309 | 858x |
| 4 (kernel) | MOR | 4.29 | 265.532 |  |
| 5 (graph) | LU | 6.45 | 2.206 | 12x |
| 5 (graph) | Leverage | 5.63 | 2.204 | 12x |
| 5 (graph) | MOR | 5.79 | 27.253 |  |
| 6 (transformer) | LU | 15.21 | 1.550 | 56x |
| 6 (transformer) | Leverage | 15.21 | 3.277 | 27x |
| 6 (transformer) | MOR | 14.77 | 87.238 |  |

The error files (`res.dat`, `eigval.dat`, `eigvec.dat`) contain, for every test parameter, the error of `LU`, the median error of `Leverage`, the error of `MOR` (columns `*_POD`) and the 95% confidence interval for the median of `Leverage` (`*_Leverage_lo`, `*_Leverage_hi`). See the help of `run_experiment.m`.

`theory/1000_<t_out>.tex` are the data of Figures 1-3, written by `theory/theory.m` (`./run_all.sh theory`).
