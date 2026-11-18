# Randomized matrix recovery

Code for the paper *Randomized Least Square for Efficient Low-Rank Matrix Recovery*. Randomized Least Squares (RLS) recovers a low-rank matrix from a subset of its entries in a single pass, without iterations: the missing entries are first imputed, the imputed matrix is multiplied by a random Gaussian matrix to obtain a rank-k sketch, and each column is then fitted to its observed entries by a small ridge regression on that sketch. Because it needs only one matrix product and one k × k linear solve per column, RLS is much faster than iterative methods such as alternating least squares and nuclear-norm minimisation, which are included here as baselines. `Utilities.py` contains RLS and the baselines, and the folders contain the scripts for the experiments on real and synthetic data.

## Layout

```
Randomized-matrix-recovery/
├── README.md
├── Utilities.py          ← RLS, ALS, NuNo, NuNoR, CSMC, evaluation
├── requirements.txt
├── .gitignore
├── RLS/
│   ├── data_splits.py
│   ├── run_experiment.py
│   ├── plot_results.py
│   ├── imputation_vs_rls.py
│   └── plot_imputation_vs_rls.py
└── synthetic/
    ├── common.py
    ├── runtime_scalability.py
    ├── scalability_noiseless.py
    ├── scalability_noisy.py
    ├── plot_runtime_scalability.py
    ├── plot_scalability_noiseless.py
    └── plot_scalability_noisy.py
```
