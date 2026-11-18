import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))   # Utilities.py is in the repository root
import random
import argparse
import numpy as np
import pandas as pd
from Utilities import RLS, ALS, nuclear_norm, nuclear_norm_random, csmc, er_recover
from data_splits import load_movielens_1m, DATA_DIR, RESULTS_DIR, ROUND_STEP

N_RUNS = 20
K_VALUES = [17, 25, 35]
LAMBDA_RLS = 45000                       # RLS: regularisation (random Gaussian imputation)
LAMBDA_ALS, ITER_ALS = 23, 30            # ALS: regularisation, maximum number of sweeps
EPSILON, ITER_NUNO = 10, 30              # NuNo: singular-value threshold, maximum number of iterations
ITER_NUNOR = 20                          # NuNoR: maximum number of iterations (same threshold as NuNo)
COL_CSMC, TAU_CSMC = 0.2, 10             # CSMC: fraction of sampled columns, threshold of its nuclear-norm stage
METHODS = ['RLS', 'ALS', 'NuNo', 'NuNoR', 'CSMC']

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default=DATA_DIR)
    ap.add_argument('--quick', action='store_true', help='one rank, two runs, two iterations: smoke test only')
    args = ap.parse_args()
    if args.quick:
        N_RUNS, K_VALUES, ITER_ALS, ITER_NUNO, ITER_NUNOR = 2, [17], 2, 2, 2

    M_fit, M_test = load_movielens_1m(args.data_dir)
    raw = []

    def record(method, k, run, recovery, time_ms):
        mse, er = er_recover(recovery, M_test)
        raw.append(dict(k=k, Method=method, run=run, seed=run * 10, MSE=mse, Error=er, Time_ms=time_ms))

    # NuNo does not depend on k: run it once per seed and report the same values at every k.
    # It computes a full SVD of the 6040 x 3952 matrix at every iteration and takes several minutes per run.
    for run in range(N_RUNS):
        recovery, t = nuclear_norm(M_fit, EPSILON, ITER_NUNO, seednum=run * 10, round_step=ROUND_STEP)
        for k in K_VALUES:
            record('NuNo', k, run, recovery, t)
        print(f"  NuNo run {run + 1}/{N_RUNS} done", flush=True)

    for k in K_VALUES:
        for run in range(N_RUNS):
            seed = run * 10
            record('RLS', k, run, *RLS(M_fit, k, LAMBDA_RLS, seed, impute_mode='random', round_step=ROUND_STEP))
            record('ALS', k, run, *ALS(M_fit, k, LAMBDA_ALS, ITER_ALS, seed, round_step=ROUND_STEP))
            record('NuNoR', k, run, *nuclear_norm_random(M_fit, k, EPSILON, ITER_NUNOR, seed, round_step=ROUND_STEP))
            random.seed(seed)            # csmc() samples its columns with Python's random module
            record('CSMC', k, run, *csmc(M_fit, COL_CSMC, k, TAU_CSMC, ITER_NUNO, LAMBDA_ALS, seed, round_step=ROUND_STEP))
        d = pd.DataFrame([r for r in raw if r['k'] == k]).groupby('Method')[['MSE', 'Error', 'Time_ms']].mean()
        print(f"  k={k:3d}  " + '  '.join(f"{m}: {d.loc[m, 'MSE']:.4f} / {d.loc[m, 'Error']:.4f} / {d.loc[m, 'Time_ms']:.0f} ms" for m in METHODS), flush=True)

    raw = pd.DataFrame(raw)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    raw.to_csv(os.path.join(RESULTS_DIR, 'results_raw.csv'), index=False)
    std = lambda x: x.std(ddof=0)
    summary = (raw.groupby(['k', 'Method'])
               .agg(MSE_mean=('MSE', 'mean'), MSE_std=('MSE', std), Error_mean=('Error', 'mean'), Error_std=('Error', std),
                    Time_mean=('Time_ms', 'mean'), Time_std=('Time_ms', std)).reset_index())
    summary.to_csv(os.path.join(RESULTS_DIR, 'results.csv'), index=False)
    print("saved results/results.csv and results/results_raw.csv")
