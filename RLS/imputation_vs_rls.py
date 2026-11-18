import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))   # Utilities.py is in the repository root
import argparse
import numpy as np
import pandas as pd
from Utilities import RLS, er_recover, impute_missing_with_random
from data_splits import load_movielens_1m, DATA_DIR, RESULTS_DIR, ROUND_STEP

N_RUNS = 20
K_VALUES = [5, 7, 9, 11, 15, 20, 25, 30]
LAMBDA_RLS = 45000
IMPUTATIONS = ['mean', 'median', 'random']       # 'random' = random Gaussian imputation

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default=DATA_DIR)
    ap.add_argument('--quick', action='store_true', help='two ranks, two runs: smoke test only')
    args = ap.parse_args()
    if args.quick:
        N_RUNS, K_VALUES = 2, [5, 30]
    os.makedirs(RESULTS_DIR, exist_ok=True)

    train, test = load_movielens_1m(args.data_dir)
    mask = train > 0
    observed = train[mask]
    lo, hi = observed.min(), observed.max()

    def finish(X):
        """Same post-processing as RLS: clip to the observed range, round to the rating grid."""
        return np.round(np.clip(X, lo, hi) / ROUND_STEP) * ROUND_STEP

    # ---- imputation alone ----
    baselines = []
    for mode, value in [('mean', observed.mean()), ('median', np.median(observed))]:
        filled = train.copy()
        filled[~mask] = value
        mse, er = er_recover(finish(filled), test)
        baselines.append(dict(imputation=mode, MSE=mse, Error=er))
    scores = [er_recover(finish(impute_missing_with_random(train, random_state=run * 10)), test) for run in range(N_RUNS)]
    baselines.append(dict(imputation='random', MSE=np.mean([s[0] for s in scores]), Error=np.mean([s[1] for s in scores])))
    baselines = pd.DataFrame(baselines)
    baselines.to_csv(os.path.join(RESULTS_DIR, 'imputation_baselines.csv'), index=False)
    print(baselines.to_string(index=False))

    # ---- the same imputation followed by RLS ----
    rows = []
    for mode in IMPUTATIONS:
        for k in K_VALUES:
            for run in range(N_RUNS):
                seed = run * 10
                recovery, run_time = RLS(train, k, LAMBDA_RLS, seed, impute_mode=mode, round_step=ROUND_STEP)
                mse, er = er_recover(recovery, test)
                rows.append(dict(imputation=mode, k=k, run=run, seed=seed, lam=LAMBDA_RLS, MSE=mse, Error=er, Time_ms=run_time))
            last = pd.DataFrame(rows[-N_RUNS:])
            print(f"  {mode:6s} k={k:2d}  MSE={last.MSE.mean():.4f}±{last.MSE.std(ddof=0):.4f}  "
                  f"Error={last.Error.mean():.4f}  Time={last.Time_ms.mean():.0f} ms", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, 'imputation_vs_rls.csv'), index=False)
    print("saved results/imputation_baselines.csv and results/imputation_vs_rls.csv")
