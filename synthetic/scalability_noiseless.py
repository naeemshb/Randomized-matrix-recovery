import os
import sys
import time
import numpy as np
import pandas as pd
from common import (timed_rls, timed_als, mse, er_recover, validation_split, select_lambda, ci95_halfwidth,
                    loglog_fit, save_environment, output_dirs)

QUICK = '--quick' in sys.argv
RESULTS_DIR, _ = output_dirs('scalability_noiseless', QUICK)

TRUE_RANK = K = 20
MATRIX_SIZES = [(500, 400), (1000, 800), (1500, 1200), (2000, 1600), (3000, 2400)]
TRAIN_DENSITY, TEST_DENSITY = 0.10, 0.02
RUN_SEEDS = [10 * r for r in range(20)]            # 20 runs: seeds 0, 10, ..., 190
REFERENCE_SIZE, TUNING_SEEDS = (1000, 800), [0, 10, 20]
RLS_LAMBDAS, RLS_EXPAND_UP, RLS_EXPAND_DOWN = [0.1, 1, 10, 100, 300, 1000, 3000, 10000], [30000, 100000, 300000], [0.01, 0.001]
ALS_LAMBDAS, ALS_EXPAND_UP, ALS_EXPAND_DOWN = [0.001, 0.01, 0.1, 1, 10, 100], [300, 1000, 3000], [1e-4, 1e-5]
ALS_MAX_ITER, ALS_TOL = 20, 1e-4
ER_GRIDS = {'Error_grid0.1': 0.1, 'Error_grid0.5': 0.5}
if QUICK:
    MATRIX_SIZES, RUN_SEEDS, TUNING_SEEDS, REFERENCE_SIZE = [(200, 160), (400, 320)], [0], [0], (200, 160)
    RLS_LAMBDAS, ALS_LAMBDAS = [1, 100, 10000], [0.01, 1, 100]
    RLS_EXPAND_UP = RLS_EXPAND_DOWN = ALS_EXPAND_UP = ALS_EXPAND_DOWN = []


def make_problem(m, n, seed):
    """Seeded rank-20 problem. Returns (train, test): dense matrices, 0 = not observed, disjoint positions."""
    rng = np.random.default_rng(seed)
    U = rng.uniform(0, 1, size=(m, TRUE_RANK))
    V = rng.uniform(0, 1, size=(TRUE_RANK, n))
    X = U @ V
    perm = rng.permutation(m * n)
    n_train, n_test = int(TRAIN_DENSITY * m * n), int(TEST_DENSITY * m * n)
    train = np.zeros((m, n))
    train.flat[perm[:n_train]] = X.flat[perm[:n_train]]
    test = np.zeros((m, n))
    test.flat[perm[n_train:n_train + n_test]] = X.flat[perm[n_train:n_train + n_test]]
    assert not np.any((train > 0) & (test > 0))
    return train, test


def rls(train, lam, seed):
    return timed_rls(train, K, lam, seed)


def als(train, lam, seed):
    return timed_als(train, K, lam, ALS_MAX_ITER, ALS_TOL, seed)


if __name__ == '__main__':
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_environment(os.path.join(RESULTS_DIR, 'environment.json'))
    t0 = time.perf_counter()

    # warm-up calls, not recorded
    warm, _ = make_problem(100, 80, 999)
    rls(warm, 1.0, 0)
    als(warm, 1.0, 0)

    # ---- step 1: lambdas, selected on the reference problem's training observations ----
    print("Step 1: lambda selection on the reference problem")
    splits = {s: validation_split(make_problem(*REFERENCE_SIZE, s)[0]) for s in TUNING_SEEDS}
    lam_rls, v_rls, exp_rls, rows_rls = select_lambda(splits, lambda f, lam, s: rls(f, lam, s)[0], RLS_LAMBDAS, RLS_EXPAND_UP, RLS_EXPAND_DOWN, 'RLS')
    lam_als, v_als, exp_als, rows_als = select_lambda(splits, lambda f, lam, s: als(f, lam, s)[0], ALS_LAMBDAS, ALS_EXPAND_UP, ALS_EXPAND_DOWN, 'ALS')
    del splits
    pd.DataFrame([dict(method='RLS', **r) for r in rows_rls] + [dict(method='ALS', **r) for r in rows_als]).to_csv(
        os.path.join(RESULTS_DIR, 'lambda_validation.csv'), index=False)
    pd.DataFrame([dict(method='RLS', selected_lambda=lam_rls, validation_MSE=v_rls, grid_expanded=exp_rls),
                  dict(method='ALS', selected_lambda=lam_als, validation_MSE=v_als, grid_expanded=exp_als)]).to_csv(
        os.path.join(RESULTS_DIR, 'selected_lambdas.csv'), index=False)

    # ---- step 2: timed runs ----
    print("Step 2: timed runs")
    raw = []
    for (m, n) in MATRIX_SIZES:
        for seed in RUN_SEEDS:
            train, test = make_problem(m, n, seed)
            info = dict(m=m, n=n, entries=m * n, seed=seed, train_density=float((train > 0).mean()), test_density=float((test > 0).mean()))
            rec, ext_r, int_r = rls(train, lam_rls, seed)
            raw.append(dict(method='RLS', **info, lam=lam_rls, iterations=1, MSE=mse(rec, test),
                            **{c: er_recover(rec, test, round_step=g)[1] for c, g in ER_GRIDS.items()}, time_ms=ext_r, internal_time_ms=int_r))
            rec, ext_a, int_a, sweeps = als(train, lam_als, seed)
            raw.append(dict(method='ALS', **info, lam=lam_als, iterations=sweeps, MSE=mse(rec, test),
                            **{c: er_recover(rec, test, round_step=g)[1] for c, g in ER_GRIDS.items()}, time_ms=ext_a, internal_time_ms=int_a))
            print(f"  {m:4d}x{n:<4d} seed {seed:2d}: RLS {ext_r:8.1f} ms  MSE {raw[-2]['MSE']:.3e} | "
                  f"ALS {ext_a:8.1f} ms  {sweeps:2d} sweeps  MSE {raw[-1]['MSE']:.3e}", flush=True)
            del train, test, rec
    raw = pd.DataFrame(raw)
    raw.to_csv(os.path.join(RESULTS_DIR, 'raw.csv'), index=False)

    # ---- summary, speed-up, empirical scaling ----
    summary = []
    for (method, m, n), d in raw.groupby(['method', 'm', 'n']):
        summary.append(dict(method=method, m=m, n=n, entries=m * n, n_runs=len(d), lam=d.lam.iloc[0], iterations_mean=d.iterations.mean(),
                            time_mean_ms=d.time_ms.mean(), time_std_ms=d.time_ms.std(ddof=1), time_ci95_ms=ci95_halfwidth(d.time_ms),
                            internal_time_mean_ms=d.internal_time_ms.mean(),
                            MSE_mean=d.MSE.mean(), MSE_std=d.MSE.std(ddof=1), MSE_ci95=ci95_halfwidth(d.MSE),
                            **{f'{c}_mean': d[c].mean() for c in ER_GRIDS}))
    summary = pd.DataFrame(summary).sort_values(['entries', 'method'], ascending=[True, False])
    summary.to_csv(os.path.join(RESULTS_DIR, 'summary.csv'), index=False)

    r, a = (summary[summary.method == x].set_index(['m', 'n', 'entries']) for x in ('RLS', 'ALS'))
    speedup = pd.DataFrame(dict(RLS_time_ms=r.time_mean_ms, ALS_time_ms=a.time_mean_ms, speedup_ALS_over_RLS=a.time_mean_ms / r.time_mean_ms,
                                RLS_MSE=r.MSE_mean, ALS_MSE=a.MSE_mean, RLS_Error=r['Error_grid0.1_mean'], ALS_Error=a['Error_grid0.1_mean'],
                                ALS_sweeps=a.iterations_mean)).reset_index()
    speedup.to_csv(os.path.join(RESULTS_DIR, 'speedup.csv'), index=False)

    fit = pd.DataFrame([dict(method=x, **dict(zip(('slope', 'intercept', 'r_squared'), loglog_fit(d.entries, d.time_mean_ms))))
                        for x, d in summary.groupby('method', sort=False)])
    fit.to_csv(os.path.join(RESULTS_DIR, 'fit.csv'), index=False)

    # ---- LaTeX table ----
    def fmt(v):
        return f"{v:.4f}" if v >= 0.01 else f"{v:.1e}"
    lines = [r"\begin{table}[t]", r"\centering", r"\caption{Runtime scalability of RLS and ALS at fixed rank and observation density.}",
             r"\label{tab:scalability}", r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
             r"Matrix size & Entries & RLS time (ms) & ALS time (ms) & Speedup & RLS MSE & ALS MSE & RLS ER & ALS ER \\", r"\midrule"]
    for s in speedup.itertuples():
        lines.append(f"${s.m}\\times{s.n}$ & {s.entries:,} & {s.RLS_time_ms:.0f} & {s.ALS_time_ms:.0f} & {s.speedup_ALS_over_RLS:.1f}$\\times$ "
                     f"& {fmt(s.RLS_MSE)} & {fmt(s.ALS_MSE)} & {s.RLS_Error:.3f} & {s.ALS_Error:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(RESULTS_DIR, 'table.tex'), 'w') as f:
        f.write('\n'.join(lines) + '\n')

    pd.set_option('display.width', 250)
    print("\n" + speedup.to_string(index=False, float_format=lambda v: f'{v:.4g}'))
    print("\n" + fit.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    print(f"\nRLS lambda {lam_rls:g}, ALS lambda {lam_als:g}; results in {RESULTS_DIR}; total {(time.perf_counter() - t0) / 60:.1f} min")
