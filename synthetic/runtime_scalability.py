import os
import sys
import time
import numpy as np
import pandas as pd
from common import (timed_rls, timed_als, timed_csmc, mse, validation_split, select_lambda, ci95_halfwidth, loglog_fit,
                    save_environment, output_dirs, TIE_TOL)

QUICK = '--quick' in sys.argv
RESULTS_DIR, _ = output_dirs('runtime_scalability', QUICK)

TRUE_RANK = K = 20
MATRIX_SIZES = [(500, 400), (1000, 800), (2000, 1600), (3000, 2400), (5000, 4000), (7500, 6000)]
TRAIN_DENSITY, TEST_DENSITY = 0.10, 0.02
RUN_SEEDS = [10 * r for r in range(20)]            # 20 runs: seeds 0, 10, ..., 190
ALS_ITERATIONS, ALS_TOL = 10, 0.0
CSMC_ITERATIONS, CSMC_TOL, CSMC_COL_PERCENT = 10, 0.0, 0.10
REFERENCE_SIZE, TUNING_SEEDS = (1000, 800), [0, 10, 20]
RLS_LAMBDAS, RLS_EXPAND_UP, RLS_EXPAND_DOWN = [0.1, 1, 10, 100, 300, 1000, 3000, 10000, 30000, 100000], [300000, 1000000, 3000000], [0.01, 0.001]
ALS_LAMBDAS, ALS_EXPAND_UP, ALS_EXPAND_DOWN = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100], [300, 1000, 3000], [1e-5, 1e-6]
CSMC_EPSILONS, CSMC_EPS_EXPAND_UP, CSMC_EPS_EXPAND_DOWN = [0.01, 0.1, 1, 10, 100], [1000, 10000], [0.001, 0.0001]
CSMC_LAMBDAS, CSMC_LAM_EXPAND_UP, CSMC_LAM_EXPAND_DOWN = [0.01, 0.1, 1, 10, 100, 1000], [10000, 100000], [0.001, 0.0001]
METHODS = ['RLS', 'CSMC', 'ALS']
if QUICK:
    MATRIX_SIZES, RUN_SEEDS, ALS_ITERATIONS, CSMC_ITERATIONS, TUNING_SEEDS, REFERENCE_SIZE = [(200, 160), (400, 320)], [0], 3, 3, [0], (200, 160)
    RLS_LAMBDAS, ALS_LAMBDAS, CSMC_EPSILONS, CSMC_LAMBDAS = [1, 100, 10000], [0.001, 0.1, 10], [0.1, 10], [0.1, 10]
    RLS_EXPAND_UP = RLS_EXPAND_DOWN = ALS_EXPAND_UP = ALS_EXPAND_DOWN = []
    CSMC_EPS_EXPAND_UP = CSMC_EPS_EXPAND_DOWN = CSMC_LAM_EXPAND_UP = CSMC_LAM_EXPAND_DOWN = []


def make_problem(m, n, seed):
    """Seeded rank-20 problem with values in [0.5, 5]. Returns (train, test): 0 = not observed."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, size=(m, TRUE_RANK)) @ rng.uniform(0, 1, size=(TRUE_RANK, n))
    X = 0.5 + 4.5 * (X - X.min()) / (X.max() - X.min())
    perm = rng.permutation(m * n)
    n_train, n_test = int(round(TRAIN_DENSITY * m * n)), int(round(TEST_DENSITY * m * n))
    train = np.zeros((m, n))
    train.flat[perm[:n_train]] = X.flat[perm[:n_train]]
    test = np.zeros((m, n))
    test.flat[perm[n_train:n_train + n_test]] = X.flat[perm[n_train:n_train + n_test]]
    return train, test


def rls(train, lam, seed):
    return timed_rls(train, K, lam, seed)


def als(train, lam, seed):
    return timed_als(train, K, lam, ALS_ITERATIONS, ALS_TOL, seed)


def csmc(train, epsilon, lam, seed):
    return timed_csmc(train, CSMC_COL_PERCENT, K, epsilon, CSMC_ITERATIONS, lam, CSMC_TOL, seed)


def select_csmc(splits):
    """CSMC (epsilon, lambda): full grid, mean validation MSE over the tuning seeds. Each grid dimension is
    extended once if the best value lies on its edge. Ties go to the smaller epsilon, then the smaller lambda."""
    grid = {}

    def evaluate(eps_list, lam_list):
        for eps in eps_list:
            for lam in lam_list:
                if (eps, lam) not in grid:
                    v = [mse(csmc(fit, eps, lam, seed)[0], val) for seed, (fit, val) in splits.items()]
                    grid[(eps, lam)] = float(np.mean(v))
                    print(f"    CSMC epsilon={eps:<7g} lambda={lam:<7g} validation MSE {np.mean(v):.6f}", flush=True)

    def best():
        low = min(grid.values())
        return min(key for key, v in grid.items() if v <= low + TIE_TOL), low

    eps_all, lam_all, expanded = list(CSMC_EPSILONS), list(CSMC_LAMBDAS), []
    evaluate(eps_all, lam_all)
    (eps, lam), _ = best()
    extra = CSMC_EPS_EXPAND_UP if eps == max(eps_all) else CSMC_EPS_EXPAND_DOWN if eps == min(eps_all) else []
    if len(extra):
        expanded.append(f'epsilon {extra}')
        eps_all += extra
        evaluate(extra, lam_all)
    (eps, lam), _ = best()
    extra = CSMC_LAM_EXPAND_UP if lam == max(lam_all) else CSMC_LAM_EXPAND_DOWN if lam == min(lam_all) else []
    if len(extra):
        expanded.append(f'lambda {extra}')
        lam_all += extra
        evaluate(eps_all, extra)
    (eps, lam), low = best()
    print(f"  CSMC selected epsilon = {eps:g}, lambda = {lam:g} (validation MSE {low:.6f}, grid expanded: {'; '.join(expanded) or 'no'})")
    rows = [dict(method='CSMC', epsilon=float(e), lam=float(l), val_MSE=v) for (e, l), v in grid.items()]
    return eps, lam, low, '; '.join(expanded) or 'no', rows


if __name__ == '__main__':
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_environment(os.path.join(RESULTS_DIR, 'environment.json'))
    t0 = time.perf_counter()

    # warm-up calls, not recorded
    warm, _ = make_problem(100, 80, 999)
    rls(warm, 1.0, 0)
    als(warm, 1.0, 0)
    csmc(warm, 1.0, 1.0, 0)

    # ---- step 1: hyper-parameters, selected on the reference problem's observed entries ----
    print("Step 1: hyper-parameter selection on the reference problem")
    splits = {s: validation_split(make_problem(*REFERENCE_SIZE, s)[0]) for s in TUNING_SEEDS}
    lam_rls, v_rls, exp_rls, rows_rls = select_lambda(splits, lambda f, lam, s: rls(f, lam, s)[0], RLS_LAMBDAS, RLS_EXPAND_UP, RLS_EXPAND_DOWN, 'RLS')
    lam_als, v_als, exp_als, rows_als = select_lambda(splits, lambda f, lam, s: als(f, lam, s)[0], ALS_LAMBDAS, ALS_EXPAND_UP, ALS_EXPAND_DOWN, 'ALS')
    eps_csmc, lam_csmc, v_csmc, exp_csmc, rows_csmc = select_csmc(splits)
    del splits
    pd.DataFrame([dict(method='RLS', **r) for r in rows_rls] + [dict(method='ALS', **r) for r in rows_als]).to_csv(
        os.path.join(RESULTS_DIR, 'lambda_validation.csv'), index=False)
    pd.DataFrame(rows_csmc).to_csv(os.path.join(RESULTS_DIR, 'csmc_validation_grid.csv'), index=False)
    pd.DataFrame([dict(method='RLS', lam=lam_rls, epsilon=np.nan, validation_MSE=v_rls, grid_expanded=exp_rls),
                  dict(method='CSMC', lam=lam_csmc, epsilon=eps_csmc, validation_MSE=v_csmc, grid_expanded=exp_csmc),
                  dict(method='ALS', lam=lam_als, epsilon=np.nan, validation_MSE=v_als, grid_expanded=exp_als)]).to_csv(
        os.path.join(RESULTS_DIR, 'selected_hyperparameters.csv'), index=False)

    # ---- step 2: timed runs (RLS, then CSMC, then ALS, on the same observed matrix) ----
    print("Step 2: timed runs")
    raw = []
    for (m, n) in MATRIX_SIZES:
        for seed in RUN_SEEDS:
            train, test = make_problem(m, n, seed)
            info = dict(m=m, n=n, entries=m * n, seed=seed, train_density=float((train > 0).mean()))
            rec, ext_r, int_r = rls(train, lam_rls, seed)
            raw.append(dict(**info, method='RLS', lam=lam_rls, epsilon=np.nan, iterations=1, time_ms=ext_r, internal_time_ms=int_r, sanity_test_MSE=mse(rec, test)))
            rec, ext_c, int_c, it_c = csmc(train, eps_csmc, lam_csmc, seed)
            raw.append(dict(**info, method='CSMC', lam=lam_csmc, epsilon=eps_csmc, iterations=it_c, time_ms=ext_c, internal_time_ms=int_c, sanity_test_MSE=mse(rec, test)))
            rec, ext_a, int_a, it_a = als(train, lam_als, seed)
            raw.append(dict(**info, method='ALS', lam=lam_als, epsilon=np.nan, iterations=it_a, time_ms=ext_a, internal_time_ms=int_a, sanity_test_MSE=mse(rec, test)))
            print(f"  {m:4d}x{n:<4d} seed {seed:2d}: RLS {ext_r:8.1f} ms | CSMC {ext_c:9.1f} ms ({it_c} it.) | ALS {ext_a:9.1f} ms ({it_a} sweeps) | "
                  f"CSMC/RLS {ext_c / ext_r:5.1f}x  ALS/RLS {ext_a / ext_r:5.1f}x", flush=True)
            del train, test, rec
    raw = pd.DataFrame(raw)
    raw.to_csv(os.path.join(RESULTS_DIR, 'raw.csv'), index=False)
    # the comparison is only meaningful if the iterative work really was fixed
    assert (raw[raw.method == 'ALS'].iterations == ALS_ITERATIONS).all(), "ALS stopped early"
    assert (raw[raw.method == 'CSMC'].iterations == CSMC_ITERATIONS).all(), "CSMC stopped early"

    # ---- summary, speed-up, empirical scaling ----
    summary = []
    for (method, m, n), d in raw.groupby(['method', 'm', 'n']):
        summary.append(dict(method=method, m=m, n=n, entries=m * n, n_runs=len(d), lam=d.lam.iloc[0], epsilon=d.epsilon.iloc[0],
                            iterations=d.iterations.mean(), time_mean_ms=d.time_ms.mean(), time_std_ms=d.time_ms.std(ddof=1),
                            time_ci95_ms=ci95_halfwidth(d.time_ms), time_median_ms=d.time_ms.median(),
                            internal_time_mean_ms=d.internal_time_ms.mean(), sanity_test_MSE_mean=d.sanity_test_MSE.mean()))
    summary = pd.DataFrame(summary)
    summary = summary.sort_values(['entries', 'method'], key=lambda c: c.map(METHODS.index) if c.name == 'method' else c)
    summary.to_csv(os.path.join(RESULTS_DIR, 'summary.csv'), index=False)

    t = summary.pivot(index=['m', 'n', 'entries'], columns='method', values='time_mean_ms').sort_index(level='entries')
    speedup = pd.DataFrame(dict(RLS_time_ms=t.RLS, CSMC_time_ms=t.CSMC, ALS_time_ms=t.ALS, CSMC_over_RLS=t.CSMC / t.RLS, ALS_over_RLS=t.ALS / t.RLS)).reset_index()
    speedup.to_csv(os.path.join(RESULTS_DIR, 'speedup.csv'), index=False)

    fit = pd.DataFrame([dict(method=x, **dict(zip(('slope', 'intercept', 'r_squared'), loglog_fit(*summary[summary.method == x][['entries', 'time_mean_ms']].values.T))))
                        for x in METHODS])
    fit.to_csv(os.path.join(RESULTS_DIR, 'fit.csv'), index=False)

    # ---- LaTeX tables: compact one-column version and a wide version ----
    lines = [r"\begin{table}[t]", r"\centering", r"\caption{Runtime scalability at fixed rank, observation density, and", r"iterative budgets.}",
             r"\label{tab:runtime_scalability}", r"\setlength{\tabcolsep}{3.5pt}", r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"\textbf{Size} &", r"\textbf{RLS} &", r"\textbf{CSMC} &", r"\textbf{ALS} &", r"\textbf{C/R} &", r"\textbf{A/R} \\",
             r"& \multicolumn{3}{c}{\textbf{Time (ms)}} &", r"\multicolumn{2}{c}{\textbf{Speedup}} \\", r"\midrule"]
    wide = [r"\begin{table*}[t]", r"\centering", r"\caption{Runtime scalability at fixed rank, observation density, and iteration count.}",
            r"\label{tab:runtime_scalability_wide}", r"\begin{tabular}{lrrrrrr}", r"\toprule",
            r"\textbf{Matrix Size} & \textbf{Entries} & \textbf{RLS (ms)} & \textbf{CSMC (ms)} & \textbf{ALS (ms)} & \textbf{CSMC/RLS} & \textbf{ALS/RLS} \\", r"\midrule"]
    for s in speedup.itertuples():
        lines.append(f"${s.m}{{\\times}}{s.n}$ & {s.RLS_time_ms:.0f} & {s.CSMC_time_ms:.0f} & {s.ALS_time_ms:.0f} & {s.CSMC_over_RLS:.1f} & {s.ALS_over_RLS:.1f} \\\\")
        wide.append(f"${s.m}\\times{s.n}$ & {s.entries / 1e6:.1f}M & {s.RLS_time_ms:.0f} & {s.CSMC_time_ms:.0f} & {s.ALS_time_ms:.0f} "
                    f"& {s.CSMC_over_RLS:.1f}$\\times$ & {s.ALS_over_RLS:.1f}$\\times$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", "", r"\vspace{2pt}", r"\footnotesize",
              f"Mean wall-clock time over {len(RUN_SEEDS)} runs. $k={K}$ and {TRAIN_DENSITY:.0%} of entries are".replace('%', r'\%'),
              r"observed. C/R and A/R denote CSMC/RLS and ALS/RLS runtime ratios,",
              r"respectively; values above one indicate that RLS is faster.", r"\end{table}"]
    wide += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    for name, content in [('table.tex', lines), ('table_wide.tex', wide)]:
        with open(os.path.join(RESULTS_DIR, name), 'w') as f:
            f.write('\n'.join(content) + '\n')

    pd.set_option('display.width', 250)
    print("\n" + speedup.to_string(index=False, float_format=lambda v: f'{v:.4g}'))
    print("\n" + fit.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    print(f"\nRLS lambda {lam_rls:g}; CSMC epsilon {eps_csmc:g}, lambda {lam_csmc:g}; ALS lambda {lam_als:g}; "
          f"results in {RESULTS_DIR}; total {(time.perf_counter() - t0) / 60:.1f} min")
