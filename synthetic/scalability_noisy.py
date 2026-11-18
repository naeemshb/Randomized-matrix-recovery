import os
import sys
import time
import numpy as np
import pandas as pd
from common import (timed_rls, timed_als, mse, validation_split, select_lambda, ci95_halfwidth, save_environment, output_dirs)

QUICK = '--quick' in sys.argv
RESULTS_DIR, _ = output_dirs('scalability_noisy', QUICK)

SIGNAL_RANK = K = 20
NOISE_LEVELS = [0.10, 0.25, 0.50]
MATRIX_SIZES = [(500, 400), (1000, 800), (1500, 1200), (2000, 1600), (3000, 2400)]
TRAIN_DENSITY, TEST_DENSITY = 0.10, 0.02
RUN_SEEDS = [10 * r for r in range(20)]            # 20 runs: seeds 0, 10, ..., 190
REFERENCE_SIZE, TUNING_SEEDS = (1000, 800), [0, 10, 20]
RLS_LAMBDAS, RLS_EXPAND_UP, RLS_EXPAND_DOWN = [0.1, 1, 10, 100, 300, 1000, 3000, 10000, 30000, 100000, 300000], [1e6, 3e6, 1e7], [0.01, 0.001]
ALS_LAMBDAS, ALS_EXPAND_UP, ALS_EXPAND_DOWN = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100], [300, 1000, 3000], [1e-5, 1e-6]
ALS_MAX_ITER, ALS_TOL = 20, 1e-4
if QUICK:
    MATRIX_SIZES, RUN_SEEDS, NOISE_LEVELS, TUNING_SEEDS, REFERENCE_SIZE = [(200, 160), (400, 320)], [0], [0.10, 0.50], [0], (200, 160)
    RLS_LAMBDAS, ALS_LAMBDAS = [1, 100, 10000], [0.001, 0.1, 10]
    RLS_EXPAND_UP = RLS_EXPAND_DOWN = ALS_EXPAND_UP = ALS_EXPAND_DOWN = []


def make_base(m, n, seed):
    """One realisation shared by all noise levels: standardised rank-20 signal, noise matrix, mask positions."""
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((m, SIGNAL_RANK))
    V = rng.standard_normal((SIGNAL_RANK, n))
    S = U @ V
    S = (S - S.mean()) / S.std()
    Z = rng.standard_normal((m, n))
    perm = rng.permutation(m * n)
    n_train, n_test = int(TRAIN_DENSITY * m * n), int(TEST_DENSITY * m * n)
    return S, Z, perm[:n_train], perm[n_train:n_train + n_test]


def make_noisy(S, Z, train_idx, test_idx, sigma):
    """Returns (X, train, test) for one noise level."""
    X = np.clip(3.0 + 0.75 * (S + sigma * Z), 0.5, 5.0)
    train = np.zeros_like(X)
    train.flat[train_idx] = X.flat[train_idx]
    test = np.zeros_like(X)
    test.flat[test_idx] = X.flat[test_idx]
    assert not np.any((train > 0) & (test > 0))
    return X, train, test


def rls(train, lam, seed):
    return timed_rls(train, K, lam, seed)


def als(train, lam, seed):
    return timed_als(train, K, lam, ALS_MAX_ITER, ALS_TOL, seed)


def tag(sigma):
    return f"sigma{int(round(sigma * 100)):03d}"


if __name__ == '__main__':
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_environment(os.path.join(RESULTS_DIR, 'environment.json'))
    t0 = time.perf_counter()

    # warm-up calls, not recorded
    warm = make_noisy(*make_base(100, 80, 999), 0.1)[1]
    rls(warm, 1.0, 0)
    als(warm, 1.0, 0)

    # ---- how close to rank 20 is the full matrix? (description of the data only; not used by any method) ----
    rows = []
    S0, Z0, tr0, te0 = make_base(*REFERENCE_SIZE, 0)
    for sigma in NOISE_LEVELS:
        X0 = make_noisy(S0, Z0, tr0, te0, sigma)[0]
        sv = np.linalg.svd(X0, compute_uv=False)
        unclipped = 3.0 + 0.75 * (S0 + sigma * Z0)
        rows.append(dict(noise_sigma=sigma, reference_size=str(REFERENCE_SIZE), matrix_mean=X0.mean(), matrix_std=X0.std(),
                         clipped_fraction=float(np.mean((unclipped < 0.5) | (unclipped > 5.0))),
                         energy_in_top20_singular_values=float((sv[:SIGNAL_RANK] ** 2).sum() / (sv ** 2).sum()),
                         sv1=sv[0], sv20=sv[19], sv21=sv[20], sv40=sv[39]))
        print(f"  sigma={sigma}: mean {X0.mean():.3f}, std {X0.std():.3f}, clipped {100 * rows[-1]['clipped_fraction']:.2f}%, "
              f"energy in top-20 singular values {100 * rows[-1]['energy_in_top20_singular_values']:.1f}%")
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, 'matrix_characterisation.csv'), index=False)
    del S0, Z0, X0

    # ---- step 1: lambdas per noise level, selected on the reference problem's training observations ----
    print("Step 1: lambda selection per noise level on the reference problem")
    ref_base = {s: make_base(*REFERENCE_SIZE, s) for s in TUNING_SEEDS}
    LAM, sel_rows, val_rows = {}, [], []
    for sigma in NOISE_LEVELS:
        splits = {s: validation_split(make_noisy(*ref_base[s], sigma)[1]) for s in TUNING_SEEDS}
        lam_r, v_r, exp_r, rows_r = select_lambda(splits, lambda f, lam, s: rls(f, lam, s)[0], RLS_LAMBDAS, RLS_EXPAND_UP, RLS_EXPAND_DOWN, f'sigma={sigma} RLS')
        lam_a, v_a, exp_a, rows_a = select_lambda(splits, lambda f, lam, s: als(f, lam, s)[0], ALS_LAMBDAS, ALS_EXPAND_UP, ALS_EXPAND_DOWN, f'sigma={sigma} ALS')
        LAM[sigma] = dict(RLS=lam_r, ALS=lam_a)
        sel_rows += [dict(noise_sigma=sigma, method='RLS', selected_lambda=lam_r, validation_MSE=v_r, grid_expanded=exp_r),
                     dict(noise_sigma=sigma, method='ALS', selected_lambda=lam_a, validation_MSE=v_a, grid_expanded=exp_a)]
        val_rows += [dict(noise_sigma=sigma, method='RLS', **r) for r in rows_r] + [dict(noise_sigma=sigma, method='ALS', **r) for r in rows_a]
    del ref_base, splits
    pd.DataFrame(val_rows).to_csv(os.path.join(RESULTS_DIR, 'lambda_validation.csv'), index=False)
    pd.DataFrame(sel_rows).to_csv(os.path.join(RESULTS_DIR, 'selected_lambdas.csv'), index=False)

    # ---- step 2: timed runs ----
    print("Step 2: timed runs")
    raw = []
    for (m, n) in MATRIX_SIZES:
        for seed in RUN_SEEDS:
            S, Z, train_idx, test_idx = make_base(m, n, seed)
            for sigma in NOISE_LEVELS:
                X, train, test = make_noisy(S, Z, train_idx, test_idx, sigma)
                info = dict(noise_sigma=sigma, m=m, n=n, entries=m * n, seed=seed, train_density=float((train > 0).mean()),
                            test_density=float((test > 0).mean()), matrix_mean=float(X.mean()), matrix_std=float(X.std()))
                rec, ext_r, int_r = rls(train, LAM[sigma]['RLS'], seed)
                raw.append(dict(method='RLS', **info, lam=LAM[sigma]['RLS'], iterations=1, MSE=mse(rec, test), time_ms=ext_r, internal_time_ms=int_r))
                rec, ext_a, int_a, sweeps = als(train, LAM[sigma]['ALS'], seed)
                raw.append(dict(method='ALS', **info, lam=LAM[sigma]['ALS'], iterations=sweeps, MSE=mse(rec, test), time_ms=ext_a, internal_time_ms=int_a))
                print(f"  {m:4d}x{n:<4d} seed {seed:2d} sigma {sigma:.2f}: RLS {ext_r:8.1f} ms  MSE {raw[-2]['MSE']:.5f} | "
                      f"ALS {ext_a:8.1f} ms  {sweeps:2d} sweeps  MSE {raw[-1]['MSE']:.5f}", flush=True)
                del X, train, test, rec
            del S, Z
    raw = pd.DataFrame(raw)
    raw.to_csv(os.path.join(RESULTS_DIR, 'raw.csv'), index=False)

    # ---- summary and RLS/ALS comparison ----
    summary = []
    for (sigma, method, m, n), d in raw.groupby(['noise_sigma', 'method', 'm', 'n']):
        summary.append(dict(noise_sigma=sigma, method=method, m=m, n=n, entries=m * n, n_runs=len(d), lam=d.lam.iloc[0],
                            iterations_mean=d.iterations.mean(), time_mean_ms=d.time_ms.mean(), time_std_ms=d.time_ms.std(ddof=1),
                            time_ci95_ms=ci95_halfwidth(d.time_ms), internal_time_mean_ms=d.internal_time_ms.mean(),
                            MSE_mean=d.MSE.mean(), MSE_std=d.MSE.std(ddof=1), MSE_ci95=ci95_halfwidth(d.MSE)))
    summary = pd.DataFrame(summary).sort_values(['noise_sigma', 'entries', 'method'], ascending=[True, True, False])
    summary.to_csv(os.path.join(RESULTS_DIR, 'summary.csv'), index=False)

    keys = ['noise_sigma', 'm', 'n', 'entries']
    r, a = (summary[summary.method == x].set_index(keys) for x in ('RLS', 'ALS'))
    comparison = pd.DataFrame(dict(RLS_MSE=r.MSE_mean, ALS_MSE=a.MSE_mean, MSE_difference=r.MSE_mean - a.MSE_mean,
                                   RLS_time_ms=r.time_mean_ms, ALS_time_ms=a.time_mean_ms, speedup_ALS_over_RLS=a.time_mean_ms / r.time_mean_ms,
                                   ALS_sweeps=a.iterations_mean, RLS_lambda=r.lam, ALS_lambda=a.lam)).reset_index()
    comparison.to_csv(os.path.join(RESULTS_DIR, 'comparison.csv'), index=False)

    # ---- LaTeX tables ----
    def fmt(v):
        return f"{v:.4f}" if v >= 0.01 else f"{v:.1e}"
    for sigma in NOISE_LEVELS:
        lines = [r"\begin{table}[t]", r"\centering",
                 f"\\caption{{Runtime and accuracy of RLS and ALS versus matrix size on approximately low-rank data ($\\sigma={sigma:.2f}$).}}",
                 f"\\label{{tab:noisy_scalability_{tag(sigma)}}}", r"\begin{tabular}{lrrrrrrr}", r"\toprule",
                 r"Matrix size & Entries & RLS time (ms) & ALS time (ms) & Speedup & RLS MSE & ALS MSE & ALS sweeps \\", r"\midrule"]
        for c in comparison[np.isclose(comparison.noise_sigma, sigma)].itertuples():
            lines.append(f"${c.m}\\times{c.n}$ & {c.entries:,} & {c.RLS_time_ms:.0f} & {c.ALS_time_ms:.0f} & {c.speedup_ALS_over_RLS:.1f}$\\times$ "
                         f"& {fmt(c.RLS_MSE)} & {fmt(c.ALS_MSE)} & {c.ALS_sweeps:.1f} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        with open(os.path.join(RESULTS_DIR, f'table_{tag(sigma)}.tex'), 'w') as f:
            f.write('\n'.join(lines) + '\n')
    largest = comparison[comparison.entries == comparison.entries.max()]
    lines = [r"\begin{table}[t]", r"\centering",
             f"\\caption{{RLS versus ALS at the largest matrix (${largest.m.iloc[0]}\\times{largest.n.iloc[0]}$) across noise levels.}}",
             r"\label{tab:noisy_scalability_noise_summary}", r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             r"Noise $\sigma$ & RLS $\lambda$ & ALS $\lambda$ & RLS MSE & ALS MSE & RLS time (ms) & ALS time (ms) & Speedup \\", r"\midrule"]
    for c in largest.itertuples():
        lines.append(f"{c.noise_sigma:.2f} & {c.RLS_lambda:g} & {c.ALS_lambda:g} & {fmt(c.RLS_MSE)} & {fmt(c.ALS_MSE)} & {c.RLS_time_ms:.0f} "
                     f"& {c.ALS_time_ms:.0f} & {c.speedup_ALS_over_RLS:.1f}$\\times$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(RESULTS_DIR, 'table_noise_summary.tex'), 'w') as f:
        f.write('\n'.join(lines) + '\n')

    pd.set_option('display.width', 250)
    print("\n" + comparison.to_string(index=False, float_format=lambda v: f'{v:.5g}'))
    print(f"\nselected lambdas: {LAM}; results in {RESULTS_DIR}; total {(time.perf_counter() - t0) / 60:.1f} min")
