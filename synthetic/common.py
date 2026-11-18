import os
import io
import re
import sys
import json
import time
import random
import platform
import contextlib
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, '..'))                 # Utilities.py is in the repository root
from Utilities import RLS, ALS, csmc, er_recover             # noqa: E402

VAL_RATIO, VAL_SEED = 0.10, 12345                            # validation split used for hyper-parameter selection
TIE_TOL = 1e-12


def output_dirs(name, quick=False):
    """results/<name>/ and figures/<name>/ next to this file ('_quick' is appended for smoke tests)."""
    name += '_quick' if quick else ''
    return os.path.join(ROOT, 'results', name), os.path.join(ROOT, 'figures', name)


# ---------------------------------------------------------------------------------------------------------------
# Timed calls. The reported runtime is the wall-clock time around the bare recovery call ("external" time);
# the time each method measures internally is returned as well. Predictions are continuous (no rounding).
# ---------------------------------------------------------------------------------------------------------------
def timed_rls(train, k, lam, seed):
    """Returns (recovery, external_ms, internal_ms)."""
    t = time.perf_counter()
    recovery, internal = RLS(train, k, lam, seed, impute_mode='random', round_step=None)
    return recovery, (time.perf_counter() - t) * 1e3, internal


def timed_als(train, k, lam, max_iter, tol, seed):
    """Returns (recovery, external_ms, internal_ms, sweeps). The number of sweeps is read from the
    message ALS prints when it stops early; without that message all max_iter sweeps were run."""
    out = io.StringIO()
    t = time.perf_counter()
    with contextlib.redirect_stdout(out):
        recovery, internal = ALS(train, k, lam, max_iter, seed, round_step=None, tol=tol)
    external = (time.perf_counter() - t) * 1e3
    stopped = re.search(r'converged at iteration (\d+)/', out.getvalue())
    return recovery, external, internal, int(stopped.group(1)) if stopped else max_iter


def timed_csmc(train, col_percent, k, epsilon, max_iter, lam, tol, seed):
    """Returns (recovery, external_ms, internal_ms, iterations of the internal nuclear-norm stage).
    csmc() samples its columns with Python's `random` module, which is seeded here, outside the timed region."""
    random.seed(seed)
    out = io.StringIO()
    t = time.perf_counter()
    with contextlib.redirect_stdout(out):
        recovery, internal = csmc(train, col_percent, k, epsilon, max_iter, lam, seed, round_step=None, tol=tol)
    external = (time.perf_counter() - t) * 1e3
    stopped = re.search(r'\[NuNoR\] converged at iteration (\d+)/', out.getvalue())
    return recovery, external, internal, int(stopped.group(1)) if stopped else max_iter


def mse(recovery, held_out):
    return er_recover(recovery, held_out, round_step=None)[0]


# ---------------------------------------------------------------------------------------------------------------
# Hyper-parameter selection on a validation split of the training observations
# ---------------------------------------------------------------------------------------------------------------
def validation_split(train):
    """Move 10 % of the observed training entries into a validation matrix. Returns (fit, val)."""
    rng = np.random.default_rng(VAL_SEED)
    i, j = np.where(train > 0)
    idx = rng.choice(len(i), size=int(VAL_RATIO * len(i)), replace=False)
    val = np.zeros_like(train)
    val[i[idx], j[idx]] = train[i[idx], j[idx]]
    fit = train.copy()
    fit[i[idx], j[idx]] = 0.0
    return fit, val


def select_lambda(splits, recover, grid, expand_up=(), expand_down=(), label=''):
    """Choose the lambda with the lowest validation MSE, averaged over the tuning seeds.

    splits  : {seed: (fit, val)}
    recover : function (fit, lam, seed) -> recovery
    If the best value lies on an edge of the grid, the grid is extended once in that direction.
    Ties go to the smaller lambda. Returns (lambda, validation MSE, 'no' | 'up' | 'down', all candidate rows).
    """
    rows = []

    def evaluate(candidates):
        for lam in candidates:
            v = [mse(recover(fit, lam, seed), val) for seed, (fit, val) in splits.items()]
            rows.append(dict(lam=float(lam), val_MSE=float(np.mean(v)), n_seeds=len(v)))
            print(f"    {label} lambda={lam:<9g} validation MSE {np.mean(v):.6f}", flush=True)

    def best():
        low = min(r['val_MSE'] for r in rows)
        return min(r['lam'] for r in rows if r['val_MSE'] <= low + TIE_TOL), low

    evaluate(grid)
    lam, expanded = best()[0], 'no'
    if lam == max(r['lam'] for r in rows) and len(expand_up):
        expanded = 'up'
        evaluate(expand_up)
    elif lam == min(r['lam'] for r in rows) and len(expand_down):
        expanded = 'down'
        evaluate(expand_down)
    lam, low = best()
    print(f"  {label} selected lambda = {lam:g} (validation MSE {low:.6f}, grid expanded: {expanded})")
    return lam, low, expanded, rows


# ---------------------------------------------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------------------------------------------
def ci95_halfwidth(x):
    """Half-width of the 95 % t confidence interval of the mean."""
    x = np.asarray(x, float)
    return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan


def loglog_fit(entries, time_ms):
    """Empirical slope of log(runtime) against log(number of entries). Returns (slope, intercept, R^2)."""
    x, y = np.log(np.asarray(entries, float)), np.log(np.asarray(time_ms, float))
    if len(x) < 2:
        return np.nan, np.nan, np.nan
    slope, intercept = np.polyfit(x, y, 1)
    r2 = 1 - np.sum((y - (intercept + slope * x)) ** 2) / np.sum((y - y.mean()) ** 2) if len(x) > 2 else np.nan
    return slope, intercept, r2


def save_environment(path):
    """Runtimes depend on the machine: record it next to the results."""
    env = dict(python=platform.python_version(), numpy=np.__version__, platform=platform.platform(),
               processor=platform.processor(), logical_cpus=os.cpu_count())
    with open(path, 'w') as f:
        json.dump(env, f, indent=2)
    print("environment:", env)
