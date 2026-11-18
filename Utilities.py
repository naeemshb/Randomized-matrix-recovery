import time
import random
import numpy as np
from sklearn.utils.extmath import randomized_svd


# -----------------------------------------------------------------------------
# Imputation of missing entries (0 = missing)
# -----------------------------------------------------------------------------
def impute_missing_with_mean(A: np.ndarray) -> np.ndarray:
    A_copy = A.copy()
    mask = (A_copy != 0)
    A_copy[~mask] = A_copy[mask].mean()
    return A_copy


def impute_missing_with_median(A: np.ndarray) -> np.ndarray:
    A_copy = A.copy()
    mask = (A_copy != 0)
    A_copy[~mask] = np.median(A_copy[mask])
    return A_copy


def impute_missing_with_random(A: np.ndarray, random_state: int = 0) -> np.ndarray:
    A_copy = A.copy()
    mask = (A_copy != 0)
    observed = A_copy[mask]

    mean_val = observed.mean()
    std_val = observed.std()

    rng = np.random.default_rng(random_state)
    missing_count = np.sum(~mask)
    random_vals = rng.normal(mean_val, std_val, size=missing_count)

    A_copy[~mask] = random_vals

    return A_copy


# -----------------------------------------------------------------------------
# Randomized Least Squares (RLS): impute -> random sketch U = X~ W -> per-column ridge on observed entries
# -----------------------------------------------------------------------------
def RLS(M, k, lamda, seednum, impute_mode='random', round_step=0.5):
    mask = (M > 0)
    A_copy = M.copy()
    m, n = M.shape
    t_start = time.perf_counter()
    # impute missing values
    if impute_mode == 'mean':
        A_copy = impute_missing_with_mean(M)
    elif impute_mode == 'median':
        A_copy = impute_missing_with_median(M)
    else:
        A_copy = impute_missing_with_random(M, random_state=seednum)
    # generate random sketching matrix
    rng = np.random.default_rng(seednum)
    S = rng.normal(scale=1.0 / np.sqrt(k), size=(n, k))
    L = A_copy @ S
    W = np.zeros((k, n))
    for j in range(n):
        obs = mask[:, j]
        if not np.any(obs):
            continue
        L_obs = L[obs, :]
        A_obs = A_copy[obs, j]
        gram = L_obs.T @ L_obs + lamda * np.eye(k)
        rhs = L_obs.T @ A_obs
        W[:, j] = np.linalg.solve(gram, rhs)
    M_pred = L @ W
    M_pred = np.clip(M_pred, np.min(M[mask]), np.max(M[mask]))
    if round_step is not None:
        M_recover = np.round(M_pred / round_step) * round_step
    else:
        M_recover = M_pred
    total_time = (time.perf_counter() - t_start) * 1e3
    return M_recover, total_time


# -----------------------------------------------------------------------------
# Alternating Least Squares (ALS)
# -----------------------------------------------------------------------------
def ALS(M, k, lamda, max_iter, seednum, round_step=0.5, tol=1e-2, return_history=False, test_matrix=None):
    mask = (M > 0)
    n_obs = np.sum(mask)
    t_start = time.perf_counter()
    total_check_time = 0.0
    p, q = M.shape
    rng = np.random.default_rng(seednum)
    U = rng.standard_normal((p, k))
    V = rng.standard_normal((k, q))
    I_k = np.eye(k)
    prev_mse = None
    mse_history = []
    er_history = []
    for it in range(max_iter):
        # Update U
        for i in range(p):
            obs = mask[i, :]
            if not np.any(obs):
                continue
            V_obs = V[:, obs]
            M_obs = M[i, obs]
            A = V_obs @ V_obs.T + lamda * I_k
            b = V_obs @ M_obs
            U[i, :] = np.linalg.solve(A, b)
        # Update V
        for j in range(q):
            obs = mask[:, j]
            if not np.any(obs):
                continue
            U_obs = U[obs, :]
            M_obs = M[obs, j]
            A = U_obs.T @ U_obs + lamda * I_k
            b = U_obs.T @ M_obs
            V[:, j] = np.linalg.solve(A, b)
        # Convergence check
        t_check = time.perf_counter()
        M_tmp = U @ V
        curr_mse = np.sum((M_tmp[mask] - M[mask]) ** 2) / n_obs
        if return_history and test_matrix is not None:
            M_tmp_clip = np.clip(M_tmp, np.min(M[mask]), np.max(M[mask]))
            if round_step is not None:
                M_tmp_r = np.round(M_tmp_clip / round_step) * round_step
            else:
                M_tmp_r = M_tmp_clip
            test_mse, er = er_recover(M_tmp_r, test_matrix)
        total_check_time += time.perf_counter() - t_check
        if return_history and test_matrix is not None:
            mse_history.append(test_mse)
            er_history.append(er)
        if prev_mse is not None and abs(prev_mse - curr_mse) < tol:
            print(f"  [ALS] converged at iteration {it+1}/{max_iter} (ΔMSE={abs(prev_mse - curr_mse):.2e})")
            break
        prev_mse = curr_mse
    M_pred = U @ V
    M_pred = np.clip(M_pred, np.min(M[mask]), np.max(M[mask]))
    if round_step is not None:
        M_recover = np.round(M_pred / round_step) * round_step
    else:
        M_recover = M_pred
    total_time = (time.perf_counter() - t_start) * 1e3 - total_check_time * 1e3
    if return_history:
        return M_recover, total_time, mse_history, er_history
    return M_recover, total_time


# -----------------------------------------------------------------------------
# Nuclear-norm minimisation by iterative singular-value thresholding (NuNo)
# -----------------------------------------------------------------------------
def nuclear_norm(M, epsilon, max_iter, seednum=0, round_step=0.5, tol=1e-5, return_history=False, test_matrix=None):
    m, n = M.shape
    mask = (M > 0)
    n_obs = np.sum(mask)
    t_start = time.perf_counter()
    total_check_time = 0.0
    X = impute_missing_with_random(M, random_state=seednum)
    observed_mean = M[mask].mean() if np.sum(mask) > 0 else 0
    X_prev = None
    mse_history = []
    er_history = []
    for iteration in range(max_iter):
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        s_thresh = np.maximum(s - epsilon, 0)
        if np.sum(s_thresh > 1e-12) > 0:
            keep_idx = s_thresh > 1e-12
            U_keep = U[:, keep_idx]
            s_keep = s_thresh[keep_idx]
            Vt_keep = Vt[keep_idx, :]
            X = U_keep @ np.diag(s_keep) @ Vt_keep
        else:
            X = np.full_like(X, observed_mean)
        X[mask] = M[mask]
        # Convergence check: use change in full matrix (observed MSE is always 0)
        t_check = time.perf_counter()
        if return_history and test_matrix is not None:
            M_tmp_clip = np.clip(X, np.min(M[mask]), np.max(M[mask]))
            if round_step is not None:
                M_tmp_r = np.round(M_tmp_clip / round_step) * round_step
            else:
                M_tmp_r = M_tmp_clip
            test_mse, er = er_recover(M_tmp_r, test_matrix)
        delta_x = np.mean((X - X_prev) ** 2) if X_prev is not None else None
        total_check_time += time.perf_counter() - t_check
        if return_history and test_matrix is not None:
            mse_history.append(test_mse)
            er_history.append(er)
        if delta_x is not None and delta_x < tol:
            print(f"  [NuNo] converged at iteration {iteration+1}/{max_iter} (ΔX={delta_x:.2e})")
            break
        X_prev = X.copy()
    M_pred = np.clip(X, np.min(M[mask]), np.max(M[mask]))
    if round_step is not None:
        M_recover = np.round(M_pred / round_step) * round_step
    else:
        M_recover = M_pred
    total_time = (time.perf_counter() - t_start) * 1e3 - total_check_time * 1e3
    if return_history:
        return M_recover, total_time, mse_history, er_history
    return M_recover, total_time


# -----------------------------------------------------------------------------
# NuNo with randomized SVD (NuNoR)
# -----------------------------------------------------------------------------
def nuclear_norm_random(M, k, epsilon, max_iter, seednum, round_step=0.5, tol=1e-5, return_history=False, test_matrix=None):
    m, n = M.shape
    mask = (M > 0)
    n_obs = np.sum(mask)
    t_start = time.perf_counter()
    total_check_time = 0.0
    X = impute_missing_with_random(M, random_state=seednum)
    observed_mean = M[mask].mean() if np.sum(mask) > 0 else 0
    X_prev = None
    mse_history = []
    er_history = []
    for iteration in range(max_iter):
        U, s, Vt = randomized_svd(X, n_components=k, random_state=seednum + iteration)
        s_thresh = np.maximum(s - epsilon, 0)
        if np.sum(s_thresh > 1e-12) > 0:
            keep_idx = s_thresh > 1e-12
            U_keep = U[:, keep_idx]
            s_keep = s_thresh[keep_idx]
            Vt_keep = Vt[keep_idx, :]
            X = U_keep @ np.diag(s_keep) @ Vt_keep
        else:
            X = np.full_like(X, observed_mean)
        X[mask] = M[mask]
        # Convergence check: use change in full matrix (observed MSE is always 0)
        t_check = time.perf_counter()
        if return_history and test_matrix is not None:
            M_tmp_clip = np.clip(X, np.min(M[mask]), np.max(M[mask]))
            if round_step is not None:
                M_tmp_r = np.round(M_tmp_clip / round_step) * round_step
            else:
                M_tmp_r = M_tmp_clip
            test_mse, er = er_recover(M_tmp_r, test_matrix)
        delta_x = np.mean((X - X_prev) ** 2) if X_prev is not None else None
        total_check_time += time.perf_counter() - t_check
        if return_history and test_matrix is not None:
            mse_history.append(test_mse)
            er_history.append(er)
        if delta_x is not None and delta_x < tol:
            print(f"  [NuNoR] converged at iteration {iteration+1}/{max_iter} (ΔX={delta_x:.2e})")
            break
        X_prev = X.copy()
    M_pred = np.clip(X, np.min(M[mask]), np.max(M[mask]))
    if round_step is not None:
        M_recover = np.round(M_pred / round_step) * round_step
    else:
        M_recover = M_pred
    total_time = (time.perf_counter() - t_start) * 1e3 - total_check_time * 1e3
    if return_history:
        return M_recover, total_time, mse_history, er_history
    return M_recover, total_time


# -----------------------------------------------------------------------------
# Columns Selected Matrix Completion (CSMC)
# -----------------------------------------------------------------------------
def csmc(M, col_percent, k, epsilon, max_iter, lamda, seednum, round_step=0.5, tol=1e-4, return_history=False, test_matrix=None):
    m, n = M.shape
    mask = (M > 0)
    t_start = time.perf_counter()
    idx_sel = random.sample(range(1, n), int(n * col_percent))
    M_sub = M[:, idx_sel]

    test_sub = test_matrix[:, idx_sel] if test_matrix is not None else None
    result = nuclear_norm_random(M_sub, k, epsilon, max_iter, seednum,
                                 round_step=round_step, tol=tol, return_history=return_history,
                                 test_matrix=test_sub)
    if return_history:
        L, _, mse_history, er_history = result
    else:
        L = result[0]

    k = L.shape[1]
    W = np.zeros((k, n))
    for j in range(n):
        obs = mask[:, j]
        if not np.any(obs):
            continue
        L_obs = L[obs, :]
        A_obs = M[obs, j]
        gram = L_obs.T @ L_obs + lamda * np.eye(k)
        rhs = L_obs.T @ A_obs
        W[:, j] = np.linalg.solve(gram, rhs)
    M_pred = L @ W
    M_pred = np.clip(M_pred, np.min(M[mask]), np.max(M[mask]))
    if round_step is not None:
        M_recover = np.round(M_pred / round_step) * round_step
    else:
        M_recover = M_pred
    total_time = (time.perf_counter() - t_start) * 1e3
    if return_history:
        return M_recover, total_time, mse_history, er_history
    return M_recover, total_time


# -----------------------------------------------------------------------------
# Evaluation: MSE and error rate on the entries of M_true that are observed (> 0)
# -----------------------------------------------------------------------------
def er_recover(M_pred, M_true, round_step=None):
    mask = (M_true > 0)
    n_missing = np.sum(mask)
    if n_missing == 0:
        raise ValueError("No missing positions to evaluate")
    mse = (M_true - M_pred)[mask] ** 2
    mse = np.sum(mse) / n_missing
    if round_step is not None:
        pred_r = np.round(M_pred / round_step) * round_step
        true_r = np.round(M_true / round_step) * round_step
        er = (true_r != pred_r)[mask]
    else:
        er = (M_true != M_pred)[mask]
    er = np.sum(er) / n_missing
    return mse, er
