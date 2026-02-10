# robust_covariance.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple, Union
import numpy as np

SelectionRule = Literal["lepski", "minupper"]
NormMethod = Literal["power", "exact"]
VOpnormMethod = Literal["exact", "power", "bound"]


@dataclass
class RobustCovarianceResult:
    Sigma_raw: np.ndarray
    grid: np.ndarray
    Psi_raw: np.ndarray
    Psi_raw_monotone: np.ndarray
    bias_proxy: Optional[np.ndarray]
    Sigma_raw_path: Dict[float, np.ndarray]
    dropped_indices: np.ndarray
    folds: List[np.ndarray]


def _sym(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.T)


def _power_iteration_opnorm(matvec, d: int, *, n_iter: int = 25, n_restarts: int = 2,
                            tol: float = 1e-7, rng: Optional[np.random.Generator] = None) -> float:
    if rng is None:
        rng = np.random.default_rng(0)
    best = 0.0
    for _ in range(max(1, n_restarts)):
        v = rng.standard_normal(d)
        nv = float(np.linalg.norm(v))
        if nv == 0:
            v[0] = 1.0
            nv = 1.0
        v /= nv
        last = 0.0
        for _it in range(max(1, n_iter)):
            w = matvec(v)
            nw = float(np.linalg.norm(w))
            if nw == 0.0:
                break
            v = w / nw
            Av = matvec(v)
            val = float(np.dot(v, Av))
            if abs(val - last) <= tol * max(1.0, abs(last)):
                last = val
                break
            last = val
        best = max(best, abs(last))
    return float(best)


def _spectral_norm_symmetric(A: np.ndarray, *, method: NormMethod = "power",
                            rng: Optional[np.random.Generator] = None) -> float:
    A = _sym(A)
    if method == "exact":
        return float(np.max(np.abs(np.linalg.eigvalsh(A))))
    if method == "power":
        d = A.shape[0]
        return _power_iteration_opnorm(lambda v: A @ v, d, rng=rng)
    raise ValueError(method)


def _d_meb(n: int, d: int, alpha: float, opnorm_V: float) -> float:
    log1 = np.log((n * d) / ((n - 1) * alpha))
    log2 = np.log((2.0 * n * d) / alpha)
    term1 = log1 / (3.0 * n)
    term2 = np.sqrt((2.0 * opnorm_V * log1) / n)
    term3 = (np.sqrt(5.0 / 3.0) + 1.0) * np.sqrt(log1 * log2) / n
    return float(term1 + term2 + term3)


def _mom_1d(values: np.ndarray, n_blocks: int, rng: np.random.Generator) -> float:
    vals = np.asarray(values, dtype=float)
    m = int(vals.shape[0])
    if m == 0:
        return 0.0
    n_blocks = int(max(1, min(n_blocks, m)))
    block_size = m // n_blocks
    if block_size == 0:
        n_blocks = m
        block_size = 1
    perm = rng.permutation(m)
    vals = vals[perm]
    use = n_blocks * block_size
    vals = vals[:use].reshape(n_blocks, block_size)
    block_means = np.mean(vals, axis=1)
    return float(np.median(block_means))


def _make_folds(n: int, K: int, rng: np.random.Generator) -> Tuple[List[np.ndarray], np.ndarray]:
    perm = rng.permutation(n)
    raw = np.array_split(perm, K)
    folds, dropped = [], []
    for f in raw:
        f = np.asarray(f, dtype=int)
        if f.size % 2 == 1:
            dropped.append(int(f[-1]))
            f = f[:-1]
        if f.size > 0:
            folds.append(f)
    return folds, np.array(dropped, dtype=int)


def _grid_tail_probs(mink_J: int, rho: float) -> np.ndarray:
    gamma_max = 0.5
    gamma_min = min(0.25, 1.0 / float(mink_J))
    l_max = int(np.floor(np.log(gamma_max / gamma_min) / np.log(rho)))
    gammas = [gamma_max * (rho ** (-l)) for l in range(l_max + 1)]
    gammas.append(gamma_min)
    return np.array(sorted(set(gammas)), dtype=float)


def _opnorm_Vstar_from_U(
    U: np.ndarray,
    *,
    method: VOpnormMethod,
    rng: np.random.Generator,
    n_iter: int = 15,
    n_restarts: int = 2,
) -> float:
    """
    Operator norm of the paired-variance matrix V^* used by the closed-form MEB1 envelope.

    For a fold with normalized/clipped vectors u_i (rows of U), define A_i = u_i u_i^T and
        V^* := (1/n_k) * sum_{j=1}^{n_k/2} (A_{2j-1} - A_{2j})^2 .

    Parameters
    ----------
    method :
        - "exact": form V^* explicitly and compute its top eigenvalue via eigvalsh.
        - "power": estimate ||V^*||_op via a matvec-based power iteration.
        - "bound": return the universal bound 1/2 (rigorous but conservative) when ||u_i||_2<=1.


    """
    n_k, d = U.shape

    if method == "bound":
        return 0.5

    if n_k % 2 != 0:
        raise ValueError("Fold size must be even.")

    U0 = U[0::2, :]
    U1 = U[1::2, :]
    uu = np.sum(U0 * U0, axis=1)
    vv = np.sum(U1 * U1, axis=1)
    uv = np.sum(U0 * U1, axis=1)

    if method == "exact":
        # (A-B)^2 = ||u||^2 (u u^T) + ||v||^2 (v v^T) - (u^T v)(u v^T + v u^T)
        term0 = U0.T @ (uu[:, None] * U0)
        term1 = U1.T @ (vv[:, None] * U1)
        cross = U0.T @ (uv[:, None] * U1)
        V = (term0 + term1 - cross - cross.T) / float(n_k)
        V = _sym(V)
        w = np.linalg.eigvalsh(V)
        return float(np.max(np.abs(w)))

    if method == "power":

        def matvec(x: np.ndarray) -> np.ndarray:
            ux = U0 @ x
            vx = U1 @ x
            a = uu * ux - uv * vx
            b = vv * vx - uv * ux
            return (U0.T @ a + U1.T @ b) / float(n_k)

        return _power_iteration_opnorm(matvec, d, n_iter=n_iter, n_restarts=n_restarts, rng=rng)

    raise ValueError(method)



def robust_covariance(
    X: np.ndarray,
    *,
    delta: float = 0.05,
    K: int = 5,
    rho: float = 2.0,
    selection: SelectionRule = "lepski",
    c_bias: float = 3.0,
    center: bool = False,  
    random_state: Optional[int] = 0,
    norm_method: NormMethod = "power",
    v_opnorm_method: VOpnormMethod = "power",
) -> RobustCovarianceResult:
    """
    Cross-fitted Euclidean-norm clipping covariance estimator with automatic gamma selection.

    This is the core routine used in the paper and in the benchmark script.

    Parameters
    ----------
    norm_method : {"power","exact"}
        How to compute operator norms inside the Lepski comparisons.
        Use "exact" for reproducible, non-approximate behaviour (via eigvalsh).
    v_opnorm_method : {"power","exact","bound"}
        How to compute ||V^*||_op in the empirical-Bernstein envelope.
        - "exact": form V^* explicitly and take eigvalsh (no approximation).
        - "power": matvec-based power iteration (fast, but approximate).
        - "bound": return 1/2 (rigorous but conservative).
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    rng = np.random.default_rng(random_state)
    Z = X.copy()
    if center:
        Z -= np.mean(Z, axis=0, keepdims=True)

    folds, dropped = _make_folds(n, K, rng)
    if len(folds) < 2:
        raise ValueError("Too few folds after dropping odds.")

    retained = np.concatenate(folds)
    retained_mask = np.zeros(n, dtype=bool)
    retained_mask[retained] = True

    folds_list, J_list, nks = [], [], []
    for Ik in folds:
        folds_list.append(Ik)
        nks.append(Ik.size)
    n_ret = int(sum(nks))
    for Ik in folds_list:
        mask = retained_mask.copy()
        mask[Ik] = False
        J_list.append(np.where(mask)[0])

    mink_J = min(J.size for J in J_list)
    if mink_J < 4:
        raise ValueError("Need min_k |J_k| >= 4.")
    G = _grid_tail_probs(mink_J, rho)
    G_size = int(G.size)

    delta_var = delta / 2.0
    delta_bias = delta / 2.0
    alpha = delta_var / (2.0 * len(folds_list) * G_size)
    B_mom = int(np.ceil(8.0 * np.log((2.0 * len(folds_list) * G_size) / delta_bias)))

    fold_train_sorted_norms, fold_rmin, fold_test_Z, fold_test_norms = [], [], [], []
    for Ik, Jk in zip(folds_list, J_list):
        Ztr = Z[Jk, :]
        t = np.linalg.norm(Ztr, axis=1)
        fold_train_sorted_norms.append(np.sort(t))
        pos = t[t > 0]
        fold_rmin.append(float(np.min(pos)) if pos.size > 0 else 1.0)

        Zte = Z[Ik, :]
        fold_test_Z.append(Zte)
        fold_test_norms.append(np.linalg.norm(Zte, axis=1))

    Sigma_path: Dict[float, np.ndarray] = {}
    Psi_raw = np.zeros(G_size, dtype=float)
    bias_proxy = np.zeros(G_size, dtype=float) if selection == "minupper" else None

    for j, gamma in enumerate(G):
        Sigma_agg = np.zeros((d, d), dtype=float)
        Psi_agg = 0.0
        bias_agg = 0.0

        for k, Ik in enumerate(folds_list):
            n_k = int(Ik.size)
            t_sorted = fold_train_sorted_norms[k]
            m = int(t_sorted.size)
            p = int(np.floor(float(gamma) * m))
            p = max(1, min(p, m - 1))
            rhat = float(t_sorted[m - p - 1])
            r_k = max(rhat, fold_rmin[k])

            Zte = fold_test_Z[k]
            nte = fold_test_norms[k]
            denom = np.maximum(r_k, nte)
            U = Zte / denom[:, None]

            Sigma_k = (r_k ** 2) * (U.T @ U) / float(n_k)
            Sigma_agg += (n_k / n_ret) * Sigma_k

            opV = _opnorm_Vstar_from_U(U, method=v_opnorm_method, rng=rng)
            D = _d_meb(n_k, d, alpha, opV)
            Psi_k = (r_k ** 2) * D
            Psi_agg += (n_k / n_ret) * Psi_k

            if selection == "minupper":
                Y = (nte ** 2) * (nte > r_k)
                bias_k = _mom_1d(Y, n_blocks=B_mom, rng=rng)
                bias_agg += (n_k / n_ret) * bias_k

        Sigma_path[float(gamma)] = _sym(Sigma_agg)
        Psi_raw[j] = float(Psi_agg)
        if selection == "minupper":
            bias_proxy[j] = float(bias_agg)

    Psi_bar = np.maximum.accumulate(Psi_raw[::-1])[::-1]

    if selection == "lepski":
        admissible = []
        for j in range(G_size):
            ok = True
            Sj = Sigma_path[float(G[j])]
            for s in range(j + 1):
                Ss = Sigma_path[float(G[s])]
                if _spectral_norm_symmetric(Sj - Ss, method=norm_method, rng=rng) > 3.0 * Psi_bar[s]:
                    ok = False
                    break
            if ok:
                admissible.append(j)
        j_hat = int(max(admissible))
        gamma_hat = float(G[j_hat])
    else:
        obj = Psi_bar + c_bias * bias_proxy
        j_hat = int(np.argmin(obj))
        gamma_hat = float(G[j_hat])

    return RobustCovarianceResult(
        Sigma_raw=Sigma_path[gamma_hat],
        grid=G,
        Psi_raw=Psi_raw,
        Psi_raw_monotone=Psi_bar,
        bias_proxy=bias_proxy,
        Sigma_raw_path=Sigma_path,
        dropped_indices=dropped,
        folds=folds_list,
    )