#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""benchmark.py

Spiked covariance / spiked PCA benchmark under heavy tails + Huber contamination.

Changes requested:
  1) Removed pyRiemann-Hub as requested.
  2) Kept all other baselines (SCM, tfHuber, MinCovDet, OGK, Wrapping, pyRiemann-Stu).
  3) Student-t distribution df=4.5.

Outputs:
  - 12 tables = 4 data types × 3 eps levels.
  - Writes:
      ./benchmark_outputs/summary_<dist>_eps<eps>.csv
      ./benchmark_outputs/benchmark_tables.xlsx

Dependencies:
  Required: numpy, pandas, scipy
  Robust baselines used here:
    - tfHuber      (pip install tfHuber; import tfHuber)
    - scikit-learn (MinCovDet)
    - robpy        (OGK, WrappingCovariance)
    - pyriemann    (covariances: "stu")
"""

from __future__ import annotations

import math
import os
import time
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd


# Our estimator (imported from ./robust_covariance.py)
try:
    from robust_covariance import robust_covariance
except Exception as e:  # pragma: no cover
    raise ImportError(
        "Cannot import robust_covariance. Please place robust_covariance.py in the same directory as this benchmark script."
    ) from e


# Optional baseline: tfHuber (pip install tfHuber; import name is `tfHuber`)
try:
    import tfHuber  # type: ignore

    _HAVE_TFHUBER = True
except Exception:  # pragma: no cover
    tfHuber = None  # type: ignore
    _HAVE_TFHUBER = False


# scikit-learn MinCovDet
try:
    from sklearn.covariance import MinCovDet

    _HAVE_SKLEARN = True
except Exception:  # pragma: no cover
    MinCovDet = None  # type: ignore
    _HAVE_SKLEARN = False


# RobPy robust covariance estimators
try:
    from robpy.covariance import OGK, WrappingCovariance

    _HAVE_ROBPY = True
except Exception:  # pragma: no cover
    OGK = None  # type: ignore
    WrappingCovariance = None  # type: ignore
    _HAVE_ROBPY = False


# pyRiemann robust covariance estimators
try:
    from pyriemann.utils.covariance import covariances as pr_covariances

    _HAVE_PYRIEMANN = True
except Exception:  # pragma: no cover
    pr_covariances = None  # type: ignore
    _HAVE_PYRIEMANN = False


# =========================
# Utilities
# =========================


def sym(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.T)


def opnorm_sym(A: np.ndarray) -> float:
    """Exact spectral norm for symmetric matrix via eigvalsh."""
    w = np.linalg.eigvalsh(sym(A))
    return float(np.max(np.abs(w)))


def corr_like(A: np.ndarray, tau0: float = 1.0 / 8.0) -> np.ndarray:
    """Paper-style stabilized correlation-like normalization."""
    A = sym(A)
    diag = np.diag(A).copy()
    diag = np.maximum(diag, tau0)
    inv_sqrt = 1.0 / np.sqrt(diag)
    R = (A * inv_sqrt[None, :]) * inv_sqrt[:, None]
    R = sym(R)
    np.fill_diagonal(R, np.minimum(1.0, np.diag(R)))
    return R


def projector_from_top_eigs(A: np.ndarray, r: int) -> np.ndarray:
    """Projector onto top-r eigenspace of symmetric A."""
    A = sym(A)
    _, V = np.linalg.eigh(A)
    U = V[:, -r:]
    return U @ U.T


def subspace_error(P_hat: np.ndarray, P_true: np.ndarray, r: int) -> float:
    """Normalized Frobenius distance between two rank-r projectors."""
    return float(np.linalg.norm(P_hat - P_true, ord="fro") / math.sqrt(2.0 * r))


def top_r_eig_rel_error(A_hat: np.ndarray, A_true: np.ndarray, r: int) -> float:
    """Mean relative error of the top-r eigenvalues."""
    w_true = np.linalg.eigvalsh(sym(A_true))
    w_hat = np.linalg.eigvalsh(sym(A_hat))
    lam_true = w_true[-r:]
    lam_hat = w_hat[-r:]
    return float(np.mean(np.abs(lam_hat - lam_true) / np.maximum(1e-12, np.abs(lam_true))))


# =========================
# Spiked covariance model
# =========================


@dataclass(frozen=True)
class SpikedModel:
    Sigma: np.ndarray
    Sigma_sqrt: np.ndarray
    P_true: np.ndarray
    R_true: np.ndarray
    opnorm_Sigma: float


def make_spiked_model(d: int, r: int, theta: float, rng: np.random.Generator, tau0: float) -> SpikedModel:
    """Σ = Q diag(λ) Q^T, with λ_1..λ_r = 1+theta, remaining = 1."""
    A = rng.standard_normal((d, d))
    Q, _ = np.linalg.qr(A)
    lam = np.ones(d, dtype=float)
    lam[:r] = 1.0 + float(theta)
    Sigma = sym(Q @ np.diag(lam) @ Q.T)
    Sigma_sqrt = sym(Q @ np.diag(np.sqrt(lam)) @ Q.T)
    P_true = Q[:, :r] @ Q[:, :r].T
    R_true = corr_like(Sigma, tau0=tau0)
    opS = float(np.max(lam))
    return SpikedModel(Sigma=Sigma, Sigma_sqrt=Sigma_sqrt, P_true=P_true, R_true=R_true, opnorm_Sigma=opS)


# =========================
# Data generators (mean = 0, Cov = Σ for clean part)
# =========================


def gen_elliptical_gaussian(n: int, d: int, Sigma_sqrt: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    Z = rng.standard_normal((n, d))
    return Z @ Sigma_sqrt.T


def gen_elliptical_student_t(
    n: int, d: int, Sigma_sqrt: np.ndarray, rng: np.random.Generator, df: float
) -> np.ndarray:
    """Multivariate t via Gaussian / chi-square scaling, then rescaled so the latent has Cov=I."""
    df = float(df)
    if df <= 4.0:
        raise ValueError("Need df>4 for finite 4th moment (this benchmark uses df=4.5).")
    G = rng.standard_normal((n, d))
    chi2 = rng.chisquare(df, size=n)
    scale = np.sqrt(chi2 / df)
    Z = (G / scale[:, None]) * math.sqrt((df - 2.0) / df)
    return Z @ Sigma_sqrt.T


def gen_nonelliptical_laplace(n: int, d: int, Sigma_sqrt: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """i.i.d. Laplace coords standardized to Var=1."""
    b = 1.0 / math.sqrt(2.0)
    Y = rng.laplace(loc=0.0, scale=b, size=(n, d))
    return Y @ Sigma_sqrt.T


def gen_nonelliptical_signed_lognormal(
    n: int, d: int, Sigma_sqrt: np.ndarray, rng: np.random.Generator, sigma: float
) -> np.ndarray:
    """Signed log-normal with exact mean 0 and Var 1 (in expectation)."""
    sigma = float(sigma)
    Z = rng.standard_normal((n, d))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n, d))
    Y = signs * np.exp(sigma * Z - sigma * sigma)
    return Y @ Sigma_sqrt.T


def contaminate_huber(X: np.ndarray, eps: float, Sigma_sqrt: np.ndarray, rng: np.random.Generator, kappa: float) -> np.ndarray:
    """Huber contamination: replace eps*n samples by N(0, kappa Σ) outliers."""
    eps = float(eps)
    if eps <= 0:
        return X
    n, d = X.shape
    m = int(round(eps * n))
    if m <= 0:
        return X
    idx = rng.choice(n, size=m, replace=False)
    out = rng.standard_normal((m, d)) @ (math.sqrt(kappa) * Sigma_sqrt).T
    X2 = X.copy()
    X2[idx, :] = out
    return X2


# =========================
# Estimators
# =========================


def est_scm(X: np.ndarray) -> np.ndarray:
    n = X.shape[0]
    return sym((X.T @ X) / float(n))


def est_tfHuber_cov(
    X: np.ndarray,
    *,
    cov_type: str = "element",
    pairwise: bool = False,
    tol: float = 1e-5,
    max_iter: int = 500,
) -> np.ndarray:
    """tfHuber tuning-free Huber covariance estimator (pip install tfHuber; import tfHuber)."""
    if not _HAVE_TFHUBER:
        raise RuntimeError(
            "tfHuber not available. Install with: pip install tfHuber (capital H). Import name is: import tfHuber."
        )
    X = np.asarray(X, dtype=float)
    Sigma = tfHuber.cov(X, type=cov_type, pairwise=pairwise, tol=tol, max_iter=max_iter)
    return sym(np.asarray(Sigma, dtype=float))


def est_sklearn_mcd(X: np.ndarray, *, support_fraction: float = 0.9, random_state: int = 0) -> np.ndarray:
    if not _HAVE_SKLEARN:
        raise RuntimeError("scikit-learn not available; needed for MinCovDet.")
    mcd = MinCovDet(support_fraction=float(support_fraction), random_state=int(random_state)).fit(X)
    return sym(np.asarray(mcd.covariance_, dtype=float))


def est_robpy_ogk(X: np.ndarray, *, reweighting: bool = True, beta: float = 0.975) -> np.ndarray:
    if not _HAVE_ROBPY:
        raise RuntimeError("robpy not available; needed for OGK.")
    est = OGK(reweighting=bool(reweighting), reweighting_beta=float(beta)).fit(X)
    return sym(np.asarray(est.covariance, dtype=float))


def est_robpy_wrapping(X: np.ndarray) -> np.ndarray:
    if not _HAVE_ROBPY:
        raise RuntimeError("robpy not available; needed for WrappingCovariance.")
    est = WrappingCovariance().fit(X)
    return sym(np.asarray(est.covariance, dtype=float))


def est_pyriemann_mest(X: np.ndarray, *, kind: str) -> np.ndarray:
    """pyRiemann M-estimator covariance. kind in {"hub","stu"}."""
    if not _HAVE_PYRIEMANN:
        raise RuntimeError("pyriemann not available; needed for pyRiemann robust estimators.")
    # X: (n, d) -> (1, d, n) as a single multi-channel time series.
    X3 = X.T[None, :, :]
    C = pr_covariances(X3, estimator=kind)[0]
    return sym(np.asarray(C, dtype=float))


# =========================
# Benchmark runner
# =========================


@dataclass(frozen=True)
class BenchConfig:
    n: int = 400
    d: int = 200
    r: int = 5
    theta: float = 10.0
    reps: int = 3
    seed: int = 0

    # our estimator settings
    delta: float = 0.05
    K: int = 5
    rho: float = 2.0
    c_bias: float = 1.0

    # stabilized correlation
    tau0: float = 1.0 / 8.0

    # distributions
    t_df: float = 4.5
    logn_sigma: float = 0.5

    # contamination
    eps_list: Tuple[float, ...] = (0.0, 0.05, 0.10)
    kappa: float = 100.0

    out_dir: str = "benchmark_outputs"


def run_one_setting(cfg: BenchConfig, model: SpikedModel, dist_name: str, eps: float) -> pd.DataFrame:
    """Run cfg.reps replications for a fixed (distribution, eps), return summary table."""

    methods: Dict[str, Callable[[np.ndarray, np.random.Generator], np.ndarray]] = {}

    def wrap_ours(selection: str):
        def _f(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
            res = robust_covariance(
                X,
                delta=cfg.delta,
                K=cfg.K,
                rho=cfg.rho,
                selection=selection,
                c_bias=cfg.c_bias,
                center=False,  # mean is 0 by construction
                random_state=int(rng.integers(0, 2**31 - 1)),
                norm_method="exact",
                v_opnorm_method="exact",
            )
            return res.Sigma_raw

        return _f

    methods["Ours-Lepski"] = wrap_ours("lepski")
    methods["Ours-MinUpper"] = wrap_ours("minupper")
    methods["SCM"] = lambda X, rng: est_scm(X)
    methods["tfHuber"] = lambda X, rng: est_tfHuber_cov(X)

    # Robust competitors (directly callable)
    methods["MinCovDet"] = lambda X, rng: est_sklearn_mcd(X, support_fraction=0.9, random_state=int(rng.integers(0, 2**31 - 1)))
    methods["OGK"] = lambda X, rng: est_robpy_ogk(X)
    methods["Wrapping"] = lambda X, rng: est_robpy_wrapping(X)
    
    # Removed pyRiemann-Hub as requested
    methods["pyRiemann-Stu"] = lambda X, rng: est_pyriemann_mest(X, kind="stu")

    # Choose generator
    def gen_clean(rng: np.random.Generator) -> np.ndarray:
        if dist_name == "ellip_gaussian":
            return gen_elliptical_gaussian(cfg.n, cfg.d, model.Sigma_sqrt, rng)
        if dist_name == "ellip_t":
            return gen_elliptical_student_t(cfg.n, cfg.d, model.Sigma_sqrt, rng, df=cfg.t_df)
        if dist_name == "nonellip_laplace":
            return gen_nonelliptical_laplace(cfg.n, cfg.d, model.Sigma_sqrt, rng)
        if dist_name == "nonellip_signed_lognormal":
            return gen_nonelliptical_signed_lognormal(cfg.n, cfg.d, model.Sigma_sqrt, rng, sigma=cfg.logn_sigma)
        raise ValueError(f"Unknown dist_name: {dist_name}")

    records: List[Dict[str, float]] = []
    for rep in range(cfg.reps):
        rng = np.random.default_rng(cfg.seed + 1000 * rep + 17)
        X = contaminate_huber(gen_clean(rng), eps, model.Sigma_sqrt, rng, kappa=cfg.kappa)

        for mname, mfun in methods.items():
            rec: Dict[str, float] = {"rep": float(rep), "method": mname}
            t0 = time.perf_counter()
            try:
                Sigma_hat = sym(np.asarray(mfun(X, rng), dtype=float))
                rec["rel_op_cov"] = opnorm_sym(Sigma_hat - model.Sigma) / max(1e-12, model.opnorm_Sigma)
                R_hat = corr_like(Sigma_hat, tau0=cfg.tau0)
                rec["rel_op_corr"] = opnorm_sym(R_hat - model.R_true) / max(1e-12, opnorm_sym(model.R_true))
                P_hat = projector_from_top_eigs(Sigma_hat, cfg.r)
                rec["subspace_err"] = subspace_error(P_hat, model.P_true, cfg.r)
                rec["top_r_eig_rel"] = top_r_eig_rel_error(Sigma_hat, model.Sigma, cfg.r)
                rec["fail"] = 0.0
            except Exception as e:
                rec["rel_op_cov"] = np.nan
                rec["rel_op_corr"] = np.nan
                rec["subspace_err"] = np.nan
                rec["top_r_eig_rel"] = np.nan
                rec["fail"] = 1.0
                rec["error_msg"] = str(e)
            rec["time_sec"] = time.perf_counter() - t0
            records.append(rec)

    df = pd.DataFrame(records)
    metrics = ["rel_op_cov", "rel_op_corr", "subspace_err", "top_r_eig_rel", "time_sec", "fail"]
    grp = df.groupby("method")[metrics]
    mean = grp.mean()
    std = grp.std(ddof=1)

    out = pd.DataFrame(index=mean.index)
    for col in ["rel_op_cov", "rel_op_corr", "subspace_err", "top_r_eig_rel", "time_sec"]:
        out[col + "_mean"] = mean[col]
        out[col + "_std"] = std[col]
    out["fail_rate"] = mean["fail"]

    order = [
        "Ours-Lepski",
        "Ours-MinUpper",
        "SCM",
        "tfHuber",
        "MinCovDet",
        "OGK",
        "Wrapping",
        "pyRiemann-Stu",
    ]
    out = out.reindex([m for m in order if m in out.index])
    return out


def main() -> None:
    cfg = BenchConfig()
    os.makedirs(cfg.out_dir, exist_ok=True)

    rng_model = np.random.default_rng(cfg.seed)
    model = make_spiked_model(cfg.d, cfg.r, cfg.theta, rng_model, tau0=cfg.tau0)

    dists = [
        "ellip_gaussian",
        "ellip_t",
        "nonellip_laplace",
        "nonellip_signed_lognormal",
    ]

    xlsx_path = os.path.join(cfg.out_dir, "benchmark_tables.xlsx")
    writer = None
    try:
        writer = pd.ExcelWriter(xlsx_path, engine="openpyxl")
        have_excel = True
    except Exception:
        have_excel = False
        warnings.warn("openpyxl not available; will write CSV/TEX only (no Excel workbook).")

    try:
        for dist in dists:
            for eps in cfg.eps_list:
                print(f"\n=== Running dist={dist}, eps={eps:.2f} ===")
                table = run_one_setting(cfg, model, dist, eps)

                eps_tag = str(eps).replace(".", "p")
                base = f"summary_{dist}_eps{eps_tag}"
                csv_path = os.path.join(cfg.out_dir, base + ".csv")
                
                table.to_csv(csv_path, float_format="%.6g")

                if have_excel and writer is not None:
                    sheet = f"{dist[:10]}_e{eps_tag}"[:31]
                    table.to_excel(writer, sheet_name=sheet)

                print(table)
    finally:
        if have_excel and writer is not None:
            writer.close()

    print("\nDone.")
    print(f"Outputs written to: {os.path.abspath(cfg.out_dir)}")
    print(f"Excel workbook: {os.path.abspath(xlsx_path)}")


if __name__ == "__main__":
    main()