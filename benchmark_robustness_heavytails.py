#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""benchmark_robustness_heavytails.py

Extra script to showcase robustness under *clean but extremely heavy-tailed* distributions.

Setting:
  - n=400, d=200, spiked covariance with rank r=5 and spike strength theta=10.
  - no contamination (eps = 0)

Distributions:
  1) Elliptical Student-t with df=3 (finite variance, infinite 4th moment)
  2) Non-elliptical signed F-distribution with df_den=6 (finite variance, infinite 4th moment)

Methods reported:
  - Ours-Lepski
  - Ours-MinUpper
  - SCM
"""

from __future__ import annotations

import math
import os
import time
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np
import pandas as pd


try:
    from robust_covariance import robust_covariance
except Exception as e:  # pragma: no cover
    raise ImportError(
        "Cannot import robust_covariance. Please place robust_covariance.py in the same directory as this script."
    ) from e


def sym(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.T)


def opnorm_sym(A: np.ndarray) -> float:
    w = np.linalg.eigvalsh(sym(A))
    return float(np.max(np.abs(w)))


def corr_like(A: np.ndarray, tau0: float = 1.0 / 8.0) -> np.ndarray:
    A = sym(A)
    diag = np.diag(A).copy()
    diag = np.maximum(diag, tau0)
    inv_sqrt = 1.0 / np.sqrt(diag)
    R = (A * inv_sqrt[None, :]) * inv_sqrt[:, None]
    R = sym(R)
    np.fill_diagonal(R, np.minimum(1.0, np.diag(R)))
    return R


def projector_from_top_eigs(A: np.ndarray, r: int) -> np.ndarray:
    A = sym(A)
    _, V = np.linalg.eigh(A)
    U = V[:, -r:]
    return U @ U.T


def subspace_error(P_hat: np.ndarray, P_true: np.ndarray, r: int) -> float:
    return float(np.linalg.norm(P_hat - P_true, ord="fro") / math.sqrt(2.0 * r))


def top_r_eig_rel_error(A_hat: np.ndarray, A_true: np.ndarray, r: int) -> float:
    w_true = np.linalg.eigvalsh(sym(A_true))
    w_hat = np.linalg.eigvalsh(sym(A_hat))
    lam_true = w_true[-r:]
    lam_hat = w_hat[-r:]
    return float(np.mean(np.abs(lam_hat - lam_true) / np.maximum(1e-12, np.abs(lam_true))))


@dataclass(frozen=True)
class SpikedModel:
    Sigma: np.ndarray
    Sigma_sqrt: np.ndarray
    P_true: np.ndarray
    R_true: np.ndarray
    opnorm_Sigma: float


def make_spiked_model(d: int, r: int, theta: float, rng: np.random.Generator, tau0: float) -> SpikedModel:
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


# -------------------------
# Heavy-tailed generators
# -------------------------


def gen_elliptical_student_t(n: int, d: int, Sigma_sqrt: np.ndarray, rng: np.random.Generator, df: float) -> np.ndarray:
    """Student-t with df>2 (finite variance). Rescale so the latent has Cov=I."""
    df = float(df)
    if df <= 2.0:
        raise ValueError("Need df>2 for finite variance.")
    G = rng.standard_normal((n, d))
    chi2 = rng.chisquare(df, size=n)
    scale = np.sqrt(chi2 / df)
    Z = (G / scale[:, None]) * math.sqrt((df - 2.0) / df)
    return Z @ Sigma_sqrt.T


def gen_signed_f(n: int, d: int, Sigma_sqrt: np.ndarray, rng: np.random.Generator, df_num: float, df_den: float) -> np.ndarray:
    """Symmetric (Signed) F-distribution.

    We sample F ~ F(d1, d2), then multiply by random sign {-1, +1}.
    This creates a non-elliptical distribution with heavy tails determined by d2.

    Moments of F(d1, d2):
      - Mean exists if d2 > 2.
      - Variance (2nd moment) exists if d2 > 4.
      - 4th moment exists if d2 > 8.

    With d2=6, we have finite variance but infinite 4th moment.
    """
    if df_den <= 4.0:
        raise ValueError("Need df_den > 4 for finite variance.")

    # Sample F distribution
    F_vals = rng.f(df_num, df_den, size=(n, d))

    # Calculate theoretical second moment E[F^2] to standardize to Var=1
    # E[F^2] = d2^2 * (d1 + 2) / (d1 * (d2 - 2) * (d2 - 4))
    second_moment = (df_den**2 * (df_num + 2)) / (df_num * (df_den - 2) * (df_den - 4))
    scale = math.sqrt(second_moment)

    # Apply random signs and scale
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n, d))
    Y = signs * (F_vals / scale)

    return Y @ Sigma_sqrt.T


# -------------------------
# Estimators
# -------------------------


def est_scm(X: np.ndarray) -> np.ndarray:
    n = X.shape[0]
    return sym((X.T @ X) / float(n))


@dataclass(frozen=True)
class Config:
    n: int = 1000
    d: int = 200
    r: int = 5
    theta: float = 10.0
    reps: int = 3
    seed: int = 0

    # our estimator settings
    delta: float = 0.05
    K: int = 5
    rho: float = 2.0
    c_bias: float = 3.0

    tau0: float = 1.0 / 8.0
    out_dir: str = "heavytail_outputs"


def run_one_setting(cfg: Config, model: SpikedModel, dist_name: str) -> pd.DataFrame:
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
                center=False,
                random_state=int(rng.integers(0, 2**31 - 1)),
                norm_method="exact",
                v_opnorm_method="exact",
            )
            return res.Sigma_raw

        return _f

    methods["Ours-Lepski"] = wrap_ours("lepski")
    methods["Ours-MinUpper"] = wrap_ours("minupper")
    methods["SCM"] = lambda X, rng: est_scm(X)

    def gen_clean(rng: np.random.Generator) -> np.ndarray:
        if dist_name == "ellip_t_df3":
            return gen_elliptical_student_t(cfg.n, cfg.d, model.Sigma_sqrt, rng, df=3.0)
        # Replaced Pareto with Signed F (d2=6)
        if dist_name == "signed_f_d6":
            # using d1=5, d2=6
            return gen_signed_f(cfg.n, cfg.d, model.Sigma_sqrt, rng, df_num=5.0, df_den=6.0)
        
        raise ValueError(f"Unknown distribution: {dist_name}")

    records: List[Dict[str, float]] = []
    for rep in range(cfg.reps):
        rng = np.random.default_rng(cfg.seed + 1000 * rep + 17)
        try:
            X = gen_clean(rng)
        except Exception as e:
            print(f"Generation failed for {dist_name} rep {rep}: {e}")
            continue

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
    if df.empty:
        return pd.DataFrame()

    metrics = ["rel_op_cov", "rel_op_corr", "subspace_err", "top_r_eig_rel", "time_sec", "fail"]
    grp = df.groupby("method")[metrics]
    mean = grp.mean()
    std = grp.std(ddof=1)

    out = pd.DataFrame(index=mean.index)
    for col in ["rel_op_cov", "rel_op_corr", "subspace_err", "top_r_eig_rel", "time_sec"]:
        out[col + "_mean"] = mean[col]
        out[col + "_std"] = std[col]
    out["fail_rate"] = mean["fail"]

    order = ["Ours-Lepski", "Ours-MinUpper", "SCM"]
    out = out.reindex([m for m in order if m in out.index])
    return out


def main() -> None:
    cfg = Config()
    os.makedirs(cfg.out_dir, exist_ok=True)

    rng_model = np.random.default_rng(cfg.seed)
    model = make_spiked_model(cfg.d, cfg.r, cfg.theta, rng_model, tau0=cfg.tau0)

    # Removed "signed_lognormal_s1p1" from this list
    dists = ["ellip_t_df3", "signed_f_d6"]
    eps_tag = "0p0"

    xlsx_path = os.path.join(cfg.out_dir, "heavytail_tables.xlsx")
    writer = None
    try:
        writer = pd.ExcelWriter(xlsx_path, engine="openpyxl")
        have_excel = True
    except Exception:
        have_excel = False
        warnings.warn("openpyxl not available; will write CSV only (no Excel workbook).")

    try:
        for dist in dists:
            print(f"\n=== Running dist={dist}, eps=0.00 ===")
            table = run_one_setting(cfg, model, dist)

            base = f"summary_{dist}_eps{eps_tag}"
            csv_path = os.path.join(cfg.out_dir, base + ".csv")
            table.to_csv(csv_path, float_format="%.6g")
            if have_excel and writer is not None:
                sheet = f"{dist[:20]}"[:31]
                table.to_excel(writer, sheet_name=sheet)

            print(table)
    finally:
        if have_excel and writer is not None:
            writer.close()

    print("\nDone.")
    print(f"Outputs written to: {os.path.abspath(cfg.out_dir)}")
    if have_excel:
        print(f"Excel workbook: {os.path.abspath(xlsx_path)}")


if __name__ == "__main__":
    main()