#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""picture.py

Plots for OUR estimators (Lepski / MinUpper) under the *clean* spiked sub-Gaussian
benchmark (no contamination), matching the benchmark construction.

Requested plots:
  - Fix d=600, vary n in {200,400,600,800,1000}:
      * left: CovErr vs n
      * right: Subspace vs n
  - Fix n=600, vary d in {200,400,600,800,1000}:
      * left: CovErr vs d
      * right: Subspace vs d

Style requirements:
  - Lepski and MinUpper: different colors AND different markers (e.g. red circles vs blue triangles).

Outputs:
  - picture_fixed_d600.png
  - picture_fixed_n600.png
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

try:
    from robust_covariance import robust_covariance
except Exception as e:  # pragma: no cover
    raise ImportError(
        "Cannot import robust_covariance. Please place robust_covariance.py in the same directory as picture.py."
    ) from e


def sym(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.T)


def opnorm_sym(A: np.ndarray) -> float:
    w = np.linalg.eigvalsh(sym(A))
    return float(np.max(np.abs(w)))


def projector_from_top_eigs(A: np.ndarray, r: int) -> np.ndarray:
    A = sym(A)
    _, V = np.linalg.eigh(A)
    U = V[:, -r:]
    return U @ U.T


def subspace_error(P_hat: np.ndarray, P_true: np.ndarray, r: int) -> float:
    return float(np.linalg.norm(P_hat - P_true, ord="fro") / math.sqrt(2.0 * r))


@dataclass(frozen=True)
class SpikedModel:
    Sigma: np.ndarray
    Sigma_sqrt: np.ndarray
    P_true: np.ndarray
    opnorm_Sigma: float


def make_spiked_model(d: int, r: int, theta: float, *, seed: int) -> SpikedModel:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, d))
    Q, _ = np.linalg.qr(A)
    lam = np.ones(d, dtype=float)
    lam[:r] = 1.0 + float(theta)
    Sigma = sym(Q @ np.diag(lam) @ Q.T)
    Sigma_sqrt = sym(Q @ np.diag(np.sqrt(lam)) @ Q.T)
    P_true = Q[:, :r] @ Q[:, :r].T
    opS = float(np.max(lam))
    return SpikedModel(Sigma=Sigma, Sigma_sqrt=Sigma_sqrt, P_true=P_true, opnorm_Sigma=opS)


def gen_gaussian(n: int, model: SpikedModel, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, model.Sigma.shape[0]))
    return Z @ model.Sigma_sqrt.T


def eval_one(
    X: np.ndarray,
    model: SpikedModel,
    *,
    selection: str,
    delta: float,
    K: int,
    rho: float,
    c_bias: float,
    seed: int,
) -> Tuple[float, float]:
    res = robust_covariance(
        X,
        delta=delta,
        K=K,
        rho=rho,
        selection=selection,  # "lepski" or "minupper"
        c_bias=c_bias,
        center=False,
        random_state=seed,
        norm_method="exact",
        v_opnorm_method="exact",
    )
    Sigma_hat = sym(res.Sigma_raw)
    coverr = opnorm_sym(Sigma_hat - model.Sigma) / max(1e-12, model.opnorm_Sigma)
    P_hat = projector_from_top_eigs(Sigma_hat, r=RANK_R)
    suberr = subspace_error(P_hat, model.P_true, r=RANK_R)
    return float(coverr), float(suberr)


# --------
# Settings
# --------

RANK_R = 5
THETA = 10.0

DELTA = 0.05
K_FOLDS = 5
RHO = 2.0
C_BIAS = 1.0

REPS = 3
BASE_SEED = 0


def sweep_fixed_d_vary_n(d: int, n_list: List[int]) -> Dict[str, Dict[str, List[float]]]:
    model = make_spiked_model(d=d, r=RANK_R, theta=THETA, seed=BASE_SEED + 12345 + d)
    out: Dict[str, Dict[str, List[float]]] = {
        "lepski": {"coverr": [], "subspace": []},
        "minupper": {"coverr": [], "subspace": []},
    }
    for n in n_list:
        for sel in ["lepski", "minupper"]:
            vals_cov, vals_sub = [], []
            for rep in range(REPS):
                X = gen_gaussian(n, model, seed=BASE_SEED + 1000 * rep + 17 + n)
                cov, sub = eval_one(
                    X,
                    model,
                    selection=sel,
                    delta=DELTA,
                    K=K_FOLDS,
                    rho=RHO,
                    c_bias=C_BIAS,
                    seed=BASE_SEED + 777 + rep,
                )
                vals_cov.append(cov)
                vals_sub.append(sub)
            out[sel]["coverr"].append(float(np.mean(vals_cov)))
            out[sel]["subspace"].append(float(np.mean(vals_sub)))
    return out


def sweep_fixed_n_vary_d(n: int, d_list: List[int]) -> Dict[str, Dict[str, List[float]]]:
    out: Dict[str, Dict[str, List[float]]] = {
        "lepski": {"coverr": [], "subspace": []},
        "minupper": {"coverr": [], "subspace": []},
    }
    for d in d_list:
        model = make_spiked_model(d=d, r=RANK_R, theta=THETA, seed=BASE_SEED + 12345 + d)
        for sel in ["lepski", "minupper"]:
            vals_cov, vals_sub = [], []
            for rep in range(REPS):
                X = gen_gaussian(n, model, seed=BASE_SEED + 1000 * rep + 17 + d)
                cov, sub = eval_one(
                    X,
                    model,
                    selection=sel,
                    delta=DELTA,
                    K=K_FOLDS,
                    rho=RHO,
                    c_bias=C_BIAS,
                    seed=BASE_SEED + 888 + rep,
                )
                vals_cov.append(cov)
                vals_sub.append(sub)
            out[sel]["coverr"].append(float(np.mean(vals_cov)))
            out[sel]["subspace"].append(float(np.mean(vals_sub)))
    return out


def plot_two_panel(
    *,
    x: List[int],
    y1_lepski: List[float],
    y1_minupper: List[float],
    y2_lepski: List[float],
    y2_minupper: List[float],
    xlabel: str,
    title_left: str,
    title_right: str,
    out_png: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # left: CovErr
    axes[0].plot(x, y1_lepski, marker="o", color="red", linestyle="-", label="Lepski")
    axes[0].plot(x, y1_minupper, marker="^", color="blue", linestyle="-", label="MinUpper")
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel("CovErr (relative op norm)")
    axes[0].set_title(title_left)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # right: Subspace
    axes[1].plot(x, y2_lepski, marker="o", color="red", linestyle="-", label="Lepski")
    axes[1].plot(x, y2_minupper, marker="^", color="blue", linestyle="-", label="MinUpper")
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel("Subspace error")
    axes[1].set_title(title_right)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    print(f"Saved: {out_png}")


def main() -> None:
    # ---- fixed d=600, vary n ----
    d0 = 500
    n_list = [100,200,300, 400,500, 600,700, 800,900, 1000]
    res_n = sweep_fixed_d_vary_n(d=d0, n_list=n_list)

    plot_two_panel(
        x=n_list,
        y1_lepski=res_n["lepski"]["coverr"],
        y1_minupper=res_n["minupper"]["coverr"],
        y2_lepski=res_n["lepski"]["subspace"],
        y2_minupper=res_n["minupper"]["subspace"],
        xlabel="n (samples)",
        title_left=f"CovErr vs n (d={d0}, clean Gaussian)",
        title_right=f"Subspace vs n (d={d0}, clean Gaussian)",
        out_png="picture_fixed_d600.png",
    )

    # ---- fixed n=600, vary d ----
    n0 = 500
    d_list = [100,200,300, 400,500, 600,700, 800,900, 1000]
    res_d = sweep_fixed_n_vary_d(n=n0, d_list=d_list)

    plot_two_panel(
        x=d_list,
        y1_lepski=res_d["lepski"]["coverr"],
        y1_minupper=res_d["minupper"]["coverr"],
        y2_lepski=res_d["lepski"]["subspace"],
        y2_minupper=res_d["minupper"]["subspace"],
        xlabel="d (dimension)",
        title_left=f"CovErr vs d (n={n0}, clean Gaussian)",
        title_right=f"Subspace vs d (n={n0}, clean Gaussian)",
        out_png="picture_fixed_n600.png",
    )

    plt.show()


if __name__ == "__main__":
    main()
