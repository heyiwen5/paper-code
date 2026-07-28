"""Small, self-contained comparison with the sample covariance matrix.

The observations have a nonzero unknown mean and standardized Student-t(5)
coordinates.  Student-t(5) has a finite fourth moment, but is substantially
heavier-tailed than a Gaussian distribution.
"""

from __future__ import annotations

import argparse

import numpy as np
from numpy.typing import NDArray

from adaptive_clipped_covariance import estimate_covariance

FloatArray = NDArray[np.float64]


def make_spiked_covariance(
    dimension: int,
    rank: int,
    spike: float,
    rng: np.random.Generator,
) -> tuple[FloatArray, FloatArray]:
    """Return a spiked covariance matrix and its symmetric square root."""

    basis, _ = np.linalg.qr(rng.normal(size=(dimension, rank)))
    projector = basis @ basis.T
    covariance = np.eye(dimension) + spike * projector
    square_root = (
        np.eye(dimension)
        + (np.sqrt(1.0 + spike) - 1.0) * projector
    )
    return covariance, square_root


def draw_student_t5(
    sample_size: int,
    square_root: FloatArray,
    mean: FloatArray,
    rng: np.random.Generator,
) -> FloatArray:
    """Draw standardized Student-t(5) observations with target covariance."""

    dimension = square_root.shape[0]
    latent = rng.standard_t(df=5, size=(sample_size, dimension))
    latent /= np.sqrt(5.0 / 3.0)
    return latent @ square_root.T + mean


def sample_covariance(x: FloatArray) -> FloatArray:
    """SCM with an estimated mean and normalization by n."""

    centered = x - np.mean(x, axis=0)
    return (centered.T @ centered) / x.shape[0]


def relative_spectral_error(
    estimate: FloatArray,
    target: FloatArray,
) -> float:
    """Operator-norm error divided by the target operator norm."""

    return float(
        np.linalg.norm(estimate - target, ord=2)
        / np.linalg.norm(target, ord=2)
    )


def run_experiment(
    *,
    sample_size: int,
    dimension: int,
    trials: int,
    seed: int,
) -> None:
    if sample_size < 4:
        raise ValueError("sample_size must be at least 4.")
    if dimension < 2:
        raise ValueError("dimension must be at least 2.")
    if trials < 1:
        raise ValueError("trials must be positive.")

    master_rng = np.random.default_rng(seed)
    rank = min(3, dimension)
    covariance, square_root = make_spiked_covariance(
        dimension=dimension,
        rank=rank,
        spike=8.0,
        rng=master_rng,
    )
    unknown_mean = np.linspace(-1.0, 1.0, dimension)

    adaptive_errors = []
    scm_errors = []
    selected_gammas = []

    for trial in range(trials):
        trial_seed = seed + 10_000 + trial
        trial_rng = np.random.default_rng(trial_seed)
        x = draw_student_t5(
            sample_size,
            square_root,
            unknown_mean,
            trial_rng,
        )

        adaptive = estimate_covariance(
            x,
            # Unknown mean: the default foldwise sample mean is used.
            random_state=trial_seed,
        )
        scm = sample_covariance(x)

        adaptive_errors.append(
            relative_spectral_error(adaptive.covariance, covariance)
        )
        scm_errors.append(relative_spectral_error(scm, covariance))
        selected_gammas.append(adaptive.selected_gamma)

    adaptive_errors_array = np.asarray(adaptive_errors)
    scm_errors_array = np.asarray(scm_errors)
    selected_gammas_array = np.asarray(selected_gammas)

    print("Unknown-mean Student-t(5) covariance experiment")
    print(
        f"n={sample_size}, d={dimension}, trials={trials}, "
        f"seed={seed}"
    )
    print("Metric: relative spectral error (smaller is better)")
    print(
        "Adaptive clipping: "
        f"{adaptive_errors_array.mean():.4f} "
        f"+/- {adaptive_errors_array.std(ddof=0):.4f}"
    )
    print(
        "SCM:               "
        f"{scm_errors_array.mean():.4f} "
        f"+/- {scm_errors_array.std(ddof=0):.4f}"
    )
    print(
        "Selected gamma:    "
        f"median={np.median(selected_gammas_array):.6f}, "
        f"range=[{selected_gammas_array.min():.6f}, "
        f"{selected_gammas_array.max():.6f}]"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare adaptive radial clipping with SCM."
    )
    parser.add_argument("--n", type=int, default=300, help="sample size")
    parser.add_argument("--d", type=int, default=40, help="dimension")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run_experiment(
        sample_size=arguments.n,
        dimension=arguments.d,
        trials=arguments.trials,
        seed=arguments.seed,
    )
