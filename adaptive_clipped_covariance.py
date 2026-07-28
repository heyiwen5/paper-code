"""Fully data-driven covariance estimation by adaptive radial clipping.

The implementation uses two-fold cross-fitting.  Each fold uses the other
fold to estimate the mean (when it is unknown) and to choose candidate
clipping radii.  A data-computable variance term and a block-quantile
tail-energy term select the clipping fraction.

Only NumPy is required.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
IndexArray = NDArray[np.int64]
MeanEstimator = Callable[[FloatArray], ArrayLike]

DEFAULT_GAMMA_MAX = 0.3


@dataclass
class ClippedCovarianceFamily:
    """The complete cross-fitted family used by the selector.

    Attributes
    ----------
    gamma_grid:
        Candidate clipping fractions, from largest to smallest.
    covariance_by_gamma:
        Cross-fitted covariance estimate for every candidate.
    variance_radius_raw:
        Unadjusted empirical-Bernstein term for every candidate.
    variance_radius:
        Monotone variance term used in the final score.
    clipping_radii:
        Pair of fold-specific radii for every candidate.
    tail_excess:
        Held-out squared-norm excesses used by the tail-energy score.
    test_indices, training_indices:
        The two cross-fitting splits.
    mean_estimates:
        Fold-specific centers.  The center for a test fold is fitted only on
        its corresponding training fold.
    """

    gamma_grid: Tuple[float, ...]
    covariance_by_gamma: Dict[float, FloatArray]
    variance_radius_raw: Dict[float, float]
    variance_radius: Dict[float, float]
    clipping_radii: Dict[float, Tuple[float, float]]
    tail_excess: Dict[float, Tuple[FloatArray, FloatArray]]
    test_indices: Dict[int, IndexArray]
    training_indices: Dict[int, IndexArray]
    mean_estimates: Dict[int, FloatArray]
    mean_method: str
    n_input: int
    n_used: int
    delta: float
    rho: float
    gamma_max: float
    random_state: int


@dataclass
class CovarianceEstimate:
    """Output returned by :func:`estimate_covariance`."""

    covariance: FloatArray
    selected_gamma: float
    clipping_radii: Tuple[float, float]
    score_by_gamma: Dict[float, float]
    tail_score_by_gamma: Dict[float, float]
    variance_score_by_gamma: Dict[float, float]
    mean_estimates: Tuple[FloatArray, FloatArray]
    family: ClippedCovarianceFamily


def sample_mean(x: FloatArray) -> FloatArray:
    """Return the coordinatewise sample mean.

    This is the default estimator when the population mean is unknown.
    """

    return np.mean(np.asarray(x, dtype=float), axis=0)


def split_two_even_folds(
    n: int,
    rng: np.random.Generator,
) -> Tuple[IndexArray, IndexArray]:
    """Randomly retain ``4 * floor(n / 4)`` rows and split them evenly.

    Both folds have even size, as required by the paired variance proxy.
    At most three observations are discarded.
    """

    n_used = (n // 4) * 4
    if n_used < 4:
        raise ValueError("At least four observations are required.")

    permutation = rng.permutation(n)[:n_used]
    half = n_used // 2
    return np.sort(permutation[:half]), np.sort(permutation[half:])


def make_clipping_grid(
    min_training_size: int,
    *,
    rho: float = 2.0,
    gamma_max: float = DEFAULT_GAMMA_MAX,
) -> Tuple[float, ...]:
    """Construct the positive geometric grid of clipping fractions."""

    if min_training_size <= 0:
        raise ValueError("min_training_size must be positive.")
    if rho <= 1.0:
        raise ValueError("rho must be larger than 1.")
    if not 0.0 < gamma_max <= 0.5:
        raise ValueError("gamma_max must lie in (0, 0.5].")

    gamma_min = min(0.25, 1.0 / float(min_training_size))
    gamma_min = min(gamma_min, gamma_max)

    grid: List[float] = []
    gamma = float(gamma_max)
    while gamma >= gamma_min * (1.0 - 1e-12):
        grid.append(gamma)
        gamma /= rho
    return tuple(grid)


def clip_rows(x: FloatArray, radius: float) -> FloatArray:
    """Apply Euclidean radial clipping row by row."""

    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("x must be a two-dimensional array.")
    if radius < 0.0:
        raise ValueError("radius must be nonnegative.")
    if radius == 0.0:
        return np.zeros_like(x)

    norms = np.linalg.norm(x, axis=1)
    factors = np.ones_like(norms)
    clipped = norms > radius
    factors[clipped] = radius / norms[clipped]
    return x * factors[:, None]


def paired_variance_proxy_operator_norm(
    clipped_x: FloatArray,
    radius: float,
) -> float:
    """Compute the operator norm of the paired matrix-variance proxy."""

    clipped_x = np.asarray(clipped_x, dtype=float)
    if clipped_x.ndim != 2:
        raise ValueError("clipped_x must be a two-dimensional array.")
    if radius < 0.0:
        raise ValueError("radius must be nonnegative.")
    if radius == 0.0:
        return 0.0

    n = clipped_x.shape[0]
    if n < 2 or n % 2 != 0:
        raise ValueError("The paired variance proxy needs a positive even row count.")

    first = clipped_x[0::2]
    second = clipped_x[1::2]
    norm_first_sq = np.sum(first * first, axis=1)
    norm_second_sq = np.sum(second * second, axis=1)
    inner_products = np.sum(first * second, axis=1)

    accumulator = np.einsum(
        "i,ij,ik->jk",
        norm_first_sq,
        first,
        first,
        optimize=True,
    )
    accumulator += np.einsum(
        "i,ij,ik->jk",
        norm_second_sq,
        second,
        second,
        optimize=True,
    )
    accumulator -= np.einsum(
        "i,ij,ik->jk",
        inner_products,
        first,
        second,
        optimize=True,
    )
    accumulator -= np.einsum(
        "i,ij,ik->jk",
        inner_products,
        second,
        first,
        optimize=True,
    )
    accumulator /= n * radius**4

    largest_eigenvalue = np.linalg.eigvalsh(accumulator)[-1]
    return float(max(largest_eigenvalue, 0.0))


def matrix_empirical_bernstein_radius(
    sample_size: int,
    dimension: int,
    alpha: float,
    variance_proxy_operator_norm: float,
) -> float:
    """Compute the matrix empirical-Bernstein radius used by the selector."""

    if sample_size < 2:
        raise ValueError("sample_size must be at least 2.")
    if dimension < 1:
        raise ValueError("dimension must be positive.")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1).")

    log_first = math.log(
        (sample_size * dimension) / ((sample_size - 1) * alpha)
    )
    log_second = math.log((2 * sample_size * dimension) / alpha)
    first = log_first / (3.0 * sample_size)
    second = math.sqrt(
        2.0
        * max(variance_proxy_operator_norm, 0.0)
        * log_first
        / sample_size
    )
    third = (
        (math.sqrt(5.0 / 3.0) + 1.0)
        * math.sqrt(log_first * log_second)
        / sample_size
    )
    return float(first + second + third)


def _validate_data(x: ArrayLike) -> FloatArray:
    data = np.asarray(x, dtype=float)
    if data.ndim != 2:
        raise ValueError("x must have shape (n_samples, n_features).")
    if data.shape[0] < 4:
        raise ValueError("At least four observations are required.")
    if data.shape[1] < 1:
        raise ValueError("At least one feature is required.")
    if not np.all(np.isfinite(data)):
        raise ValueError("x must contain only finite values.")
    return data


def _validate_mean_vector(
    value: ArrayLike,
    dimension: int,
    *,
    source: str,
) -> FloatArray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (dimension,):
        raise ValueError(
            f"{source} must return shape ({dimension},), got {vector.shape}."
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{source} returned a non-finite value.")
    return vector.copy()


def _fold_mean_estimates(
    x: FloatArray,
    training_indices: Dict[int, IndexArray],
    *,
    assume_centered: bool,
    known_mean: ArrayLike | None,
    mean_estimator: MeanEstimator | None,
) -> Tuple[Dict[int, FloatArray], str]:
    dimension = x.shape[1]

    if assume_centered and known_mean is not None:
        raise ValueError("Use either assume_centered=True or known_mean, not both.")
    if assume_centered and mean_estimator is not None:
        raise ValueError(
            "mean_estimator is incompatible with assume_centered=True."
        )
    if known_mean is not None and mean_estimator is not None:
        raise ValueError("Use either known_mean or mean_estimator, not both.")

    if assume_centered:
        zero = np.zeros(dimension, dtype=float)
        return {1: zero.copy(), 2: zero.copy()}, "assumed centered"

    if known_mean is not None:
        center = _validate_mean_vector(
            known_mean,
            dimension,
            source="known_mean",
        )
        return {1: center.copy(), 2: center.copy()}, "known mean"

    estimator = sample_mean if mean_estimator is None else mean_estimator
    if not callable(estimator):
        raise TypeError("mean_estimator must be callable.")

    estimator_name = getattr(estimator, "__name__", estimator.__class__.__name__)
    centers: Dict[int, FloatArray] = {}
    for fold in (1, 2):
        training_data = x[training_indices[fold]]
        estimate = estimator(training_data)
        centers[fold] = _validate_mean_vector(
            estimate,
            dimension,
            source=f"mean_estimator on fold {fold}",
        )
    return centers, str(estimator_name)


def build_clipped_covariance_family(
    x: ArrayLike,
    *,
    assume_centered: bool = False,
    known_mean: ArrayLike | None = None,
    mean_estimator: MeanEstimator | None = None,
    delta: float = 0.05,
    rho: float = 2.0,
    gamma_max: float = DEFAULT_GAMMA_MAX,
    random_state: int = 0,
) -> ClippedCovarianceFamily:
    """Build the cross-fitted covariance family before score minimization.

    When the mean is unknown, ``mean_estimator`` is called separately on each
    training fold.  Its fold-specific output centers both the training radii
    and the corresponding held-out observations.  If ``mean_estimator`` is
    omitted, :func:`sample_mean` is used.
    """

    data = _validate_data(x)
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1).")

    n_input, dimension = data.shape
    rng = np.random.default_rng(random_state)
    fold_one, fold_two = split_two_even_folds(n_input, rng)
    test_indices = {1: fold_one, 2: fold_two}
    training_indices = {1: fold_two, 2: fold_one}
    n_used = fold_one.size + fold_two.size

    gamma_grid = make_clipping_grid(
        min(fold_one.size, fold_two.size),
        rho=rho,
        gamma_max=gamma_max,
    )
    alpha_per_candidate = (delta / 2.0) / (4.0 * len(gamma_grid))

    mean_estimates, mean_method = _fold_mean_estimates(
        data,
        training_indices,
        assume_centered=assume_centered,
        known_mean=known_mean,
        mean_estimator=mean_estimator,
    )

    centered_test: Dict[int, FloatArray] = {}
    sorted_training_radii: Dict[int, FloatArray] = {}
    for fold in (1, 2):
        center = mean_estimates[fold]
        centered_test[fold] = data[test_indices[fold]] - center
        centered_training = data[training_indices[fold]] - center
        sorted_training_radii[fold] = np.sort(
            np.linalg.norm(centered_training, axis=1)
        )

    covariance_by_gamma: Dict[float, FloatArray] = {}
    variance_radius_raw: Dict[float, float] = {}
    clipping_radii: Dict[float, Tuple[float, float]] = {}
    tail_excess: Dict[float, Tuple[FloatArray, FloatArray]] = {}

    for gamma in gamma_grid:
        covariance_parts: List[FloatArray] = []
        variance_parts: List[float] = []
        radii_for_candidate: List[float] = []
        excesses_for_candidate: List[FloatArray] = []

        for fold in (1, 2):
            held_out = centered_test[fold]
            training_size = training_indices[fold].size
            number_above = int(math.floor(gamma * training_size))
            number_above = min(max(number_above, 0), training_size - 1)
            radius_index = training_size - number_above - 1
            radius = float(sorted_training_radii[fold][radius_index])

            clipped = clip_rows(held_out, radius)
            fold_covariance = (clipped.T @ clipped) / held_out.shape[0]
            weight = held_out.shape[0] / n_used
            covariance_parts.append(weight * fold_covariance)

            variance_proxy = paired_variance_proxy_operator_norm(
                clipped,
                radius,
            )
            normalized_radius = matrix_empirical_bernstein_radius(
                held_out.shape[0],
                dimension,
                alpha_per_candidate,
                variance_proxy,
            )
            variance_parts.append(weight * radius**2 * normalized_radius)

            squared_norms = np.sum(held_out * held_out, axis=1)
            excesses_for_candidate.append(
                np.maximum(squared_norms - radius**2, 0.0)
            )
            radii_for_candidate.append(radius)

        covariance_by_gamma[gamma] = np.add.reduce(covariance_parts)
        variance_radius_raw[gamma] = float(sum(variance_parts))
        clipping_radii[gamma] = (
            float(radii_for_candidate[0]),
            float(radii_for_candidate[1]),
        )
        tail_excess[gamma] = (
            excesses_for_candidate[0],
            excesses_for_candidate[1],
        )

    # The grid is interpreted in increasing gamma order.  A suffix maximum
    # therefore becomes a running maximum from large gamma to small gamma.
    variance_radius: Dict[float, float] = {}
    running_maximum = -float("inf")
    for gamma in sorted(gamma_grid, reverse=True):
        running_maximum = max(running_maximum, variance_radius_raw[gamma])
        variance_radius[gamma] = float(running_maximum)

    return ClippedCovarianceFamily(
        gamma_grid=gamma_grid,
        covariance_by_gamma=covariance_by_gamma,
        variance_radius_raw=variance_radius_raw,
        variance_radius=variance_radius,
        clipping_radii=clipping_radii,
        tail_excess=tail_excess,
        test_indices=test_indices,
        training_indices=training_indices,
        mean_estimates=mean_estimates,
        mean_method=mean_method,
        n_input=n_input,
        n_used=n_used,
        delta=float(delta),
        rho=float(rho),
        gamma_max=float(gamma_max),
        random_state=int(random_state),
    )


def _upper_block_quantile(block_means: FloatArray, quantile: float = 0.85) -> float:
    """Return order statistic ``ceil(quantile * number_of_blocks)``."""

    ordered = np.sort(np.asarray(block_means, dtype=float))
    number_of_blocks = int(ordered.size)
    if number_of_blocks == 0:
        return 0.0
    rank = int(math.ceil(quantile * number_of_blocks))
    rank = min(max(rank, 1), number_of_blocks)
    return float(ordered[rank - 1])


def score_clipped_covariance_family(
    family: ClippedCovarianceFamily,
    *,
    random_state: int | None = None,
) -> Tuple[float, Dict[float, float], Dict[float, float]]:
    """Evaluate the tail-energy score and return the selected candidate."""

    delta_bias = family.delta / 2.0
    grid_size = len(family.gamma_grid)
    base_number_of_blocks = int(
        math.ceil(4.0 * math.log(4.0 * grid_size / delta_bias))
    )

    seed = family.random_state + 1000 if random_state is None else random_state
    rng = np.random.default_rng(seed)
    score_by_gamma: Dict[float, float] = {}
    tail_score_by_gamma: Dict[float, float] = {}

    for gamma in family.gamma_grid:
        tail_score = 0.0
        for fold, excesses in zip((1, 2), family.tail_excess[gamma]):
            number_of_blocks = min(
                max(1, base_number_of_blocks),
                excesses.size,
            )
            shuffled = rng.permutation(excesses)
            blocks = np.array_split(shuffled, number_of_blocks)
            block_means = np.asarray(
                [np.mean(block) for block in blocks if block.size > 0],
                dtype=float,
            )
            fold_tail_score = _upper_block_quantile(block_means)
            fold_weight = family.test_indices[fold].size / family.n_used
            tail_score += fold_weight * fold_tail_score

        tail_score_by_gamma[gamma] = float(tail_score)
        score_by_gamma[gamma] = float(
            family.variance_radius[gamma] + tail_score
        )

    # Deterministic tie break: prefer the larger clipping fraction.
    selected_gamma = min(
        family.gamma_grid,
        key=lambda gamma: (score_by_gamma[gamma], -gamma),
    )
    return float(selected_gamma), score_by_gamma, tail_score_by_gamma


def estimate_covariance(
    x: ArrayLike,
    *,
    assume_centered: bool = False,
    known_mean: ArrayLike | None = None,
    mean_estimator: MeanEstimator | None = None,
    delta: float = 0.05,
    rho: float = 2.0,
    gamma_max: float = DEFAULT_GAMMA_MAX,
    random_state: int = 0,
) -> CovarianceEstimate:
    """Estimate a covariance matrix with a data-selected clipping level.

    Parameters
    ----------
    x:
        Data matrix with shape ``(n_samples, n_features)``.
    assume_centered:
        Set to ``True`` only when the input is already centered at the known
        population mean.  This skips mean estimation.
    known_mean:
        Optional known population mean with shape ``(n_features,)``.
    mean_estimator:
        Optional callable ``mean_estimator(training_data) -> mean_vector``.
        It is called independently on each training fold.  When omitted and
        the mean is unknown, the sample mean is used.
    delta:
        Confidence budget used by the simultaneous score construction.
    rho:
        Ratio of adjacent clipping fractions in the geometric grid.
    gamma_max:
        Largest candidate clipping fraction.
    random_state:
        Seed controlling the two-fold split and block partitions.

    Returns
    -------
    CovarianceEstimate
        The selected covariance and complete selection diagnostics.
    """

    family = build_clipped_covariance_family(
        x,
        assume_centered=assume_centered,
        known_mean=known_mean,
        mean_estimator=mean_estimator,
        delta=delta,
        rho=rho,
        gamma_max=gamma_max,
        random_state=random_state,
    )
    selected_gamma, scores, tail_scores = score_clipped_covariance_family(
        family
    )

    return CovarianceEstimate(
        covariance=family.covariance_by_gamma[selected_gamma],
        selected_gamma=selected_gamma,
        clipping_radii=family.clipping_radii[selected_gamma],
        score_by_gamma=scores,
        tail_score_by_gamma=tail_scores,
        variance_score_by_gamma=family.variance_radius.copy(),
        mean_estimates=(
            family.mean_estimates[1].copy(),
            family.mean_estimates[2].copy(),
        ),
        family=family,
    )


__all__ = [
    "DEFAULT_GAMMA_MAX",
    "ClippedCovarianceFamily",
    "CovarianceEstimate",
    "MeanEstimator",
    "sample_mean",
    "split_two_even_folds",
    "make_clipping_grid",
    "clip_rows",
    "paired_variance_proxy_operator_norm",
    "matrix_empirical_bernstein_radius",
    "build_clipped_covariance_family",
    "score_clipped_covariance_family",
    "estimate_covariance",
]
