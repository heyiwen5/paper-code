# Adaptive Radially Clipped Covariance Estimation

A self-contained NumPy implementation of a fully data-driven clipping-level
selector for covariance estimation. The method constructs a cross-fitted
family of Euclidean radially clipped covariance estimates and selects the
clipping fraction by combining:

1. a matrix empirical-Bernstein variance term; and
2. a block-quantile estimate of the tail energy removed by clipping.

The implementation supports centered observations, a known population mean,
and an unknown mean. In the unknown-mean case, mean estimation is performed
inside the cross-fitting scheme: each test fold is centered using an estimate
computed only from the other fold.

## Files

- `adaptive_clipped_covariance.py`: complete estimator and public API.
- `example_scm.py`: a small Student-t(5) simulation comparing the estimator
  with the sample covariance matrix (SCM).

## Requirements

- Python 3.10 or newer
- NumPy

Install NumPy if it is not already available:

```bash
python -m pip install numpy
```

## Quick start

The default assumes that the population mean is unknown and estimates it with
the sample mean separately on each training fold:

```python
import numpy as np

from adaptive_clipped_covariance import estimate_covariance

rng = np.random.default_rng(0)
x = rng.standard_t(df=5, size=(400, 50))

result = estimate_covariance(x, random_state=0)

covariance = result.covariance
print(result.selected_gamma)
print(result.clipping_radii)
```

### Supplying a different mean estimator

`mean_estimator` can be any callable that accepts a two-dimensional training
array and returns one finite vector of length `n_features`. The callable is
invoked independently on the two training folds.

```python
from functools import partial

from adaptive_clipped_covariance import estimate_covariance
from my_mean_package import robust_mean

my_estimator = partial(robust_mean, confidence=0.95)
result = estimate_covariance(x, mean_estimator=my_estimator)
```

For an estimator with a `fit` interface, use a small adapter:

```python
def external_mean(x_train):
    fitted = ImportedMeanEstimator().fit(x_train)
    return fitted.location_

result = estimate_covariance(x, mean_estimator=external_mean)
```

The estimator must use only the `x_train` argument passed to it. Any
additional tuning arguments can be supplied with a wrapper, a lambda, or
`functools.partial`.

### Known mean

If the population mean is known, pass it directly:

```python
known_mean = np.zeros(x.shape[1])
result = estimate_covariance(x, known_mean=known_mean)
```

If the input rows have already been centered at the known population mean:

```python
result = estimate_covariance(x_centered, assume_centered=True)
```

`assume_centered`, `known_mean`, and `mean_estimator` are mutually exclusive.

## Main API

```python
estimate_covariance(
    x,
    *,
    assume_centered=False,
    known_mean=None,
    mean_estimator=None,
    delta=0.05,
    rho=2.0,
    gamma_max=0.3,
    random_state=0,
)
```

The returned `CovarianceEstimate` contains:

- `covariance`: selected covariance estimate;
- `selected_gamma`: selected clipping fraction;
- `clipping_radii`: fold-specific clipping radii at the selected fraction;
- `mean_estimates`: fold-specific centers;
- `score_by_gamma`: total selection score;
- `tail_score_by_gamma`: tail-energy part of the score;
- `variance_score_by_gamma`: monotone variance part of the score; and
- `family`: all candidate estimates and cross-fitting diagnostics.

At most three rows are discarded so that the two held-out folds have equal,
even sizes. Set `random_state` to reproduce the sample split and the
block-quantile partitions.

## Random splitting and repeated-split summaries

`random_state` controls two parts of the algorithm:

1. A random permutation of the row indices is generated. The first
   `4 * floor(n / 4)` indices are retained and divided into two equal, even
   folds. Each fold is used once for testing and once for training.
2. Within every fold and clipping candidate, the held-out tail-excess values
   are randomly shuffled before being divided into blocks for the
   block-quantile score.

Consequently, using the same data and the same `random_state` reproduces the
same covariance estimate. Randomness inside a user-supplied
`mean_estimator` is not controlled automatically; seed that estimator in its
wrapper if reproducibility is required.

Under strong contamination, a particular split can occasionally place an
unfavorable concentration of contaminated observations in one fold and
produce an unusually large operator-norm result or estimation error. If the
quantity of interest is the scalar operator norm of the covariance, run
several splits and take the ordinary median of those scalar estimates:

```python
import numpy as np

from adaptive_clipped_covariance import estimate_covariance

results = [
    estimate_covariance(x, random_state=seed)
    for seed in range(101)
]

operator_norm_estimates = np.asarray([
    np.linalg.norm(result.covariance, ord=2)
    for result in results
])
median_operator_norm = float(np.median(operator_norm_estimates))
```

In a simulation where the true covariance `sigma_true` is known, the same
idea can be used to summarize split-dependent operator-norm errors:

```python
operator_norm_errors = np.asarray([
    np.linalg.norm(result.covariance - sigma_true, ord=2)
    for result in results
])
median_operator_norm_error = float(np.median(operator_norm_errors))
```

These two medians are scalar summaries. The first estimates
`||Sigma||_op`; the second summarizes estimation accuracy in a simulation.
Neither one returns a complete covariance matrix.

### If a complete covariance matrix is required

When the true covariance is unknown and the desired output is a complete
matrix, the scalar error cannot be computed. One possible aggregation is the
operator-norm medoid

```text
Sigma_medoid = argmin_j sum_k ||Sigma_hat_j - Sigma_hat_k||_op.
```

This means: among the covariance matrices obtained from the repeated splits,
select the one having the smallest total operator-norm distance to all the
others. For example, if most split estimates form a tight cluster and one
split produces an extreme matrix, the extreme matrix is far from almost
every other estimate and will not be selected.

It can be computed as follows:

```python
covariances = [result.covariance for result in results]
total_distances = np.asarray([
    sum(np.linalg.norm(left - right, ord=2) for right in covariances)
    for left in covariances
])
medoid_result = results[int(np.argmin(total_distances))]
medoid_covariance = medoid_result.covariance
```

The matrix medoid is different from the median of the scalar operator norms.
It is useful only when a representative full covariance estimate is needed.
Because all repeated runs use the same observed sample, repeated splitting
is a practical stabilization device rather than an independent
confidence-amplification guarantee.

## SCM comparison

Run the included unknown-mean experiment:

```bash
python example_scm.py
```

Optional arguments control the problem size:

```bash
python example_scm.py --n 300 --d 40 --trials 20 --seed 0
```

The experiment uses a nonzero mean, a spiked covariance matrix, and
standardized Student-t(5) coordinates. The SCM baseline estimates the mean
from the full sample and uses normalization by `n`. No external estimator or
experiment package is required.

## Notes

- Larger `gamma` means a smaller clipping radius and more aggressive
  clipping.
- The default sample mean is suitable as a simple clean-data default, but it
  is not a robust location estimator. For contaminated observations, supply
  a robust mean estimator through `mean_estimator`.
- The output is positive semidefinite up to floating-point roundoff because
  every candidate is an average of clipped outer products.
- Statistical guarantees require the assumptions associated with the
  variance and tail-energy bounds; the software does not diagnose those
  assumptions from the observed sample.
