# paper-code

This repository provides the core implementation and experiment scripts for the paper **Computable Bernstein Certificates for Cross-Fitted Clipped Covariance Estimation**.

You can view it as serving two purposes:

1. **Reproduce experiments**: run `benchmark.py` / `benchmark_robustness_heavytails.py` / `picture.py` to generate tables and figures.
2. **Reuse the algorithm**: directly import `robust_covariance(...)` from `robust_covariance.py` and treat it as an “automatic clipping-level robust covariance estimator”.

> About *robustness*: the paper/experiments mainly emphasize two kinds of “robustness”
> - **Model robustness**: even if the distribution does not satisfy ideal assumptions (heavier tails, non-elliptical, or even without a finite fourth moment), the results will not collapse overnight.
> - **Outlier robustness**: even with a small fraction of outliers / Huber contamination, the estimator remains usable.

---

## Table of contents

- [Quick start (only want to reproduce the paper)](#quick-start-only-want-to-reproduce-the-paper)
- [Install dependencies](#install-dependencies)
- [Script overview](#script-overview)
- [How to choose parameters (with advanced tips)](#how-to-choose-parameters-with-advanced-tips)
- [Convert CSV into paper-style LaTeX tables](#convert-csv-into-paper-style-latex-tables)
- [FAQ](#faq)

---

## Quick start (only want to reproduce the paper)

> Goal: with minimal effort, reproduce the main tables used in the paper (and the additional heavy-tail experiments).

1) It’s recommended to create a virtual environment first (choose either approach):

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

2) Install the “minimal dependencies” (ensures the scripts run and output CSV; baseline methods with missing deps will show fail):

```bash
pip install -U numpy pandas scipy matplotlib
```

3) If you want **all baselines to run fully** (recommended for reproducing the complete tables in the paper):

```bash
pip install -U scikit-learn openpyxl
pip install -U tfHuber
pip install -U robpy
pip install -U pyriemann
```

4) Run the main benchmark (source of the paper’s main tables + all appendix tables):

```bash
python benchmark.py
```

Outputs are written to `./benchmark_outputs/`:
- `summary_<dist>_eps<eps>.csv` (one table per distribution × contamination rate)
- `benchmark_tables.xlsx` (merges all tables into a single Excel file)

5) Run the extra robustness experiment for “extremely heavy tails but no contamination” (finite variance, but no finite fourth moment):

```bash
python benchmark_robustness_heavytails.py
```

Outputs are written to `./heavytail_outputs/`.

6) If you also want the figures (only for our methods: Lepski / MinUpper):

```bash
python picture.py
```

It generates two PNGs (filenames are fixed in the script).

---

## Install dependencies

These scripts are intentionally written to “run as long as they can”, so dependencies are split into two tiers:

### 1) Minimal tier (run only our methods + SCM, produce CSV/images)

- `numpy`, `pandas`, `scipy`
- For plotting: `matplotlib`

```bash
pip install -U numpy pandas scipy matplotlib
```

### 2) Full tier (run all comparison methods + write Excel)

- `openpyxl`: write `benchmark_tables.xlsx`
- `scikit-learn`: `MinCovDet`
- `tfHuber`: `tfHuber.cov(...)`
- `robpy`: `OGK`, `WrappingCovariance`
- `pyriemann`: `pyRiemann-Stu` (M-estimator, Student-t version)

```bash
pip install -U scikit-learn openpyxl tfHuber robpy pyriemann
```

> Note: if some optional dependency is missing, the script will not crash entirely, but that method’s row will show `fail_rate=1` or metrics as NaN (equivalent to “baseline not installed / didn’t run”).

---

## Script overview

### 1) `robust_covariance.py` (core algorithm)

Provides:

```python
from robust_covariance import robust_covariance

res = robust_covariance(X, selection="minupper")
Sigma_hat = res.Sigma_raw
```

In addition to the final `Sigma_raw`, the returned `res` also includes:
- `grid`: candidate tail-probabilities (equivalently, a grid of candidate clipping quantiles)
- `Psi_raw` / `Psi_raw_monotone`: computable empirical Bernstein variance envelope
- `bias_proxy`: only available when `selection="minupper"` (a proxy for clipping bias)
- `Sigma_raw_path`: all candidate estimator matrices along the grid (useful for diagnostics/plotting curves)
- `folds` / `dropped_indices`: details of the cross-fitting split

### 2) `benchmark.py` (main benchmark: heavy tails + Huber contamination)

- Generates tables for: 4 distributions × 3 contamination rates (default eps=0/0.05/0.1).
- Metrics:
  - `rel_op_cov`: relative operator-norm error for covariance
  - `rel_op_corr`: relative operator-norm error for correlation (with `tau0` stabilization)
  - `subspace_err`: principal component subspace error (rank-r projector)
  - `top_r_eig_rel`: relative error of the top r eigenvalues
  - `time_sec`: runtime
  - `fail_rate`: failure rate for the method (exceptions / missing deps / numerical breakdown, etc.)

Default methods include:
- Ours: `Ours-Lepski`, `Ours-MinUpper`
- Baselines: `SCM`, `tfHuber`, `MinCovDet`, `OGK`, `Wrapping`, `pyRiemann-Stu`

Run:

```bash
python benchmark.py
```

### 3) `benchmark_robustness_heavytails.py` (extremely heavy tails, no contamination)

This script compares only:
- `Ours-Lepski`, `Ours-MinUpper`, `SCM`

It uses two “finite-variance but no finite fourth moment” stress-test distributions:
- Elliptical Student-t (df=3)
- Non-elliptical Signed-F (df_den=6)

Run:

```bash
python benchmark_robustness_heavytails.py
```

### 4) `picture.py` (figures: only our methods)

This script performs two sweeps:
- Fix dimension d, sweep sample size n
- Fix sample size n, sweep dimension d

Run:

```bash
python picture.py
```

> Friendly note: some default numbers in `picture.py` (e.g., `d0=500`, `n0=500`) do not perfectly match the filenames (`picture_fixed_d600.png` / `picture_fixed_n600.png`).
> If you want them to match, just change `d0/n0` in the script to what you want and also update the output filenames.

---

## How to choose parameters (with advanced tips)

This section is for two kinds of users:

- **Reproduction-only**: you just want the paper’s numbers — use default parameters.
- **Tinkerers (advanced users)**: you want to apply the method to your own data, or tune it to be faster/more stable.

### A) Key parameters of `robust_covariance(X, ...)`

Function signature (key parts):

```python
robust_covariance(
    X,
    delta=0.05,
    K=5,
    rho=2.0,
    selection="lepski" or "minupper",
    c_bias=3.0,
    center=False,
    random_state=0,
    norm_method="power" or "exact",
    v_opnorm_method="power" or "exact" or "bound",
)
```

- `delta`: overall failure-probability budget (smaller = more conservative; usually “more stable but possibly less sharp”).
- `K`: number of folds for cross-fitting. Default `K=5` is a balanced choice.
  - Larger K: finer train/test splitting and more usable information, but smaller folds, heavier computation, and in small samples it may violate minimum sample requirements.
- `rho`: geometric ratio for the grid (density of candidate tail-probabilities). Default 2.
  - For a finer grid: use a smaller rho (e.g., 1.5), but it will be slower.
- `selection`: two automatic tuning rules
  - `"lepski"`: more “variance control + robust comparison” in spirit.
  - `"minupper"`: minimizes an upper bound (computable variance envelope + bias proxy); often closer to a default when you care about practical performance.
- `c_bias`: only matters for `minupper`, controlling the weight of the bias proxy.
  - **Rule of thumb**:
    - Not too extreme data: `c_bias=1` is often more aggressive and yields smaller error;
    - Extremely heavy tails / worried about clipping bias: increase it (e.g., 3).
- `center`: whether to subtract the mean first.
  - Many synthetic datasets in the paper/scripts are mean-zero, so default is `False`.
  - **Real data**: usually recommended to set `center=True` (unless you have already centered).
- `norm_method`: how to compute the operator norm in Lepski comparisons
  - `"exact"`: uses `eigvalsh`, reproducible but O(d^3); suitable for moderate d.
  - `"power"`: power-method approximation, faster but approximate; suitable for large d.
- `v_opnorm_method`: how to compute \|V*\|_op in the empirical Bernstein envelope
  - `"exact"`: most stable but slower.
  - `"power"`: default (speed/stability trade-off).
  - `"bound"`: uses a generic upper bound 1/2 (valid but more conservative; may choose overly conservative parameters).

**Small-sample / high-dimensional notes**:
- The code forces each fold’s sample count to be even (for paired construction of V*), and may drop a small number of samples if needed (see `dropped_indices` in the return).
- Internally it requires `min_k |J_k| >= 4` (each fold’s training set must have at least 4 points).
  - If you hit an error: reduce K or increase n.

### B) Where to change parameters in `benchmark.py`?

At the top of `benchmark.py` there is a `BenchConfig` dataclass—edit that:

- `n, d, r, theta`: scale and strength of the spiked covariance model
- `reps, seed`: repetitions and random seed
- `delta, K, rho, c_bias, selection`: settings for our method
- `t_df=4.5`: degrees of freedom for elliptical t (script default 4.5)
- `eps_list=(0,0.05,0.1)`: list of Huber contamination levels
- `kappa=100`: outlier strength (contamination points ~ N(0, kappa Σ))

**Tips to run faster**:
- Set `reps` to 1 first (confirm everything works, then go back to 3 or more).
- Use `norm_method="power"` / `v_opnorm_method="power"` (already set that way in the script).
- For large d, some baselines (e.g., MCD) can be very slow—this is expected.

### C) Parameters in `benchmark_robustness_heavytails.py`

Similarly, look for a `Config` dataclass in that script. It is a stress test for “extremely heavy tails without contamination”, so it keeps only `Ours-*` and `SCM`.

---

## Convert CSV into paper-style LaTeX tables

The scripts output CSV (and optionally Excel). If you want to paste into LaTeX, the simplest approach is to use pandas:

```python
import pandas as pd

df = pd.read_csv("benchmark_outputs/summary_ellip_t_eps0p1.csv", index_col=0)
# you can also select only the mean columns
mean_cols = [c for c in df.columns if c.endswith('_mean')]
print(df[mean_cols].to_latex(float_format=lambda x: f"{x:.3g}"))
```

If you prefer copying from Excel, you can directly open `benchmark_tables.xlsx`.

---

## FAQ

### Q1: I didn’t install tfHuber / robpy / pyriemann — will the script crash?

Usually no. `benchmark.py` catches exceptions per method and counts them into `fail_rate`. When a dependency is missing, that method’s row is typically NaN with `fail_rate=1`.

### Q2: Should I choose `exact` or `power`?

- If you want results that are “more reproducible numerically” (especially for paper tables): `exact`.
- If you want to run larger dimensions faster: `power`.

### Q3: Why does `minupper` need `c_bias`?

`minupper` balances a “computable upper bound”: variance term (certificate) + bias proxy (clipping bias proxy). `c_bias` tells the algorithm how averse you are to “bias risk”.

### Q4: I want to align the output tables to my own paper format—any recommended workflow?

Recommended: **write all results to CSV/Excel first, then use a separate script just for formatting** (e.g., unified rounding, bold the best, grouped headers, etc.). This keeps the main benchmark script clean and avoids mixing in formatting logic.

---

## Citation

If you find this code useful, feel free to cite the paper and also cite this repository (you can add a bibtex entry in the README).
