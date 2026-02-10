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
