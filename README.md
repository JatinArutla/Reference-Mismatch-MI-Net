# refshift

Reference-mismatch experiments for motor-imagery EEG decoding.

Every EEG voltage is measured against some baseline, and the choice of that
baseline (the **reference**) is a preprocessing decision usually treated as
fixed. This package measures what happens when a decoder is trained under one
reference and tested under another: how large the accuracy gap is, whether it
clusters by the *kind* of reference operation, and whether training-time
reference jitter makes a model invariant to the choice.

## Install

```bash
pip install -e .
```

## Quick start

```python
from refshift import setup_kaggle_env, run_mismatch, report_matrix

setup_kaggle_env(symlink_datasets=["iv2a"])          # before any data load

df = run_mismatch("iv2a", model="shallow", seeds=[0, 1, 2],
                  cache_dir="cache", results_dir="results")
report_matrix(df, title="ShallowConvNet -- iv2a")
```

`results_dir` makes each runner write its table to CSV and reload it on a
rerun, so an interrupted session resumes instead of retraining. `cache_dir`
does the same for preprocessed windows.

## The seven references

`native` (no change), `car` (common average), `median` (robust average),
`cz_ref` (single electrode), `lap_small` / `lap_large` (Laplacians over near
and far neighbours), `rest` (head-model reference, Yao 2001).

Families for the leave-one-family-out experiment:
**global** = {car, median, rest}, **single** = {cz_ref},
**spatial** = {lap_small, lap_large}. `native` is the no-op baseline and
belongs to no family.

## Layout

```
refshift/
  datasets.py     One table of dataset configs, plus the loaders.
  preprocess.py   Raw subject -> windowed trials, and the train/test split.
  references.py   The seven operators, the neighbour graph, Euclidean Alignment.
  inversion.py    Data-free algebraic control: is an operator canonicalizable by
                  re-referencing, and are its contrasts linearly recoverable?
  models.py       CSP+LDA, ShallowConvNet, EEGNet, ATCNet.
  jitter.py       Per-sample reference jitter (the intervention).
  experiments.py  The four runners and the calibration check.
  analysis.py     Matrices, the bootstrap transfer gap, and the report_* printers.
  kaggle.py       Symlink attached Kaggle datasets into MOABB's cache layout.
notebooks/
  refshift-decoder.ipynb      CSP+LDA / Shallow / EEGNet / ATCNet experiments.
  reve-reference-mismatch.ipynb  REVE frozen-probe experiments.
scripts/
  inversion_control.py        Print the operator-invertibility table.
tests/
```

## Datasets

| dataset_id | MOABB code | classes | train / test split | notes |
|---|---|---:|---|---|
| `iv2a` | BNCI2014_001 | 4 | session `0train` / `1test` | 22 ch |
| `openbmi` | Lee2019_MI | 2 | session `0` / `1` | 62 ch |
| `schirrmeister2017` | Schirrmeister2017 | 4 | run `0train` / `1test` | 44-ch motor subset; no Cz, so `cz_ref` is excluded |
| `dreyer2023` | Dreyer2023 | 2 | 2 acquisition runs / 4 online runs | 27 ch |
| `cho2017` | Cho2017 | 2 | **none** | one session, one run: no held-out block, so it is not currently runnable |

Two loading quirks are load-bearing and documented in the code:

- **OpenBMI** defaults to `sessions=(1, 2)` but writes session keys `'0'` and
  `'1'`, so MOABB's own filter silently drops half the trials. `_make_openbmi`
  bypasses the filter. Do not simplify it away.
- **Kaggle sources** are looked up by dataset slug. If a slug is missing,
  `setup_kaggle_env` now raises rather than letting MOABB download gigabytes.
  Override a path with `REFSHIFT_IV2A_ROOT`, `REFSHIFT_OPENBMI_ROOT`,
  `REFSHIFT_SCHIRRMEISTER_ROOT` or `REFSHIFT_DREYER_ROOT`.

## Reading the output

`report_matrix` prints five blocks.

- `[A] pooled gap` pools every row, then takes diagonal minus off-diagonal.
- `[D] TRANSFER GAP` computes the gap per subject on that subject's own
  seed-averaged matrix, then bootstraps over subjects. Seeds are repeated runs
  of one subject, not independent samples, so they are averaged first. **This
  is the number to report**, because it carries a confidence interval.

In a balanced design `[A]` and `[D]` give the *same point estimate*: the mean of
per-subject gaps is algebraically equal to the pooled gap. They are not redundant
and they are not two competing numbers. If they ever disagree, a subject is
missing cells, so treat a divergence as a data alarm.

- `[B]` per-test-reference view, `[C]` per-cell spread, `[C2]` asymmetry. The
  matrix is directional: on IV-2a, CSP+LDA transfers `native -> median` at 61.3%
  and `median -> native` at 28.7%. `[C2]` reports the mean `|M - M^T|` and the
  worst pair.

Anchored gaps are a different estimator. `report_matrix` averages over all
train-references; the depth and robustness experiments in the foundation-model
notebooks train one probe on a single anchor reference. The two numbers are not
expected to agree, and a mismatch between them is not an inconsistency.

`report_loro` needs the full-jitter table to measure holdout cost correctly:

```python
report_loro(df_loro, title="LORO", full_jitter=df_jitter)
```

Without it you get `naive_cost_%`, which compares a held-out reference against
the same model's other references and so conflates "we never trained on it" with
"it is intrinsically harder". With it you also get `true_cost_%`, measured
against the full-jitter model, which differs only in that one reference. On
IV-2a the two orderings disagree: `lap_small` looks like the fourth-cheapest
holdout under `naive_cost_%` and is the third most expensive under
`true_cost_%`.

## Standardisation

The deep nets z-score per channel per trial after referencing; CSP+LDA does not,
because it is covariance-based and calibrated against MOABB. That is the only
pipeline difference between the two model families, and the two disagree most on
`native`, where the shared common-mode component is largest. `run_mismatch` takes
a `zscore` argument (defaulting to the per-family behaviour above) so you can test
whether a result is about referencing or about standardisation:

```python
run_mismatch("iv2a", model="shallow", zscore=False, ...)
```

Overriding it writes to a separate results filename, so it cannot overwrite an
existing sweep.

## Calibration

`calibrate_csp_lda("iv2a")` runs bare CSP+LDA through MOABB's own
`WithinSessionEvaluation` and checks it lands within 2 points of MOABB's
published 65.99%. It uses MOABB's loaders, not this package's, so it is an
independent check. Run it across all nine subjects; a subset will not match the
published mean.
