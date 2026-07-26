# refshift

Reference-mismatch experiments for motor-imagery EEG decoding on **BCI
Competition IV-2a**.

The choice of EEG **reference** (how each channel's baseline is defined) is a
preprocessing decision that is usually treated as fixed. This package measures
what happens when a decoder is trained under one reference and tested under
another: how large the resulting accuracy gap is, whether it clusters by the
*kind* of reference operation, and whether training-time reference jitter makes
a model invariant to the choice.

## Results

Fill these in after running `notebooks/refshift-decoder-clean.ipynb`. Each is a mean over
subjects (and seeds, for the deep nets) on IV-2a.

| Model        | Matched (diag) | Mismatched (off-diag) | Transfer gap |
|--------------|---------------:|----------------------:|-------------:|
| CSP+LDA      |                |                       |              |
| ShallowConvNet |              |                       |              |
| EEGNet       |                |                       |              |
| ATCNet       |                |                       |              |

| Intervention            | Mean acc | Worst reference | Spread |
|-------------------------|---------:|----------------:|-------:|
| Full jitter (Shallow)   |          |                 |        |
| LORO recovery gap       |          |                 |        |
| LOFO family recovery gap|          |                 |        |

## Repository structure

```
refshift/
  datasets.py     Registry of all five datasets (MOABB codes, classes, channel
                  subsets, split strategies) plus loaders for both pipelines.
  preprocess.py   The preprocessing pipeline: pick EEG, [channel subset], uV,
                  resample, bandpass, z-score, window. Disk-cached per subject.
                  Plus the train/test split (session- or run-based per dataset).
  references.py   The seven reference operators, their families, and Euclidean
                  Alignment (rank-aware whitening).
  inversion.py    Operator-invertibility control: is each operator canonicalizable
                  by re-referencing, and are its contrasts recoverable + how well
                  conditioned (contrast_recovery_report). Data-free, algebraic.
  models.py       Model factories: CSP+LDA, ShallowConvNet, EEGNet, ATCNet.
  jitter.py       Per-sample reference-jitter data augmentation.
  experiments.py  run_mismatch, run_mismatch_jitter, run_loro_matrix,
                  run_lofo_matrix, calibrate_csp_lda (all take dataset_id first).
  analysis.py     Matrix pivots and the report_* printers used in the notebook.
tests/            Unit tests (datasets, operators, families, sweeps, reports).
scripts/
  verify_equivalence.py  Diff the lean pipeline against the original repo.
  inversion_control.py   Print the operator-invertibility table (IV-2a or --schirr).
notebooks/
  refshift-decoder-clean.ipynb        CSP+LDA / Shallow / EEGNet / ATCNet experiments.
  reve-reference-mismatch-clean.ipynb REVE frozen-probe experiments.
```

### The seven references

`native` (no change), `car` (common average), `median` (robust average),
`cz_ref` (single-electrode), `lap_small` / `lap_large` (Laplacians of near and
far neighbours), `rest` (head-model reference). Families for the
leave-one-family-out experiment: **global** = {car, median, rest}, **single** =
{cz_ref}, **spatial** = {lap_small, lap_large}.

## Install

```bash
pip install -e ".[dl]"
```

The deep-learning extras (`braindecode`, `torch`, `skorch`) are needed for every
model except CSP+LDA. `moabb` is pinned to 1.5.0 because the CSP+LDA calibration
target was verified at that version.

## Reproduce

1. Open `notebooks/refshift-decoder-clean.ipynb`.
2. Section A: install the package (point `REFSHIFT_SRC` at the attached Kaggle
   dataset folder), then call `setup_kaggle_env()`. This sets `MNE_DATA` and
   thread caps and **symlinks your attached Kaggle datasets into MOABB's cache
   layout so nothing downloads**. Run it once per kernel, before any data load.
3. Section C is the calibration sanity check: bare CSP+LDA should reproduce the
   standard IV-2a baseline, and a no-op reference transformer should change
   nothing.
4. Sections D-I produce the mismatch matrices, jitter, LORO, and LOFO results.

### Kaggle data wiring

`refshift/kaggle.py` symlinks the attached Kaggle datasets into the layout MOABB
expects, handling each dataset's quirks (OpenBMI's flat layout, Dreyer2023's
`mne_bids` lock files on a read-only mount, Schirrmeister's pooch paths). The
hardcoded `/kaggle/input/...` source paths live at the top of that file; edit
them if your attached dataset slugs differ.

## Verifying against the original repo

`scripts/verify_equivalence.py` checks that this lean package reproduces the
original implementation: `check_preprocessing` confirms the preprocessed trials
are identical, and `diff_result_csvs` compares result tables cell by cell. The
reference operators are byte-identical to the original by construction (the
unit tests prove this), so matching preprocessing implies matching results up
to GPU/seed nondeterminism.

## Datasets

Five MOABB motor-imagery datasets. Each runner takes ``dataset_id`` first.

| dataset_id          | MOABB code       | classes | split   | notes |
|---------------------|------------------|--------:|---------|-------|
| `iv2a`              | BNCI2014_001     | 4       | session | 22 ch, 2 sessions |
| `openbmi`           | Lee2019_MI       | 2       | session | needs a compat fix to load both sessions |
| `cho2017`           | Cho2017          | 2       | session | |
| `dreyer2023`        | Dreyer2023       | 2       | session | |
| `schirrmeister2017` | Schirrmeister2017| 4       | run     | 44-ch motor subset; no Cz, so cz_ref is excluded |

Each dataset always uses its full class set. Per-dataset behaviour lives in one
config table (`DATASETS` in `datasets.py`); adding a sixth dataset means adding
one `DatasetSpec` there.
