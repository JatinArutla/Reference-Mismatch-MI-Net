# refshift

A benchmark for reference-induced distribution shift in EEG motor imagery
decoding. Studies how re-referencing (CAR, Laplacian, bipolar, etc.)
creates a systematic distribution shift between training and test that
causes cross-reference transfer to collapse to near chance.

## What this repo does

- Loads four MI datasets with identical preprocessing: IV-2a (9 subjects,
  4-class), OpenBMI / Lee2019 (53 subjects, 2-class), Cho2017 (52 subjects,
  2-class), Dreyer2023 (87 subjects, 2-class).
- Applies six reference operators: native, CAR, median, GS (Gram-Schmidt),
  Laplacian-kNN4, and bipolar-nearest.
- Evaluates a 6x6 train-reference by test-reference mismatch matrix using
  ATCNet (braindecode) and CSP+LDA (MOABB canonical pipeline).
- Supports reference jitter as a training-time augmentation.

## File layout

```
loader.py            Subject-level loading (all 4 datasets -> epoched arrays)
preprocessing.py     Zero-phase Butterworth bandpass
reference_ops.py     Six reference operators + graph construction
standardization.py   Mechanistic + deployment standardization protocols
models.py            ATCNet (braindecode) and CSP+LDA (pyriemann)
training.py          Fit / evaluate / jitter batch factory
experiments.py       6x6 mismatch matrix + jitter runners, aggregation

configs/
  paper.yaml             Main benchmark spec (datasets, hyperparams, seeds)
  reference_ops.yaml     Formal definitions of the 6 operators
  standardization.yaml   Mechanistic vs deployment protocols
  seeds.yaml             Seed assignments

tests/
  test_loader.py         Loader correctness (shape, dtype, trial counts)
  test_preprocessing.py  Bandpass correctness (passband, stopband, zero-phase)
  test_refs_and_std.py   Reference-op identities + standardization
  test_chunk4.py         Model forward pass + ATCNet/CSP+LDA training
  test_experiments.py    End-to-end experiment runners

requirements.txt
```

## Setup on Kaggle

1. Upload this repo as a Kaggle Dataset (or put the files in an existing
   dataset), say at `/kaggle/input/refshift/`.

2. Attach the four source datasets to your notebook:
   - `delhialli/four-class-motor-imagery-bnci-001-2014` (IV-2a)
   - `imaginer369/openbmi-dataset` (OpenBMI)
   - `delhialli/cho2017` (Cho2017)
   - `delhialli/dreyer2023` (Dreyer2023)

   Default paths assumed by the loader:
   - IV-2a: `/kaggle/input/datasets/delhialli/four-class-motor-imagery-bnci-001-2014`
   - OpenBMI: `/kaggle/input/datasets/imaginer369/openbmi-dataset`
   - Cho2017: `/kaggle/input/datasets/delhialli/cho2017`
   - Dreyer2023: `/kaggle/input/datasets/delhialli/dreyer2023/MNE-Dreyer2023-data`

   Override any of these via environment variables `REFSHIFT_OPENBMI_ROOT`
   and `REFSHIFT_DREYER_ROOT`; IV-2a and Cho2017 are MOABB-managed via
   symlinks that `loader._ensure_moabb_cache_symlinks()` sets up automatically.

3. Set the Kaggle accelerator to **GPU T4 x2** (not P100). P100 is sm_60
   and the current PyTorch lacks compatible kernels. `training.py` probes
   CUDA on startup and falls back to CPU if kernels fail, so P100 will run
   but will be much slower than T4.

4. Install missing dependencies:
   ```python
   !pip install braindecode --quiet
   ```
   Everything else (torch, mne, moabb, pyriemann, sklearn, pandas, numpy,
   scipy) is pre-installed on Kaggle as of 2026.

5. Import the modules:
   ```python
   import sys
   sys.path.insert(0, '/kaggle/input/refshift')
   from experiments import run_mismatch_matrix, run_dataset_benchmark
   ```

## Running the main benchmark

Single subject, headline result:

```python
from experiments import run_mismatch_matrix

df = run_mismatch_matrix(
    "iv2a", subject=1,
    model_type="atcnet",
    standardization="mechanistic",
    n_epochs=200, batch_size=32, seed=0, verbose=True,
)
print(df.head())
df.to_csv("iv2a_sub1_seed0.csv", index=False)
```

Full dataset:

```python
from experiments import run_dataset_benchmark

df = run_dataset_benchmark(
    "iv2a",
    subjects=None,          # None -> all subjects
    seeds=[0, 1, 2],
    model_type="atcnet",
    standardization="mechanistic",
    n_epochs=200,
)
df.to_csv("iv2a_full.csv", index=False)
```

Aggregated 6x6 matrix:

```python
from experiments import mismatch_matrix_mean, mismatch_matrix_std

mean_mat = mismatch_matrix_mean(df, metric="accuracy")
std_mat  = mismatch_matrix_std(df,  metric="accuracy")
print(mean_mat.round(3))
```

## Running jitter

```python
from experiments import run_jitter

df = run_jitter(
    "iv2a", subject=1,
    training_refs=["native", "car", "laplacian"],
    n_epochs=200, seed=0,
)
print(df[["training_refs", "test_ref", "accuracy"]])
```

## Reproducing the paper

The exact spec lives in `configs/paper.yaml`. Defaults in the runners
match it. Full-scale reproduction: run `run_dataset_benchmark` for each
of the four datasets with seeds `[0, 1, 2]` and each of the two models
(`atcnet`, `csp_lda`). Expected GPU time on T4: ~3-5 hours for IV-2a,
longer for OpenBMI (more subjects, wider networks).

## Compute budgeting

For IV-2a with 200 epochs, one ATCNet training on T4 takes ~40 seconds.
Per subject that's 6 trainings + 36 evaluations, or ~5 minutes.
Nine subjects times three seeds is ~4 hours of T4 time. Kaggle's 30-hour
weekly T4 budget is enough for ATCNet on all four datasets single-seeded,
or the two-class datasets three-seeded. Budget accordingly.

## Tests

```python
# Replace `from tests.FOO import run_all` with a flat import if you're
# running inside the Kaggle notebook with sys.path set to the repo root.

from test_loader        import run_all as test_loader_all
from test_preprocessing import run_all as test_preprocessing_all
from test_refs_and_std  import run_all as test_refs_all
from test_chunk4        import run_all as test_models_all
from test_experiments   import run_all as test_experiments_all

for fn in [test_loader_all, test_preprocessing_all, test_refs_all,
           test_models_all, test_experiments_all]:
    failed = fn()
    assert not failed
```

Test runtime on T4: ~8-10 minutes total for all five suites.

## Methodology notes for the paper

- **Trials kept**: We keep artifact-flagged trials (`reject_by_annotation=False`
  in MNE). This matches the convention of ATCNet, EEGNet, and braindecode for
  fair deep-learning comparisons. CSP+LDA numbers will run ~5-15 points below
  MOABB leaderboard because MOABB drops artifacts.

- **IV-2a trial count**: 282 per session (not 288). MOABB drops the final
  boundary trial of each of 6 runs because the `[2, 6]` window extends past
  the run's recording end. Matches braindecode behavior.

- **OpenBMI exclusion**: Subject 29's `sess01` file is corrupt in the
  Kaggle copy. All other 53 subjects are included.

- **Cho2017 scale**: Raw Biosemi data is ~1000x larger amplitude than
  typical EEG (~20 mV vs ~20 uV). Within-dataset standardization normalizes
  this; cross-dataset comparisons need explicit handling.

- **Montage**: All channel positions come from MNE's `standard_1005`. Every
  channel across the four datasets is present in this montage.

## Dependencies

See `requirements.txt`. Minimum versions:
- torch >= 2.0
- braindecode >= 0.8
- mne >= 1.5
- moabb >= 1.0
- pyriemann >= 0.5
- scikit-learn >= 1.2
- pandas >= 1.5
- numpy >= 1.23
- scipy >= 1.10
