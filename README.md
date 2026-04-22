# refshift

Reference-shift experiments for motor-imagery (MI) EEG decoding, built on top
of [MOABB](https://github.com/NeuroTechX/moabb).

**Empirical claim:** a classifier trained under one EEG reference operator
(CAR, Laplacian, REST, ...) and tested under another suffers a structured,
predictable accuracy collapse. We measure it across four MOABB motor-imagery
datasets and seven reference operators using the CSP+LDA pipeline from
MOABB's canonical `CSP.yml`.

Phase 1 (this drop) — seven reference operators, MOABB-matching CSP+LDA
pipeline, four-dataset mismatch matrix, calibration against MOABB's published
numbers, 19 passing unit tests.

Phase 2 (planned) — EEGNet / ShallowFBCSPNet / ATCNet via
[braindecode](https://github.com/braindecode/braindecode), full-jitter
training intervention, SSL with reference-pair positives. Not in this drop.

## Reference operators

| Family | Operator | What it does |
|---|---|---|
| Global-mean | `native` | identity (whatever the dataset was recorded with) |
|  | `car` | X − mean across channels |
|  | `median` | X − median across channels |
|  | `gs` | X − leave-one-out mean projection (Gram-Schmidt) |
| Spatial-differential | `laplacian` | X − mean of k=4 nearest neighbours |
|  | `bipolar` | X − single nearest neighbour |
| Source-model | `rest` | Yao 2001 Reference Electrode Standardization Technique (estimates the potential at infinity via a three-layer spherical head model) |

Nearest-neighbour graphs are computed from MNE's `standard_1005` montage,
which covers every EEG channel in the four datasets. REST is built from the
same montage via `mne.make_sphere_model` + `mne.make_forward_solution` —
10–60 seconds per dataset, computed once and cached.

## Package layout

```
Reference-Mismatch-MI-Net/
├── refshift/
│   ├── __init__.py       public API
│   ├── reference.py      7 operators, neighbour graph, REST matrix, sklearn transformer
│   ├── pipelines.py      make_csp_lda_pipeline (matches MOABB CSP.yml verbatim)
│   ├── experiments.py    calibrate_csp_lda, run_mismatch, mismatch_matrix
│   ├── env.py            Kaggle setup + per-dataset cache symlinks + Dreyer patch
│   └── plotting.py       plot_mismatch_matrix (viridis 0–100, diagonal boxes)
├── tests/
│   └── test_reference.py    19 pure-numpy tests, no MOABB/network needed
├── pyproject.toml
├── requirements.txt
├── LICENSE
└── README.md
```

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

MOABB is pinned to **1.5.0**. Loosen only after re-verifying the 65.99 ± 15.47
IV-2a CSP+LDA calibration target against MOABB's published benchmark.

## Tests

```bash
pytest tests/ -v
```

Expect **19/19 passing**. None requires MOABB or network access.

## Kaggle usage

Clone → install → setup → calibrate → run. The notebook API is the intended
entry point; there is no CLI.

### Cell 1 — clone, install, environment

```python
import os, pathlib, subprocess

os.chdir("/kaggle/working")
if pathlib.Path("Reference-Mismatch-MI-Net").exists():
    subprocess.run(["rm", "-rf", "Reference-Mismatch-MI-Net"], check=True)
subprocess.run(
    ["git", "clone", "https://github.com/JatinArutla/Reference-Mismatch-MI-Net"],
    check=True,
)
os.chdir("Reference-Mismatch-MI-Net")

!pip install -r requirements.txt --quiet
!pip install -e . --quiet

from refshift import setup_kaggle_env
setup_kaggle_env()

RESULTS = "/kaggle/working/results"
os.makedirs(RESULTS, exist_ok=True)
```

`setup_kaggle_env()` sets `MNE_DATA=/kaggle/working/mne_data`,
`MOABB_RESULTS=/kaggle/working/moabb_results`, caps BLAS threads at 1, and
symlinks the Kaggle input datasets into MOABB's cache layout so MOABB does
not re-download.

Known Kaggle paths (override with env vars if yours differ):

| Dataset | Env var | Default Kaggle path |
|---|---|---|
| IV-2a | `REFSHIFT_IV2A_ROOT` | `/kaggle/input/datasets/delhialli/four-class-motor-imagery-bnci-001-2014` |
| OpenBMI | `REFSHIFT_OPENBMI_ROOT` | `/kaggle/input/datasets/imaginer369/openbmi-dataset` |
| Cho2017 | `REFSHIFT_CHO2017_ROOT` | `/kaggle/input/datasets/delhialli/cho2017` |
| Dreyer2023 | `REFSHIFT_DREYER_ROOT` | `/kaggle/input/datasets/delhialli/dreyer2023/MNE-Dreyer2023-data` |

### Cell 2 — calibration

```python
from refshift import calibrate_csp_lda
_, summary, passed = calibrate_csp_lda("iv2a")
assert passed, "calibration FAILED — do not trust downstream numbers"
```

Expected: `CSP+LDA (bare)` ≈ 65.99 ± 15, identity check within 0.5% of bare.
If this doesn't pass, stop and debug before any further runs.

### Cell 3 — the 7×7 mismatch matrix

```python
from refshift import run_mismatch, mismatch_matrix, plot_mismatch_matrix
import matplotlib.pyplot as plt

df = run_mismatch("iv2a", seeds=[0])
df.to_csv(f"{RESULTS}/iv2a_csp_lda.csv", index=False)

print(mismatch_matrix(df).round(3))

fig = plot_mismatch_matrix(
    df, out_path=f"{RESULTS}/iv2a_heatmap.png",
    title="CSP+LDA IV-2a (cross-session, 9 subjects)",
)
plt.close(fig)
```

Repeat for `openbmi`, `cho2017`, `dreyer2023`:

```python
openbmi_subjects = [s for s in range(1, 55) if s != 29]  # exclude corrupted sub 29

for ds, kwargs in [
    ("openbmi",    {"subjects": openbmi_subjects, "seeds": [0]}),
    ("cho2017",    {"seeds": [0, 1, 2]}),
    ("dreyer2023", {"seeds": [0, 1, 2]}),
]:
    df = run_mismatch(ds, **kwargs)
    df.to_csv(f"{RESULTS}/{ds}_csp_lda.csv", index=False)
    fig = plot_mismatch_matrix(df, out_path=f"{RESULTS}/{ds}_heatmap.png",
                                title=f"CSP+LDA {ds}")
    plt.close(fig)
```

`run_mismatch` shows a single tqdm progress bar over (subject, seed) pairs.
No per-subject logging.

First call on a fresh subject triggers MOABB's full pipeline
(load → filter-on-raw → epoch → scale); repeat calls read MOABB's disk
cache under `$MNE_DATA`. Pass `cache=False` to force fresh loads.

### Cell 4 — MOABB-native cross-session sanity check (optional but recommended)

Calibration (Cell 2) uses `WithinSessionEvaluation`. `run_mismatch` uses a
cross-session split. These are different evaluation protocols and will yield
different diagonal numbers. This cell validates the cross-session number
against MOABB's own `CrossSessionEvaluation`:

```python
from moabb.evaluations import CrossSessionEvaluation
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
from refshift.pipelines import make_csp_lda_pipeline

evaluation = CrossSessionEvaluation(
    paradigm=MotorImagery(n_classes=4),
    datasets=[BNCI2014_001()],
    overwrite=True,
    random_state=42,
)
cross_results = evaluation.process({
    "CSP+LDA (bare, cross-session)": make_csp_lda_pipeline(reference_mode=None),
})
cross_mean = 100 * cross_results["score"].mean()
cross_std  = 100 * cross_results["score"].std()
print(f"MOABB CrossSession IV-2a: {cross_mean:.2f} ± {cross_std:.2f}")
```

`run_mismatch`'s diagonal `[native, native]` should match this within ~1
percentage point. If it doesn't, there is a split-logic bug and all
downstream numbers are suspect.

## Design notes

**Why `run_mismatch` does not use MOABB's `Evaluation` class.** MOABB
evaluators train once and score once per fold. The mismatch matrix needs
one training per `train_ref` and seven scorings per fitted model. Wrapping
MOABB's evaluation to extract the fitted classifier mid-fold requires
private-API access. Instead, `run_mismatch` calls `paradigm.get_data()`
directly (which carries all of MOABB's preprocessing: filter-on-raw,
dataset-native epoch windows, correct scaling, dataset-specific quirks)
and loops over train/test reference pairs in ~50 library lines. The
calibration function uses `WithinSessionEvaluation` to validate the
pipeline components against MOABB's published 65.99% number, and the
mismatch runner inherits trust from the shared `make_csp_lda_pipeline`.

**Cross-session vs within-session.** Calibration is within-session
(5-fold CV per session, matches MOABB's 65.99%). `run_mismatch` is
cross-session for multi-session datasets (IV-2a, OpenBMI) and stratified
80/20 for single-session datasets (Cho2017, Dreyer2023). The diagonal of
the mismatch matrix is therefore a few points below the calibration
number — expected, not a bug.

**`ReferenceTransformer('native')` is a true identity but we still
calibrate it.** `calibrate_csp_lda` runs both the bare CSP+LDA and the
`ReferenceTransformer('native') → CSP+LDA` variants; their mean scores
must match within 0.5%. This catches any accidental side-effect of
inserting a step into MOABB's pipeline (dtype change, contiguity,
non-clone-ability). A silent regression here would corrupt every cell of
the mismatch matrix.

**REST (Yao 2001).** Implemented as a linear transformation matrix built
from a three-layer spherical head model fit to the channel montage. The
transform incorporates the centering operator `(I − 1_C 1_C^T / C)` so
the result is invariant to additive re-referencing. Validated by unit
tests: `T @ 1_C = 0` (reference invariance) and `REST(V) ≠ V` (non-trivial).
Built on demand (`build_graph(..., include_rest=True)`); skip if you are
not running the 'rest' mode to avoid a 10–60 s forward-solution compute
per dataset.

**Graph construction uses `standard_1005`.** Every EEG channel in the
four datasets is present. C3's single nearest neighbour on IV-2a under
this montage is CP3 (anatomically correct; unit-tested).

**Standardization.** Phase 1 uses none — CSP's OAS covariance estimator
handles scale. Phase 2 will use braindecode's
`exponential_moving_standardize` on continuous raw before epoching, which
is applied per-channel and therefore does not interact with the
reference operator.

## Calibration reference run (2026-04-22)

```
CSP+LDA (bare)                              65.99 ± 15.92
CSP+LDA (ReferenceTransformer='native')     65.96 ± 15.96
Target 1 (MOABB 65.99% ± 2.0%):  got 65.99% -> PASS
Target 2 (identity within 0.5%):  delta=-0.04% -> PASS
```

IV-2a 6×6 CSP+LDA (9 subjects, seed 0, cross-session, pre-REST):
diagonal mean 0.609, off-diagonal mean 0.380, gap +0.229. Bipolar column
off-diagonals 0.25–0.33 (at chance for 4-class). The 7×7 result with REST
and the three remaining datasets are pending the next run.

## License

MIT — see `LICENSE`.
