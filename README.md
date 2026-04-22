# refshift

Reference-shift experiments for motor-imagery EEG decoding, built on top of
MOABB. The goal is a controlled benchmark showing that re-referencing
induces a systematic distribution shift that collapses cross-reference
transfer, and that a jitter / SSL intervention fixes it.

Phase 1 (this drop) ships the foundation: the reference operator as a
scikit-learn transformer, a CSP+LDA pipeline that matches MOABB's canonical
recipe, MOABB-backed dataset loading with a Kaggle cache layer, and
notebook-style entry points. Phase 2 (braindecode DL models, jitter, SSL)
is scaffolded in the roadmap.

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

MOABB is pinned to 1.5.0. Loosen the pin only after re-verifying the
65.99 ± 15.47 calibration target on a newer version.

For Phase 2 DL work: `pip install -e '.[dl]'` pulls in braindecode, torch,
skorch. You don't need these for Phase 1.

## Kaggle notebook: typical usage

Three cells. No `!python ...` needed after the setup cell.

### Cell 1 — clone, install, setup

```python
import os, pathlib, subprocess
os.chdir("/kaggle/working")
REPO_URL = "https://github.com/JatinArutla/Reference-Mismatch-MI-Net"
REPO_ROOT = pathlib.Path("Reference-Mismatch-MI-Net")
if not REPO_ROOT.exists():
    subprocess.run(["git", "clone", REPO_URL, str(REPO_ROOT)], check=True)
os.chdir(REPO_ROOT)

!pip install -r requirements.txt --quiet
!pip install -e . --quiet

from refshift import setup_kaggle_env
setup_kaggle_env()
```

`setup_kaggle_env()` sets `MNE_DATA=/kaggle/working/mne_data`,
`MOABB_RESULTS=/kaggle/working/moabb_results`, caps BLAS threads at 1, and
symlinks the Kaggle input datasets into MOABB's cache layout so MOABB
doesn't re-download. Known paths:

| Dataset | Kaggle path |
|---|---|
| IV-2a | `/kaggle/input/datasets/delhialli/four-class-motor-imagery-bnci-001-2014` |
| OpenBMI | `/kaggle/input/datasets/imaginer369/openbmi-dataset` |
| Cho2017 | `/kaggle/input/datasets/delhialli/cho2017` |
| Dreyer2023 | `/kaggle/input/datasets/delhialli/dreyer2023/MNE-Dreyer2023-data` |

Override with env vars `REFSHIFT_IV2A_ROOT`, `REFSHIFT_OPENBMI_ROOT`,
`REFSHIFT_CHO2017_ROOT`, `REFSHIFT_DREYER_ROOT` if your paths differ.

IV-2a and Cho2017 symlinks are known to work. OpenBMI and Dreyer2023 are
best-effort (the exact cache subdirectory layout may differ by MOABB
version); if the symlinks don't match what MOABB expects, it falls back
to downloading — slow but correct.

### Cell 2 — calibrate against MOABB's published benchmark

```python
from refshift import calibrate_csp_lda
results, summary, passed = calibrate_csp_lda(dataset_id="iv2a")
print(summary)
assert passed, "Calibration failed — do not trust downstream numbers."
```

Expected output: `CSP+LDA (bare)` ≈ 65.99 ± 15, `CSP+LDA
(ReferenceTransformer='native')` within 0.5% of bare. Both pass bars
print at the end of the call.

If this doesn't pass, stop here. Every downstream number on any model
will be unreliable until the pipeline is correctly wired against MOABB.

### Cell 3 — run the 6x6 mismatch matrix

```python
from refshift import run_mismatch, mismatch_matrix
import os

RESULTS = "/kaggle/working/results"
os.makedirs(RESULTS, exist_ok=True)

df = run_mismatch(
    dataset_id="iv2a",
    subjects=None,      # None = all 9
    seeds=[0, 1, 2],
    model="csp_lda",    # Phase 1 only supports csp_lda; DL is Phase 2
)
df.to_csv(f"{RESULTS}/iv2a_csp_lda.csv", index=False)

print(mismatch_matrix(df, metric="accuracy", aggregate="mean").round(3))
print(mismatch_matrix(df, metric="accuracy", aggregate="std").round(3))
```

CSP+LDA is nearly deterministic for session-split datasets (IV-2a,
OpenBMI); multi-seed only varies the shuffle for Cho2017 and Dreyer2023's
stratified 80/20 split. On IV-2a `seeds=[0]` is sufficient.

Repeat the cell for other datasets by changing `dataset_id`:

```python
for dataset_id in ("iv2a", "openbmi", "cho2017", "dreyer2023"):
    df = run_mismatch(dataset_id=dataset_id, seeds=[0])
    df.to_csv(f"{RESULTS}/{dataset_id}_csp_lda.csv", index=False)
    print(f"\n=== {dataset_id} ===")
    print(mismatch_matrix(df).round(3))
```

The first call on a fresh subject triggers MOABB's full pipeline
(download → filter-on-raw → epoch → resample); repeat calls read from
MOABB's disk cache (under `$MNE_DATA`). Speedup is roughly 10–20× on
rerun. Disable by passing `cache=False` if you need fresh loads.

## CLI alternative

Scripts are thin wrappers around the library functions. Useful for
reproducibility or cluster jobs.

```bash
python scripts/calibrate.py --dataset iv2a
python scripts/run_mismatch.py --dataset iv2a --seeds 0 --out results/iv2a.csv
```

## Package layout

```
refshift/
├── refshift/
│   ├── __init__.py       public API (notebook-style + primitives)
│   ├── env.py            setup_kaggle_env, setup_moabb_symlinks
│   ├── registry.py       dataset_id -> (dataset, paradigm)
│   ├── reference.py      six reference ops + ReferenceTransformer + build_graph
│   ├── data.py           MOABB loading + cache_config helper + split
│   ├── pipelines.py      make_csp_lda_pipeline (matches MOABB CSP.yml)
│   ├── calibration.py    calibrate_csp_lda (library form)
│   └── mismatch.py       run_mismatch(id)  +  run_mismatch_matrix(paradigm, dataset)
├── scripts/
│   ├── calibrate.py      thin CLI around calibrate_csp_lda
│   └── run_mismatch.py   thin CLI around run_mismatch
├── tests/
│   ├── test_reference.py    13 pure-numpy tests, always run
│   └── test_integration.py  MOABB-dependent, auto-skips
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Design notes

**Why we don't use MOABB's `Evaluation` for the mismatch matrix.** MOABB's
evaluators train once and score once per fold; the mismatch matrix needs
one training per `train_ref` and six scorings per fitted model. Wrapping
MOABB's evaluation loop to extract the fitted classifier mid-fold requires
private-API access (`_fit_cv`, `_build_scored_result`). Instead,
`run_mismatch_matrix` calls `paradigm.get_data()` directly — which carries
all of MOABB's preprocessing (filter-on-raw, correct epoching,
dataset-specific quirks) — and does the 6×6 loop in ~50 lines of
library code. The calibration script *does* use
`WithinSessionEvaluation` to validate pipeline components against MOABB's
published number; once calibration passes, the mismatch runner inherits
trust from its shared pipeline factory.

**Why `ReferenceTransformer('native')` is a real identity but still
tested.** It is mathematically identity (up to a fresh copy), confirmed
by the integration test. We nonetheless include it as a second pipeline
in `calibrate_csp_lda` to catch any accidental side-effect of inserting a
step into MOABB's pipeline (dtype change, contiguity issue,
non-clone-ability). A silent regression here would corrupt every later
row of the mismatch matrix.

**Graph construction uses MNE's `standard_1005` montage.** Every EEG
channel in the four datasets is present. `build_graph(ch_names, k=4)`
builds nearest-neighbor indices from Euclidean xyz distances. C3's
nearest neighbor on IV-2a under this montage is CP3, asserted by
`test_build_graph_iv2a_c3_nearest_is_cp3`.

**Standardization: Phase 1 uses none** (CSP+LDA handles scale internally
via `Covariances(oas)`). Phase 2 will use
`braindecode.preprocessing.exponential_moving_standardize` applied to
continuous Raw before epoching — identical across reference conditions
and therefore not a confound.

## Calibration results (your run, 2026-04-22)

```
CSP+LDA (bare)                              65.99 ± 15.92
CSP+LDA (ReferenceTransformer='native')     65.96 ± 15.96
Target 1 (MOABB 65.99% ± 2.0%):  got 65.99% -> PASS
Target 2 (identity within 0.5%):  delta=-0.04% -> PASS
```

IV-2a 6×6 CSP+LDA (9 subjects, seed 0, cross-session):
diagonal mean 0.609, off-diagonal mean 0.380, gap +0.229. Bipolar column
off-diagonals 0.25–0.33 (at chance for 4-class). Structural claim of the
paper holds on the classical baseline.

## Phase 2 roadmap

In dependency order:

1. **DL pipelines (EEGNet, ShallowFBCSPNet, ATCNet).** braindecode +
   skorch, EMS standardization on Raw before epoching.
2. **Jitter training.** Pre-compute all 6 reference variants per subject;
   per-batch sampler for full-jitter and leave-one-reference-out.
3. **SSL pretraining.** BarlowTwins or VICReg with reference-pair
   positives `(R_i(x), R_j(x))`; fine-tune at label fractions
   {1.0, 0.5, 0.25, 0.1}; evaluate per reference.
4. **Controls** (bandpass mismatch, temporal crop, channel subset).
5. **V2 mechanistic analyses**: family-structure clustering,
   operator-distance ↔ transfer correlation, representation-level CKA.

Triggered only after the CSP+LDA mismatch matrix is reproduced on all
four datasets.

## Handoff

`refshift_handoff.md` (not in this repo — in the project's working
documents) has full scientific context, dataset specifics, v1/v2
retrospective, and numerical targets per experiment.
