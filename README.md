# refshift

Reference-shift experiments for motor-imagery EEG decoding, built on top of
MOABB. The goal is a controlled benchmark showing that re-referencing induces
a systematic, family-structured distribution shift that collapses
cross-reference transfer; and that a simple jitter/SSL intervention fixes it.

This repo is organised in phases. **Phase 1 (this drop)** ships the
foundation: the reference operator as a scikit-learn transformer, a CSP+LDA
pipeline that matches MOABB's canonical recipe, and two scripts — a
calibration anchor against MOABB's published benchmark, and a 6×6 mismatch
matrix runner. Phase 2 (braindecode DL models, jitter, SSL) is scaffolded but
not yet implemented; see the roadmap at the end.

## Why Phase 1 is self-contained

If the CSP+LDA calibration does not reproduce MOABB's 65.99 ± 15.47 on
BCI IV-2a to within ~2%, every downstream number on any model is unreliable.
Phase 1 exists to prove that the pipeline is correctly wired *before* any
deep-learning or SSL work is layered on top. Run the calibration first, then
the mismatch matrix, and verify both before moving on.

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

MOABB is pinned to 1.5.0 because the calibration target (65.99%) was
measured at that version and MotorImagery defaults have drifted in later
releases. Loosen the pin only after re-verifying calibration on the newer
version.

For Phase 2 DL work: `pip install -e '.[dl]'` pulls in `braindecode`,
`torch`, `skorch`. You don't need these for Phase 1.

## Usage

**1. Run the tests.** The 13 pure-numpy tests cover the six operators and
the graph construction. They run in seconds and don't need MOABB:

```bash
pytest tests/test_reference.py -v
```

The integration test (`tests/test_integration.py`) loads IV-2a subject 1
through MOABB and checks that `ReferenceTransformer('native')` is a true
identity in the CSP+LDA pipeline. It's the cheapest end-to-end correctness
check. Auto-skips if MOABB isn't installed.

**2. Calibrate CSP+LDA against MOABB.** Must pass before trusting
anything else:

```bash
python scripts/calibrate_csp_lda.py
```

This runs `WithinSessionEvaluation` on all 9 IV-2a subjects with two
pipelines: bare MOABB canonical, and canonical with
`ReferenceTransformer('native')` prepended. Expected output:

```
Target 1 (MOABB 65.99% ± 2.0%):  got 65.xx%  -> PASS
Target 2 (identity transformer within 0.5% of bare):  delta=+0.00% -> PASS
```

Use `--subjects 1 2 3` for a quick sanity check.

**3. Run the 6×6 mismatch matrix on IV-2a CSP+LDA.**

```bash
python scripts/run_mismatch_iv2a_csp_lda.py --out results/iv2a_csp_lda.csv
```

IV-2a has two sessions per subject, so this uses cross-session evaluation
(session 1 → train, session 2 → test). Per-cell results are saved to the
CSV; the script also prints mean and std 6×6 tables plus a diagonal-vs-
off-diagonal summary.

## Package layout

```
refshift/
├── refshift/
│   ├── __init__.py       public API
│   ├── reference.py      six reference ops + ReferenceTransformer + build_graph
│   ├── data.py           MOABB loading helpers + train/test split
│   ├── pipelines.py      make_csp_lda_pipeline (matches MOABB CSP.yml)
│   └── mismatch.py       run_mismatch_matrix (train once, score six times)
├── scripts/
│   ├── calibrate_csp_lda.py          the trust anchor
│   └── run_mismatch_iv2a_csp_lda.py  the headline CSP+LDA experiment
├── tests/
│   ├── test_reference.py    numpy-only, always runs
│   └── test_integration.py  MOABB-dependent, auto-skips
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Design notes

**Why we don't use MOABB's `Evaluation` for the mismatch matrix.** MOABB's
`WithinSessionEvaluation` and `CrossSessionEvaluation` train once and score
once per fold. The mismatch matrix needs one training per `train_ref` and
six scorings per fitted model; wrapping MOABB's evaluation loop to extract
the fitted classifier mid-fold requires reaching into private methods
(`_fit_cv`, `_build_scored_result`). Instead, `run_mismatch_matrix` calls
`paradigm.get_data()` directly (which carries all of MOABB's preprocessing
— filter-on-raw, correct epoching, dataset-specific quirks) and then does
the 6-train × 6-score loop itself. About 50 lines of code, no private APIs.

The calibration script *does* use `WithinSessionEvaluation` — that's how we
validate the pipeline components against MOABB's published number. Once
calibration passes, the mismatch-matrix runner inherits trust from its
shared pipeline factory.

**Why `ReferenceTransformer('native')` is not a no-op in test wiring.** It
*is* mathematically an identity (up to a fresh copy), and the integration
test confirms this. We include it in the calibration script to catch any
accidental side-effect of inserting a step into the pipeline (e.g., a
dtype change, a contiguity issue, a clone-ability failure). A silent
regression here would corrupt every later row of the mismatch matrix.

**Graph construction uses MNE's `standard_1005` montage.** Every EEG
channel in IV-2a, OpenBMI, Cho2017, and Dreyer2023 is present in this
montage. `build_graph(ch_names, k=4)` returns a frozen `DatasetGraph` with
Laplacian (k-NN) and bipolar (1-NN) indices, built from Euclidean distances
in xyz space. `C3`'s nearest neighbor on IV-2a's 22-channel set is `CP3`
under this montage; this is asserted by
`test_build_graph_iv2a_c3_nearest_is_cp3`.

**No per-trial / instance standardization.** The mismatch-matrix runner
applies reference operators to raw MOABB output (filter-on-raw + scaling
to μV). CSP+LDA handles scale internally via `Covariances(oas)`. Phase 2
DL runs will use `braindecode.preprocessing.exponential_moving_standardize`
applied to the continuous Raw *before* epoching (the canonical
braindecode protocol), so standardization is done once per subject and
does not depend on the reference choice — avoiding the confound where
per-trial standardization computed after the reference operator would
depend on which reference was applied.

## Phase 2 roadmap (not yet implemented)

In order of dependency:

1. **DL pipelines (EEGNet, ShallowFBCSPNet, ATCNet).** Use
   `braindecode.datasets.MOABBDataset` + `braindecode.preprocessing` with
   the canonical four-step preprocess (pick EEG → V→μV → bandpass → EMS).
   Wrap each braindecode model in `skorch.NeuralNetClassifier` for
   sklearn-compatibility, insert `ReferenceTransformer` at the front of
   the pipeline.
2. **Jitter training.** Pre-compute all 6 reference variants of the
   training set once per subject, then a custom per-batch sampler that
   draws a reference uniformly (full jitter) or from a subset
   (leave-one-reference-out).
3. **SSL pretraining.** Within-dataset BarlowTwins or VICReg, positive
   pairs constructed as `(R_i(x), R_j(x))` — two references of the same
   trial as two views of the same latent. Fine-tune at
   label-fractions {1.0, 0.5, 0.25, 0.1}, evaluate on all 6 refs.
4. **Controls** (bandpass mismatch, temporal crop, channel subset).
5. **V2 mechanistic analyses:** family-structure clustering,
   operator-distance ↔ transfer correlation, representation-level CKA.

## Reference

Handoff document (`refshift_handoff.md`) in the project root has the full
scientific context, dataset specifics, v1/v2 retrospective, and numerical
targets per experiment.
