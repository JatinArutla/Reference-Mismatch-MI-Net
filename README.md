# refshift

Reference-shift experiments for motor-imagery (MI) EEG decoding, built on
top of [MOABB](https://github.com/NeuroTechX/moabb) and
[braindecode](https://github.com/braindecode/braindecode).

**Empirical claim.** A classifier trained under one EEG reference (or
spatial-derivative) operator and tested under another suffers a structured,
predictable accuracy collapse. We measure this across five MOABB
motor-imagery datasets, two deep architectures (ShallowFBCSPNet, EEGNet)
plus the classical CSP+LDA pipeline, six reference and spatial operators,
and three interventions: per-sample reference jitter, leave-one-reference-out
training, and an EMS-control ablation that pins down the order in which
exponential moving standardization composes with reference operators.

## Reference and spatial operators

| Family | Operator | What it does |
|---|---|---|
| Global / symmetric | `native` | identity (whatever the dataset was recorded with) |
|  | `car` | X − mean across channels |
|  | `median` | X − median across channels (robustness control) |
|  | `rest` | REST-like spherical-model re-reference (Yao 2001 approximation) |
| Global / asymmetric | `cz_ref` | X − X[Cz] (single-electrode reference; sets Cz channel to zero) |
| Local spatial-derivative | `laplacian` | X − mean of k=4 nearest neighbours (kNN local Laplacian; not formal CSD) |

Nearest-neighbour graphs are computed from MNE's `standard_1005` montage.
REST is built from the same montage via a three-layer spherical head model
(`mne.make_sphere_model` + `mne.make_forward_solution`) with a regularized
pseudo-inverse (rcond=1e-4), 10–60 seconds per dataset, computed once.

`cz_ref` requires `Cz` to be present in the dataset's analysis montage.
On Schirrmeister2017, `Cz` was the recording reference and is excluded
from the published 44-channel motor subset; `cz_ref` is therefore
undefined on Schirrmeister2017, and `apply_reference(..., "cz_ref", ...)`
raises a clear error there. Drop `cz_ref` from `reference_modes` for
Schirrmeister runs. The other four datasets (IV-2a, OpenBMI, Cho2017,
Dreyer2023) all have Cz as a separately-recorded channel.

We deliberately do not include a leave-one-out (LOO) mean reference because
LOO_i = (C/(C−1)) · CAR_i — a constant scaling of CAR that produces
identical results for any scale-invariant decoder (CSP+LDA's eigenvalue
problem; batch-normalised neural networks). We do not include a
projection-based "Gram-Schmidt" operator in the main set because the
natural implementation is data-dependent and doesn't form a fixed C×C
linear operator. Earlier versions also included a "NN-diff" operator
(nearest-neighbour difference, `Y_i = X_i − X_{nn(i)}`); it was removed
in v0.13 because it is not a literature-recognised reference choice — it
was constructed for this codebase as an analogue to clinical bipolar
montages — and its dimension-reducing rank deficiency on dense montages
would confound the per-sample jitter and SSL experiments.

## Package layout

```
Reference-Mismatch-MI-Net/
├── refshift/
│   ├── __init__.py       public API surface
│   ├── reference.py      6 operators, neighbour graph, REST matrix, sklearn transformer
│   ├── pipelines.py      make_csp_lda_pipeline (matches MOABB CSP.yml verbatim)
│   ├── experiments.py    calibrate_csp_lda, run_mismatch, run_mismatch_jitter,
│                         run_lofo_matrix, run_pre_ems_diagonal, run_bandpass_mismatch
│   ├── dl.py             load_dl_data, make_dl_model (braindecode + skorch)
│   ├── jitter.py         RandomReferenceTransform for per-sample reference jitter
│   ├── analysis.py       std matrix, clustering, op-distance correlation
│                         (with bootstrap CI + permutation p), Wilcoxon
│   ├── plotting.py       plot_mismatch_matrix
│   ├── env.py            Kaggle setup + per-dataset cache symlinks
│   └── compat.py         MOABB / braindecode workarounds (one place, documented)
├── tests/                synthetic-only unit tests, none requires MOABB downloads
├── KNOWN_LIMITATIONS.md  methodological caveats and upstream-bug workarounds
├── pyproject.toml
├── requirements.txt
├── LICENSE
└── README.md
```

## Install

```bash
pip install -r requirements.txt
pip install -e ".[dl]"      # `[dl]` adds torch / braindecode / skorch
```

MOABB is pinned to 1.5.0. See `KNOWN_LIMITATIONS.md` for upstream issues
the codebase works around at this version.

## Tests

```bash
pytest tests/ -v
```

None of the tests requires MOABB downloads or network access.

## Kaggle usage

Clone → install → setup → run. The notebook API is the intended entry
point; there is no CLI.

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
!pip install -e ".[dl]" --quiet

from refshift import setup_kaggle_env
setup_kaggle_env()

RESULTS = "/kaggle/working/results"
os.makedirs(RESULTS, exist_ok=True)
```

`setup_kaggle_env()` sets `MNE_DATA=/kaggle/working/mne_data`,
`MOABB_RESULTS=/kaggle/working/moabb_results`, caps BLAS threads at 1, and
symlinks Kaggle input datasets into MOABB's cache layout so MOABB does not
re-download.

Known Kaggle paths (override via env vars if yours differ):

| Dataset | Env var | Default Kaggle path |
|---|---|---|
| IV-2a | `REFSHIFT_IV2A_ROOT` | `/kaggle/input/datasets/delhialli/four-class-motor-imagery-bnci-001-2014` |
| OpenBMI | `REFSHIFT_OPENBMI_ROOT` | `/kaggle/input/datasets/imaginer369/openbmi-dataset` |
| Cho2017 | `REFSHIFT_CHO2017_ROOT` | `/kaggle/input/datasets/delhialli/cho2017` |
| Dreyer2023 | `REFSHIFT_DREYER_ROOT` | `/kaggle/input/datasets/delhialli/dreyer2023/MNE-Dreyer2023-data` |
| Schirrmeister2017 | `REFSHIFT_SCHIRRMEISTER_ROOT` | `/kaggle/input/datasets/hangtrance/high-gamma-dts` |

### Cell 2 — calibration (optional but recommended for first-time setup)

```python
from refshift import calibrate_csp_lda
_, summary, passed = calibrate_csp_lda("iv2a")
assert passed, "calibration FAILED — do not trust downstream numbers"
```

Expected: CSP+LDA (bare) ≈ 65.99 ± 15 on IV-2a, identity check within 0.5%
of bare. If this fails, stop and debug before running the mismatch matrix.

### Cell 3 — Phase 1: CSP+LDA mismatch matrix per dataset

```python
from refshift import run_mismatch, mismatch_matrix, plot_mismatch_matrix
import matplotlib.pyplot as plt

for ds in ["iv2a", "openbmi", "cho2017", "dreyer2023", "schirrmeister2017"]:
    df = run_mismatch(ds, model="csp_lda", seeds=[0])
    df.to_csv(f"{RESULTS}/{ds}_csp_lda.csv", index=False)
    fig = plot_mismatch_matrix(
        df, out_path=f"{RESULTS}/{ds}_csp_lda_heatmap.png",
        title=f"CSP+LDA {ds}",
    )
    plt.close(fig)
    print(mismatch_matrix(df).round(3))
```

`subjects=None` (default) excludes any known-bad subjects automatically; see
`KNOWN_LIMITATIONS.md` for the OpenBMI subject-29 exclusion. Pass
`subjects=[...]` to override.

### Cell 4 — Phase 2: DL mismatch matrix (Shallow / EEGNet)

```python
df = run_mismatch(
    "iv2a",
    model="shallow",                  # or "eegnet"
    seeds=[0, 1, 2],
    dl_max_epochs=200,
    dl_batch_size=32,
    dl_cache_dir="/kaggle/working/cache",   # preprocessed-tensor cache
)
df.to_csv(f"{RESULTS}/iv2a_shallow_seeds012.csv", index=False)
```

The DL pipeline resamples to 250 Hz on every dataset (`dl_resample=250.0`
default) so Shallow's `filter_time_length=25` corresponds to ~100 ms of
physical time regardless of the dataset's native acquisition rate. Pass
a different `dl_resample=` to override; the value is part of the cache
key, so different rates get different cache entries.

EEGNet uses 5e-4 for all datasets (Lawhern et al. 2018 small-data MI
recommendation); no per-dataset learning-rate override is needed.

### Cell 5 — Phase 2 intervention: per-sample reference jitter

```python
from refshift import run_mismatch_jitter, run_lofo_matrix

# Full-jitter: each training sample gets a random reference from all 6 modes
df_full = run_mismatch_jitter(
    "iv2a",
    model="shallow",
    condition="full",
    seeds=[0, 1, 2],
    dl_cache_dir="/kaggle/working/cache",
)
df_full.to_csv(f"{RESULTS}/iv2a_shallow_jitter_full_seeds012.csv", index=False)

# LOFO: hold out one reference at a time; train on the other 5; evaluate
# on all 6. Produces the full LOFO matrix in one call.
df_lofo = run_lofo_matrix(
    "iv2a",
    model="shallow",
    seeds=[0, 1, 2],
    dl_cache_dir="/kaggle/working/cache",
)
df_lofo.to_csv(f"{RESULTS}/iv2a_shallow_lofo_matrix_seeds012.csv", index=False)
```

### Cell 6 — EMS-control ablation (highest-priority methodological control)

```python
from refshift import run_pre_ems_diagonal

# 6-element diagonal with the reference applied BEFORE EMS, on the
# continuous filtered raw. Compare against the diagonal of the standard
# run_mismatch matrix to verify the reference operator's effect on
# accuracy is not driven by EMS-after-reference non-commutativity.
df_pre = run_pre_ems_diagonal(
    "iv2a",
    model="shallow",
    seeds=[0, 1, 2],
    dl_cache_dir="/kaggle/working/cache",
)
df_pre.to_csv(f"{RESULTS}/iv2a_shallow_pre_ems_diag_seeds012.csv", index=False)
```

### Cell 7 — Bandpass-mismatch control (preprocessing-brittleness baseline)

```python
from refshift import run_bandpass_mismatch

# Train at 8-32 Hz; test at 6-32 Hz and 8-30 Hz, holding reference fixed
# at native. Quantifies the per-test-band accuracy drop attributable to
# generic bandpass mismatch, against which the reference-mismatch gap
# can be compared.
df_bp = run_bandpass_mismatch(
    "iv2a",
    model="shallow",
    train_band=(8.0, 32.0),
    test_bands=((6.0, 32.0), (8.0, 30.0)),
    seeds=[0, 1, 2],
    dl_cache_dir="/kaggle/working/cache",
)
df_bp.to_csv(f"{RESULTS}/iv2a_shallow_bandpass_mismatch_seeds012.csv", index=False)
```

### Cell 8 — post-hoc analyses

Three analyses, all pure numpy/scipy, no MOABB, ~1 second each (operator
distance correlation runs ~10 seconds because of the bootstrap +
permutation):

```python
import pandas as pd
import matplotlib.pyplot as plt
from refshift import (
    mismatch_matrix, mismatch_std_matrix,
    cluster_references, plot_dendrogram,
    operator_distance_correlation, plot_operator_distance_scatter,
    paired_wilcoxon_per_test_ref,
    baseline_diagonal_view, baseline_col_off_diag_view,
)

IV2A_CHS = ["Fz","FC3","FC1","FCz","FC2","FC4","C5","C3","C1","Cz","C2","C4","C6",
            "CP3","CP1","CPz","CP2","CP4","P1","Pz","P2","POz"]

baseline = pd.read_csv(f"{RESULTS}/iv2a_shallow_seeds012.csv")
M = mismatch_matrix(baseline)

# Hierarchical clustering of references by transfer pattern
cluster = cluster_references(M)
plot_dendrogram(cluster, out_path=f"{RESULTS}/iv2a_dendrogram.png")

# Spearman ρ between operator-matrix Frobenius distance and transfer gap.
# Reports a 95% bootstrap CI and a permutation p-value alongside the
# asymptotic p, because at 6 operators we have only 15 pairs.
odc = operator_distance_correlation(M, IV2A_CHS)
print(
    f"Spearman ρ = {odc.spearman_rho:.3f} "
    f"(95% CI [{odc.ci95_spearman[0]:.3f}, {odc.ci95_spearman[1]:.3f}], "
    f"perm p = {odc.perm_p_spearman:.4f})"
)

# Significance of jitter intervention vs baseline diagonal
jitter = pd.read_csv(f"{RESULTS}/iv2a_shallow_jitter_full_seeds012.csv")
result = paired_wilcoxon_per_test_ref(
    jitter, baseline_diagonal_view(baseline),
    label_a="full_jitter", label_b="baseline_diag",
    alternative="two-sided",
)
print(result.round(4).to_string(index=False))
```

## Design notes

**Why `run_mismatch` does not use MOABB's `Evaluation` class.** MOABB
evaluators train once and score once per fold. The mismatch matrix needs
one training per `train_ref` and six scorings per fitted model. Wrapping
MOABB's evaluation to extract the fitted classifier mid-fold requires
private-API access. Instead, `run_mismatch` calls `paradigm.get_data()`
directly and loops over train/test reference pairs in ~50 lines. The
calibration function (Cell 2) uses `WithinSessionEvaluation` to validate
against MOABB's published 65.99% number, and the mismatch runner inherits
trust from the shared `make_csp_lda_pipeline`.

**Cross-session vs within-session.** The mismatch runner uses cross-session
splits where the dataset has more than one MOABB session (IV-2a, OpenBMI),
run-based splits where the dataset has a natural train/test run split
within a single session (Schirrmeister2017), and stratified 80/20
within-session splits otherwise (Cho2017, Dreyer2023). The diagonal of the
mismatch matrix is therefore a few points below the calibration number —
expected, not a bug.

**`ReferenceTransformer('native')` is a true identity but we still
calibrate it.** `calibrate_csp_lda` runs both the bare CSP+LDA and the
`ReferenceTransformer('native') → CSP+LDA` variants; their mean scores
must match within 0.5%. This catches any accidental side-effect of
inserting a step into MOABB's pipeline (dtype change, contiguity,
non-clone-ability). A silent regression here would corrupt every cell of
the mismatch matrix.

**REST (Yao 2001).** Implemented as a linear transformation matrix built
from a three-layer spherical head model fit to the channel montage. The
transform incorporates the centering operator `(I − 1_C 1_C^T / C)` so the
result is invariant to additive re-referencing. Validated by unit tests:
`T @ 1_C = 0` (reference invariance) and `REST(V) ≠ V` (non-trivial).

**Standardization and reference order.** Phase 1 (CSP+LDA) uses none —
CSP's OAS covariance estimator handles scale. Phase 2 (DL) uses
braindecode's `exponential_moving_standardize` on continuous raw before
epoching; references are then applied to the windowed, standardized X
array. EMS is per-channel and adaptive, so it does **not** commute with
channel-mixing reference operators. The standard pipeline therefore
measures "reference applied to EMS-standardized signals." The
`run_pre_ems_diagonal` ablation flips this order — applying the
reference to the continuous filtered raw before EMS — and reports a
diagonal that should match the standard pipeline's diagonal within seed
noise if the EMS-after-reference order is not driving the cluster
structure.

**DL resample standardization.** The DL pipeline resamples every dataset
to a common rate (`dl_resample=250.0` default) before bandpass + EMS, so
the time-domain receptive field of every model is identical regardless
of the dataset's native acquisition rate. Without this, Shallow's
`filter_time_length=25` (samples) gave ~100 ms physical-time at IV-2a's
250 Hz, ~50 ms at Cho2017/Dreyer2023's 512 Hz, and ~25 ms at OpenBMI's
1000 Hz, complicating cross-dataset comparisons. The CSP+LDA path is left
at MOABB's native paradigm settings (Schirrmeister2017 uses 250 Hz via
`paradigm.resample`; the others run at native rate); CSP-based decoders
are not sensitive to absolute sample rate, so the inconsistency does not
affect their comparability.

**EEGNet learning rate.** `make_dl_model("eegnet", ...)` uses 5e-4 by
default — Lawhern et al. 2018's recommendation for small-data MI — for
all datasets uniformly. Earlier code branches set 1e-3 for some datasets
and 5e-4 for Cho2017 specifically; uniform 5e-4 removes the per-dataset
override.

**Per-sample reference jitter.** Implemented as a braindecode `Transform`
plugged into `AugmentedDataLoader`. Each training sample independently
gets a reference drawn uniformly from `allowed_modes`. Full-jitter uses
all available operators on the dataset (typically all 6; on
Schirrmeister2017, drop `cz_ref` for 5); LOFO uses one fewer (the
held-out operator excluded from the training distribution). See
`refshift/jitter.py`.

**Operator-distance correlation.** `operator_distance_correlation`
estimates each reference operator's linear C×C matrix on a Gaussian probe
(averaged over multiple probes for the median operator's linear-tangent
estimate), computes pairwise Frobenius distances, and correlates them
with the symmetric transfer gap `gap_ij = diag_mean - 0.5*(M_ij + M_ji)`.
Because the upper triangle has only 15 pairs at 6 operators, the
asymptotic Spearman/Pearson p-values are unreliable; the function
additionally returns a bootstrap 95% confidence interval and a
permutation p-value computed by shuffling operator labels of the gap
matrix. We do not interpret this as a Ben-David H-divergence bound;
Frobenius distance is a data-free quantity, and its empirical correlation
with transfer gap is a structural finding, not a tight theoretical bound.

**Compatibility shims.** All MOABB / braindecode workarounds live in
`refshift/compat.py` so the rest of the codebase stays library-faithful.
Each shim documents the upstream issue, the version the workaround was
needed at, and what would let it be removed. See also
`KNOWN_LIMITATIONS.md`.

## License

MIT — see `LICENSE`.
