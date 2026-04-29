# refshift

Reference-shift experiments for motor-imagery (MI) EEG decoding, built on
top of [MOABB](https://github.com/NeuroTechX/moabb) and
[braindecode](https://github.com/braindecode/braindecode).

**Empirical claim.** A classifier trained under one EEG reference operator
(CAR, Laplacian, REST, ...) and tested under another suffers a structured,
predictable accuracy collapse. We measure this across four MOABB
motor-imagery datasets, three decoder families (CSP+LDA,
ShallowFBCSPNet, EEGNet), seven reference operators, and one mitigation
(per-sample reference jitter).

## Reference operators

| Family | Operator | What it does |
|---|---|---|
| Global-mean | `native` | identity (whatever the dataset was recorded with) |
|  | `car` | X − mean across channels |
|  | `median` | X − median across channels |
|  | `gs` | X − leave-one-out mean projection (Gram-Schmidt) |
| Spatial-differential | `laplacian` | X − mean of k=4 nearest neighbours |
|  | `bipolar` | X − single nearest neighbour |
| Source-model | `rest` | Yao 2001 Reference Electrode Standardization Technique |

Nearest-neighbour graphs are computed from MNE's `standard_1005` montage.
REST is built from the same montage via a three-layer spherical head model
(`mne.make_sphere_model` + `mne.make_forward_solution`), 10–60 seconds per
dataset, computed once.

## Package layout

```
Reference-Mismatch-MI-Net/
├── refshift/
│   ├── __init__.py       public API surface
│   ├── reference.py      7 operators, neighbour graph, REST matrix, sklearn transformer
│   ├── pipelines.py      make_csp_lda_pipeline (matches MOABB CSP.yml verbatim)
│   ├── experiments.py    calibrate_csp_lda, run_mismatch, run_mismatch_jitter
│   ├── dl.py             load_dl_data, make_dl_model (braindecode + skorch)
│   ├── jitter.py         RandomReferenceTransform for per-sample reference jitter
│   ├── analysis.py       std matrix, clustering, op-distance correlation, Wilcoxon
│   ├── plotting.py       plot_mismatch_matrix
│   ├── env.py            Kaggle setup + per-dataset cache symlinks
│   └── compat.py         MOABB / braindecode workarounds (one place, documented)
├── tests/                87 unit tests, none requires MOABB or network
├── KNOWN_LIMITATIONS.md  methodological caveats and upstream-bug workarounds
├── pyproject.toml
├── requirements.txt
├── LICENSE
└── README.md
```

## Install

```bash
pip install -r requirements.txt
pip install -e .[dl]   # `[dl]` adds torch / braindecode / skorch
```

MOABB is pinned to 1.5.0. See `KNOWN_LIMITATIONS.md` for upstream issues
the codebase works around at this version.

## Tests

```bash
pytest tests/ -v
```

Expect **87/87 passing**. None requires MOABB downloads or network access.

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
!pip install -e .[dl] --quiet

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

for ds in ["iv2a", "openbmi", "cho2017", "dreyer2023"]:
    df = run_mismatch(ds, model="csp_lda", seeds=[0])
    df.to_csv(f"{RESULTS}/{ds}_csp_lda.csv", index=False)
    fig = plot_mismatch_matrix(
        df, out_path=f"{RESULTS}/{ds}_heatmap.png",
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

`dl_cache_dir=` saves the preprocessed (X, y, metadata) tuple to disk per
subject, keyed on a hash of all preprocessing parameters. Subsequent runs
on the same dataset (different architecture, different seed, jitter
condition) skip re-preprocessing.

EEGNet on Cho2017 specifically is sensitive to learning rate; pass
`dl_lr=5e-4` (Lawhern et al. 2018) to recover from the chance-level result
under `dl_lr=1e-3` defaults. See `KNOWN_LIMITATIONS.md`.

### Cell 5 — Phase 2 intervention: per-sample reference jitter

```python
from refshift import run_mismatch_jitter

# Full-jitter: each training sample gets a random reference from all 7 modes
df = run_mismatch_jitter(
    "iv2a",
    model="shallow",
    condition="full",
    seeds=[0, 1, 2],
    dl_cache_dir="/kaggle/working/cache",
)
df.to_csv(f"{RESULTS}/iv2a_shallow_jitter_full_seeds012.csv", index=False)

# Leave-one-reference-out: bipolar held out; train on the other 6
df = run_mismatch_jitter(
    "iv2a",
    model="shallow",
    condition="lofo",
    holdout_ref="bipolar",
    seeds=[0, 1, 2],
    dl_cache_dir="/kaggle/working/cache",
)
df.to_csv(f"{RESULTS}/iv2a_shallow_jitter_lofo_bipolar_seeds012.csv", index=False)
```

### Cell 6 — post-hoc analyses

Three analyses, all pure numpy/scipy, no MOABB, ~1 second each:

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

# Spearman ρ between operator-matrix Frobenius distance and transfer gap
odc = operator_distance_correlation(M, IV2A_CHS)
print(f"Spearman ρ = {odc.spearman_rho:.3f} (p={odc.spearman_p:.1e})")

# Significance of jitter intervention vs baseline
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
one training per `train_ref` and seven scorings per fitted model. Wrapping
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

**Standardization.** Phase 1 (CSP+LDA) uses none — CSP's OAS covariance
estimator handles scale. Phase 2 (DL) uses braindecode's
`exponential_moving_standardize` on continuous raw before epoching, which
is applied per-channel and therefore does not interact with the reference
operator.

**Per-sample reference jitter.** Implemented as a braindecode `Transform`
plugged into `AugmentedDataLoader`. Each training sample independently
gets a reference drawn uniformly from `allowed_modes`. The full-jitter
condition uses all 7; LOFO uses 6 (one held out). See `refshift/jitter.py`.

**Compatibility shims.** All MOABB / braindecode workarounds live in
`refshift/compat.py` so the rest of the codebase stays library-faithful.
Each shim documents the upstream issue, the version the workaround was
needed at, and what would let it be removed. See also
`KNOWN_LIMITATIONS.md`.

## License

MIT — see `LICENSE`.
