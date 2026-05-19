# NOTES.md — design-decision log

A running log of non-obvious choices in this codebase, in author's voice.
Each entry records: **what** I chose, **what I rejected**, **why**.

This file exists because in interviews ("why did you …?") I want to be able
to point to a written record from when I made the call, not reconstruct
the reasoning under pressure.

Append, do not edit. New entries on top.

---

## v0.16.1 — trace-normalised CSP+LDA ablation for CSD-scale confound

External reviewer feedback flagged that the CSD-at-chance cross-reference
finding in v0.16's mismatch matrix is methodologically vulnerable: a
skeptical reviewer can argue the failure is driven by CSD's ~10^3
amplitude scale (the OAS-covariance argument is real for the covariance
itself but doesn't fully cover the end-to-end pipeline — CSP filters
fit on data with one variance structure produce projected features
with different statistics than the same filters applied to data with
10^6 variance structure, and the LDA boundary trained on the former
will not work on the latter).

Three suggested ablations:
1. Trace-normalise covariance per trial: Sigma_i / tr(Sigma_i).
2. Per-trial RMS normalisation of raw (C, T) array after spatial operator.
3. Operator-normalised distance for the cross-correlation plot.

#3 is already in v0.16 (`distance_metric="frobenius_normed"` default).
#1 and #2 are mathematically equivalent for CSP+LDA (both rescale every
trial to unit total power before downstream estimation), so I implemented
#1 only -- it sits inside the canonical MOABB CSP pipeline as one
inserted step rather than at the raw-signal layer, and is exactly what
the reviewer literally specified.

Added `TraceNormalizer` (sklearn-compatible, stateless transformer)
in refshift.model. Inserts between Covariances(oas) and CSP when
`make_csp_lda_pipeline(trace_normalize=True)`. Threaded through
`run_mismatch` via `csp_trace_normalize: bool = False`. Default
False preserves MOABB CSP.yml canonical behaviour and v0.14/v0.15
reproducibility.

If the CSD cross-reference cells remain at chance under
`csp_trace_normalize=True`, the result graduates from "consistent with
amplitude-scale confound" to "operator-topology effect that survives
the obvious scale control". That changes the paper's defensibility
substantially.

### What I rejected

- **Always-on trace normalisation.** Would break v0.14/v0.15
  reproducibility; trace-normalisation is a methodological choice,
  not an obviously correct default.
- **Per-trial RMS at the raw-signal level (Ablation #2).** Equivalent
  to #1 for the CSP+LDA pipeline. Different code path, same outcome.
  Could add later for DL pipelines where the scale-control story
  differs.
- **Tukey/Z-score normalisation across the dataset.** Population-level
  rescaling violates train/test isolation. Per-trial only.

---



This release responds to an external review round that identified eight
issues. The work is in three buckets: framing/documentation that was
defensively wrong (CSD scale, lap_large naming), API gaps that should
have been caught earlier (`k_large_*` not exposed in runners, raw
operator distance dominated by CSD scale, MNE log-level not restored
when previous was `None`), and one genuinely new piece of science
machinery (`run_pre_ems_mismatch`).

### Framing fix: CSD amplitude scale is a confound, not a feature

v0.15's `KNOWN_LIMITATIONS.md` framed CSD's ~10^3 amplitude scale as
"the intended finding": *CSD is genuinely a different operator and its
cross-reference transfer should be visibly bad; reporting that is the
point.* A reviewer flagged this as defensive rationalisation, and I
agree on reflection. The honest position:

- **What we measure** in a `run_mismatch` CSD cell with the standard
  pipeline (reference after EMS): the *joint* effect of CSD's
  spatial-derivative topology AND the unit/scale mismatch between
  CSD-output and CAR-output distributions.
- **What we want to claim**: that operator topology drives transfer
  failure. The scale argument is uninteresting and uninformative
  about EEG.

These are not the same thing. A reviewer can fairly say "your CSD
mismatch is just unit mismatch" and would be right. The v0.16 response:

1. Default `operator_distance_correlation(distance_metric=...)` is now
   `"frobenius_normed"` instead of `"frobenius_raw"`. Each operator is
   rescaled to unit Frobenius norm before pairwise differencing, so
   distances reflect spatial-topology direction only. Raw remains
   available for appendix runs and cross-version comparison.
2. New `run_pre_ems_mismatch` runner: applies the reference BEFORE
   per-channel EMS rather than after. If the family structure of the
   mismatch matrix is similar under both pipelines, the v0.14/v0.15
   results were not preprocessing-order artefacts. If they differ for
   CSD specifically, the pre-EMS pipeline becomes the primary report.

I considered dropping CSD from the operator set to sidestep the
problem. Rejected: CSD is a textbook EEG reference, and excluding it
because it's awkward methodologically is worse than including it
correctly. The paper now reports both raw and scale-normed analyses,
and both standard and pre-EMS mismatch matrices.

### lap_large naming: deterministic next-ring approximation, not the McFarland Laplacian

v0.15 described lap_large as "the McFarland 1997 next-ring large
Laplacian". On dense montages this is approximately correct; on
IV-2a's sparse 22-channel layout it overclaims. For Fz, ranks 4..7 in
Euclidean order are FC4, Cz, C2, C1 — frontal + midline + central,
not a frontal ring. The operator math is fine and the disjoint-from-
lap_small property holds, but calling it "the McFarland large
Laplacian" invites a fair rebuttal from anyone who actually checks
the neighbour sets on IV-2a.

Renamed throughout to "deterministic next-ring large-Laplacian
approximation, motivated by McFarland 1997 but not equivalent on
sparse montages". Implementation unchanged; only docstrings, README,
and KNOWN_LIMITATIONS revised.

### API additions and ergonomic fixes

**`k_large_skip` / `k_large_use` exposed in runners.** In v0.15 these
were accepted by `build_graph` but the only path from `run_mismatch`
to `build_graph` went through `laplacian_k` (the legacy `k=` alias
for `k_small`). Now every runner (`run_mismatch`, `run_mismatch_jitter`,
`run_lofo_matrix`, `run_pre_ems_diagonal`, `run_bandpass_mismatch`,
new `run_pre_ems_mismatch`) accepts both `k_large_skip` and
`k_large_use` and threads them through. `load_dl_data` adds
`pre_ems_k_large_skip` and `pre_ems_k_large_use` to its signature and
to `_CACHE_KEY_PARAMS`, so distinct settings cache separately.

**`operator_distance_correlation` defaults lowered to 1000/1000.**
v0.15 defaults of `n_permutations=10_000`, `n_bootstrap=5_000` made
interactive calls hang. Lowered to 1000 each; for paper-final CIs
users can raise back up.

**MNE log-level restoration uses `mne.use_log_level("ERROR")` context
manager.** v0.15's save-restore pattern did nothing useful when the
previous level was `None`; the context manager handles all cases.

### New runner: `run_pre_ems_mismatch`

DL-only (CSP+LDA doesn't use EMS, so the question doesn't apply). For
each (subject, seed) and each reference `r` in the mode set, loads
`load_dl_data(pre_ems_reference=r)` once (cache hit on repeats), then
runs the full NxN inner loop: train on `X_tr_train_ref`, predict on
`X_te_test_ref` for every `test_ref`. Trial counts and labels are
asserted equal across references (the split is metadata-driven, not
data-driven, so this should always hold; the assertion guards against
silent corruption).

The output DataFrame has a `pipeline="pre_ems_mismatch"` column to
allow direct `pd.concat` with `run_mismatch` output for side-by-side
comparison plotting.

Compute scaling: N refs × S subjects loads (cached after first call,
shared across seeds), N × S × K trainings. Comparable cost to
`run_mismatch` (which does N × S × K trainings but only S × K loads).

I considered redesigning the standard DL preprocessing pipeline to do
reference-then-standardisation by default. Rejected: would break
v0.14/v0.15 reproducibility for no scientific gain. Add the
alternative pipeline as a parallel analysis instead, and report both.

### Synthetic test fixture: separated mechanics from interpretation

v0.15 `tests/test_analysis.py`'s `synthetic_df` fixture set
`base = 0.18` for any CSD pair, with a comment explaining this as
"CSD's amplitude-scale outlier behaviour". A reviewer correctly
pointed out that this bakes in a controversial empirical interpretation
into a test fixture.

The v0.16 fixture keeps the same numerical structure (clustering tests
need a known ground truth) but reframes the docstring: this is an
arbitrary structural choice for testing clustering mechanics, not an
empirical claim. Added a parallel `synthetic_neutral_df` fixture where
CSD has the same transfer profile as any other spatial-derivative
operator, for tests that should not depend on CSD being a structural
outlier (e.g. the new operator-distance metric tests).

### Documentation polish

- "classes is currently CSP+LDA only" claim in `run_mismatch` docstring
  was stale since v0.14.2 (DL classes plumbing was added then). Removed.
- "6x6" framing in `experiments/mismatch.py` and `experiments/__init__.py`
  module docstrings replaced with "NxN" (N depends on the resolved
  reference set: 8 by default, 7 for Schirrmeister).

### Verification

90 → 95 unit tests after v0.16 additions (1 of the 5 new tests is the
mocked `run_pre_ems_mismatch` test which skips outside braindecode-
enabled environments; locally on Kaggle it passes). 0 failures.

### What I rejected

- **Dropping CSD from the operator set.** Methodologically cleaner but
  scientifically uninformative; CSD is in the EEG decoding literature
  and our matrix tells readers what they can expect.
- **Replacing the standard DL pipeline with reference-before-EMS.**
  Breaks v0.14/v0.15 cache compatibility and downstream notebooks.
  Adding a parallel runner is the right additive change.
- **Auto-resolving the `distance_metric` default by inspecting the
  operator set.** Considered: "if CSD is in the modes, switch to
  normed; otherwise raw". Rejected as too magic — explicit beats
  implicit, and a user who wants raw-Frobenius with CSD (e.g. to
  cross-compare with v0.15) shouldn't have to override an
  auto-detection.

---



### Framing locked: "reference mismatch", not "spatial operator shift"

The TMLR/NeurIPS-workshop paper writes about **reference mismatch**:
trained under one reference, tested under another. Earlier drafts
flirted with "spatial operator shift" because three of the eight
operators (lap_small, lap_large, csd) are spatial-derivative not
reference-replacement. Dropped. "Reference" is the term every EEG
practitioner uses for any operator that recombines channel values, and
the paper is about cross-operator transfer failures, full stop. The
broader term made readers ask "why is this not framed as covariate
shift" — a 200-page rabbit hole I don't need to enter for a workshop
paper.

### Operator set: 6 → 8

Old set: `native, car, median, laplacian, rest, cz_ref` (six modes).

New set: `native, car, median, rest, cz_ref, lap_small, lap_large, csd`
(eight modes).

The `laplacian` operator is renamed to `lap_small` (the math is
unchanged; the rename clarifies that `lap_small` is one of three
spatial-derivative variants). The legacy name is kept as an alias —
old CSVs and notebooks work without edits.

Two new spatial-derivative operators:

- **`lap_large`** (McFarland 1997 next-ring large Laplacian): for each
  channel, subtract the mean of the ring of neighbours at distance ranks
  4..7 (skipping the closest 4, which `lap_small` uses). With these
  defaults the two Laplacians have disjoint neighbour sets for every
  channel by construction, giving a clean fine-vs-coarse scale separation.
  I picked the next-ring variant over the k=8-NN variant because the
  scale separation against lap_small is the point — k=8-NN would just
  be a noisier lap_small.
- **`csd`** (Perrin spherical-spline surface Laplacian): the formal CSD
  from `mne.preprocessing.compute_current_source_density`. We recover the
  fixed C×C operator by pushing the identity basis through MNE and reading
  the output, then cache the matrix. Verified empirically: matrix
  application is identical to per-epoch MNE application to within 1e-16
  in float64 (machine precision). MNE defaults: `lambda2=1e-5`,
  `stiffness=4`, `n_legendre_terms=50`, `sphere="auto"`.

### cz_ref kept, despite pushback

A reviewer suggested dropping `cz_ref` because zeroing the Cz row is
capacity waste for any decoder that uses Cz. Rejected. The capacity
loss IS the headline. `cz_ref` is what some labs publish under — single-
electrode clinical reference is standard practice — and the mismatch
matrix's job is to show how badly transfer fails between operators that
all see different views of the same signal. Hiding `cz_ref` would
suppress the cleanest single-electrode failure mode the matrix has.

Sanity-check fallback (not in the headline matrix, optional appendix):
re-run with a 21-channel IV-2a subset that drops Cz from the input
entirely, so all operators see the same channel count. Defer until the
paper is otherwise locked.

### lap_large parameter choice (k_skip=4, k_use=4)

Considered:
- k_skip=4, k_use=4 (disjoint from lap_small by construction).  CHOSEN.
- k_skip=0, k_use=8 (the "k=8-NN large Laplacian" some papers use).
  Rejected: not disjoint from lap_small at k_small=4; the two operators
  share 4 neighbours per channel. The whole point of lap_large is to be
  a different spatial-frequency operator from lap_small, not a noisier
  version of it.
- k_skip=8, k_use=4 (skip further). Rejected: on IV-2a's 22-channel set
  the next ring beyond rank 8 already includes far-side channels;
  anatomically the operator stops being interpretable as a "ring" once
  you're picking POz's nearest neighbour out of a parietal-frontal mix.

The McFarland 1997 paper uses a skip-NN scheme on a 64-channel dense
montage; our k_skip=4, k_use=4 is the closest analogue at IV-2a's 22
channels.

### `references` notebook parameter (`reference_modes=` accepts iterables)

Every runner already accepted `reference_modes=` as a tuple. v0.15 makes
it accept any iterable (set, tuple, list, frozenset) and adds the helper
`canonical_mode_tuple(refs)` that resolves aliases and reorders to
`REFERENCE_MODES` order. Usage:

```python
REFERENCES = {"native", "car", "csd"}
df = run_mismatch("iv2a", model="csp_lda", reference_modes=REFERENCES, seeds=[0, 1, 2])
```

Output column order is deterministic regardless of set iteration order.
This was a small change but it removes the most common notebook footgun:
"why is my matrix in a different order this run."

### Operator-distance correlation now uses exact matrices

The seven linear operators (native, CAR, REST, CSD, cz_ref, lap_small,
lap_large) have closed-form C×C matrices. `_exact_operator_matrix`
returns them directly. Only `median` falls back to the probe-based
linear-tangent estimate. This removes the small variance source from
Gaussian-probe linear regression on operators where it was unnecessary.

### What did NOT change in v0.15

- The five datasets (IV-2a, OpenBMI, Cho2017, Dreyer2023, Schirrmeister2017).
- The two DL architectures (ShallowFBCSPNet, EEGNet) + CSP+LDA.
- The three interventions (per-sample jitter, LOFO, EMS-control).
- REST implementation. The algebraic invariants (`T @ 1_C = 0`, rank
  `C-1`) still hold. Cross-validation against the Dong et al. MATLAB
  toolbox remains unfinished and is a documented caveat.
- IV-2a CSP+LDA calibration target (65.99% ± 2%).

---

## v0.14.2 — 2-class DL support, package-level report helper

### Threaded `classes` through all DL runners

Previous code rejected `classes=(...)` on the DL path with a
`NotImplementedError`. The block was: "threading the class subset through
`load_dl_data` is a separate change." This release does that change.

Implementation:
1. `load_dl_data` accepts `classes=None | tuple`. Pre-windowing, build an
   explicit `mapping={class_name: i for i, name in enumerate(full_classes)}`
   and pass it to braindecode's `create_windows_from_events`. This makes the
   integer<->class mapping deterministic across subjects (fixed a latent
   bug: previously braindecode auto-built the mapping per subject, so subject
   3 with no "tongue" trials could have a different y=2 meaning than subject 1).
2. Post-windowing, filter trials whose integer label is in `kept_int_set`,
   then re-index to 0..len(kept_classes)-1 in the order of the user's
   `kept_classes` tuple.
3. `classes` is part of `_CACHE_KEY_PARAMS` so 2-class and 4-class entries
   never share cache paths.
4. `iter_per_subject_dl_jobs`, `run_mismatch`, `run_mismatch_jitter`,
   `run_pre_ems_diagonal`, `run_bandpass_mismatch` all accept `classes`
   and pass it through. `run_lofo_matrix` accepts it via `**jitter_kwargs`.

Promoted `IV2A_CLASSES`, `SCHIRR_CLASSES`, `LR_CLASSES` to module-level
in `_datasets.py`. Added `dataset_full_classes(dataset_id)` and
`resolve_classes(dataset_id, classes)` helpers.

### Latent bug fixed: per-subject event mapping was non-deterministic

This is a real bug that pre-dates the rewrite. Without an explicit `mapping`
argument, braindecode's `create_windows_from_events` builds the
description→int map by enumerating distinct event descriptions in
encounter order. For 4-class datasets where every subject has every class
in their data, the order is stable in practice (MOABB always emits classes
in the same order). But for any per-subject pipeline where a subject might
be missing a class (e.g. due to dropped runs, MOABB version drift, or
the 2-class subset itself), the mapping could differ across subjects, and
trial y=2 would not mean the same class. The fix is to always pass an
explicit `mapping` derived from `dataset_full_classes(dataset_id)`.

### Added `refshift.report` module

`report_experiment(df, kind=..., name=..., results_dir=..., figs_dir=...)`
is the single source of truth for: pretty-printing the natural matrix for
each experiment type, computing the summary stats specific to that type,
saving the long-form CSV, and rendering the heatmap (or bar chart, in the
case of jitter/EMS). Five `kind` values: mismatch, jitter_full, lofo,
ems_diag, bandpass.

Lives in the package because: (1) every notebook reimplements the same
logic, (2) unit-testable on synthetic frames, (3) the matrix layout for
each kind is part of the experimental design, not a notebook concern.

Did NOT fold `_run_or_skip` (the CSV-cache-or-recompute helper) into the
package — that's notebook orchestration. The CSV path is in the return
value of `report_experiment` so a notebook can check existence and skip.

### Sample size warnings for 2-class on small datasets

2-class on a 4-class dataset cuts trial count roughly in half. For
Schirrmeister, the 4→2 reduction on `left_hand` vs `right_hand` keeps
~half the trials per subject (the other two classes are dropped). EEGNet
has ~3000 parameters; trial counts in the low hundreds are borderline.
This isn't a bug, but should be noted in the paper writeup: 2-class
results have noisier per-subject accuracies than 4-class.

---

## v0.14.1 — code review fixes

### Schirrmeister cz_ref handling: auto-resolve, validate early

Schirrmeister2017 uses Cz as the recording reference, so there's no Cz
channel in the data. Before this fix, the default `reference_modes` for
every runner was the full 6-mode tuple including `cz_ref`, and on
Schirrmeister the call would crash deep inside `apply_reference` with
"cz_idx=None" — confusing if you didn't already know the dataset.

Fix had two parts:
1. **Helper** `refshift.reference.reference_modes_for_dataset(dataset_id)`
   returns the dataset-safe subset (currently: drops `cz_ref` for
   Schirrmeister, returns full set for the other four).
2. **Auto-resolution**: every runner (`run_mismatch`, `run_mismatch_jitter`,
   `run_lofo_matrix`, `run_pre_ems_diagonal`) now defaults
   `reference_modes=None` and resolves via the helper. Explicit
   `reference_modes=(...)` still overrides.
3. **Early validation** via `validate_reference_modes(modes, graph, dataset_id)`
   fires before any training starts. If a user explicitly passes
   `reference_modes` containing `cz_ref` to a Schirrmeister run, they
   get a clear error pointing at the helper, not a crash 5 minutes into
   subject 1's training.

Considered alternatives:
- **Silent drop**: `validate_reference_modes` could just remove `cz_ref`
  from the modes and warn. Rejected because silent semantic changes are
  the worst kind of bug; a `cz_ref` in the user's config dict needs to
  be a deliberate decision.
- **Per-dataset config files**: overkill. The exclusion logic is one
  line of code; the helper does it.

### `run_mismatch_jitter` train universe

Earlier code hardcoded `REFERENCE_MODES` as the train-time sampler
universe regardless of what the user passed. So even
`run_mismatch_jitter("schirrmeister2017", test_reference_modes=(modes_without_cz_ref))`
would crash, because the train sampler still tried to draw `cz_ref`.

Fix: added a `reference_modes` parameter that is the universe; both
`train_modes` and `test_modes` derive from it. `train_modes_full = universe`,
`train_modes_lofo = universe \ {holdout_ref}`, `test_modes` defaults to
`universe`.

This also means `condition='lofo'` with `holdout_ref='cz_ref'` on
Schirrmeister naturally raises (because `cz_ref` isn't in the
auto-resolved universe), with a clear error.

### `run_pre_ems_diagonal` ignored `laplacian_k` and `montage`

The function declared these as parameters but never passed them through
to `load_dl_data`. The pre-EMS graph was always built with the build_graph
defaults (`k=4`, `montage="standard_1005"`). Same bug existed in v0.13;
not introduced by the rewrite, but inherited.

Fix: threaded `pre_ems_laplacian_k` and `pre_ems_montage` parameters
through `load_dl_data` to `_apply_pre_ems_ref`. Both added to
`_CACHE_KEY_PARAMS` so distinct settings produce distinct cache entries
(otherwise `k=8` would silently read `k=4` cache entries).

### Schema consistency: `dataset` column in `run_pre_ems_diagonal`

The other runners include `dataset: dataset.code` in their row dicts so
results from multiple datasets can be `pd.concat` and grouped without
keys. `run_pre_ems_diagonal` was missing this. Fixed.

### CSP+LDA channel-order check

DL path in `run_mismatch` already asserted full channel-name equality
between data and graph. CSP+LDA path only checked count. Strengthened
to also check order. Catches a hypothetical MOABB version drift that
would silently corrupt every cell of the mismatch matrix.

### Things I considered and chose NOT to fix in this round

- **Pinning braindecode/torch/skorch/mne to exact versions**: premature.
  The right time is after I freeze a working Kaggle environment, not
  now. Pinning to versions that don't exist on Kaggle's image is worse
  than a loose pin.
- **Adding `@pytest.mark.slow` to REST/MNE-heavy tests**: 30s for the
  whole test_reference + test_analysis batch. Not worth the
  scaffolding to mark, document, and remember to skip.
- **Refactoring `__init__.py` to lazy imports**: a reviewer flagged this
  as a hygiene issue, but I verified that `import refshift` doesn't
  actually pull in torch/braindecode/mne/moabb (those are imported
  inside function bodies, not at module top). The package init is
  numpy/pandas/scipy/sklearn only, which are mandatory anyway.

---

## v0.14.0 — initial restructure

### Architecture

#### split `experiments.py` into a package

Single 1400-line module became unreadable. New layout: one file per scientific
question (mismatch, jitter, ems_control, bandpass) plus three shared scaffolding
modules (`_datasets`, `_split`, `_dl_runner`). Considered keeping it flat
with longer functions; rejected because navigating to a runner now means
opening one short file rather than scrolling through a god-file. Trade-off:
8 files for what was 1, but each is short enough to read end-to-end.

#### dropped 2D-array path in reference operators

`apply_reference` previously accepted `(C, T)` and `(N, C, T)`. Only the
tests used the 2D path; nothing in the experimental pipeline. Removed it
along with `np.atleast_3d`/squeeze plumbing. Operators now take `(N, C, T)`
only and `_check_3d` raises on anything else.

#### `ReferenceTransformer` validates at `transform`, not `fit`

Earlier design had a `_check` method called from both `fit` and `transform`,
which validated mode/graph compatibility. Tightened to a single check inside
`apply_reference` (where it matters). `fit` is now a no-op. `transform`
raises if anything's wrong.

#### pre-EMS reference applied AFTER bandpass filter (CRITICAL FIX)

Earlier code put the `Preprocessor(_apply_pre_ems_ref, ...)` step BEFORE
`Preprocessor("filter", ...)` in the chain inside `load_dl_data`. The
docstring claimed the reference was applied "to the filtered raw" but the
code applied it to the broadband raw. For linear operators (CAR, REST,
kNN-Laplacian, cz_ref) this is a no-op — they commute with the bandpass
filter. For median (non-linear) it changes the result. Earlier diagonal
numbers under the EMS-control ablation, for `pre_ems_reference="median"`
specifically, should be regarded as off-spec.

#### EEGNet alias migration

braindecode 1.12 renamed `EEGNetv4` to `EEGNet` (with v4 alias kept as
deprecated). 1.14 removes the alias. `make_dl_model` now does:
```python
try:
    from braindecode.models import EEGNet as _EEGNet
except ImportError:
    from braindecode.models import EEGNetv4 as _EEGNet
```
Works on 1.4 (where I'm currently testing) and 1.12+. requirements.txt
keeps `braindecode>=1.0` since the fallback covers everything.

---

### Reference operator set

#### v0.13 — removed NN-diff

Earlier set included a "nearest-neighbour difference" operator
`Y_i = X_i − X_{nn(i)}`. Removed because it's not a literature-recognised
reference choice (constructed for this codebase as an analogue to clinical
bipolar montages) and its rank deficiency on dense montages would confound
the per-sample jitter and SSL experiments.

#### v0.10 — chose six operators

Three families:
- **Global symmetric**: native, CAR, median, REST. CAR and REST are the
  textbook re-references. Median is a robustness control: same kind of
  global subtraction but non-linear; tells me whether the family-level
  effect I see is "any global operator works" or "specifically linear
  averaging matters."
- **Global asymmetric**: cz_ref. The single-electrode reference real BCI
  hardware uses (Cz, mastoid, earlobe). Rank-1 difference from native: not
  in the symmetric family.
- **Local spatial-derivative**: kNN Laplacian (k=4). Approximation to a
  surface Laplacian; literature precedent in McFarland 1997 (CSD).

LOO mean intentionally excluded: `LOO_i = (C/(C-1)) · CAR_i`, scalar
multiple of CAR, identical for any scale-invariant decoder. GS-like operators
intentionally excluded: data-dependent, don't form a fixed C×C matrix,
break the operator-distance vs transfer-gap analysis.

---

### Datasets

#### Five MOABB MI datasets in the paper

IV-2a, OpenBMI, Cho2017, Dreyer2023, Schirrmeister2017. Choices:
- **IV-2a**: smallest, 9 subjects × 4 classes × 2 sessions; the dataset
  with the longest published-result trail to compare against (calibration
  target = 65.99% mean within-session CSP+LDA).
- **OpenBMI**: 54 subjects × 2 classes × 2 sessions. Largest within-dataset
  cross-session sample. Subject 29 has corrupt .mat in GigaDB release;
  excluded by default.
- **Cho2017**: 52 subjects × 2 classes × 1 session. Shorter trials (~3s);
  stratified split.
- **Dreyer2023**: 87 subjects × 2 classes × 1 session. Largest subject pool;
  needs the mne_bids monkey-patch on Kaggle (read-only `/kaggle/input`).
- **Schirrmeister2017**: 14 subjects × 4 classes; high-density montage;
  Cz is the recording reference (so cz_ref is undefined here — the
  `reference_modes_for_dataset` helper drops it automatically since v0.14.1).

#### Schirrmeister 44-channel motor subset

Used the published motor-cortex subset (Schirrmeister 2017 Section 2.7.1)
not the full 128 channels. The paper reports better accuracy on the subset,
and CSP scales O(C³) so 44 chans cuts CSP+LDA per-subject runtime from ~13min
to ~1min on CPU. Cz excluded from the subset because it was the recording
reference.

---

### Statistical analysis

#### Bootstrap CI + permutation p on operator-distance/transfer-gap correlation

n = 15 upper-triangle pairs (6 references × 5 / 2). At n=15 the asymptotic
Spearman/Pearson p-values from scipy are unreliable. Use bootstrap CI over
pairs (5000 resamples) + permutation p over operator-label shuffles
(10000 perms) with Phipson–Smyth +1/+1 correction. Report all four quantities;
asymptotic p-values are also reported but de-emphasized.

For Schirrmeister, n drops to 10 pairs (5 modes × 4 / 2) since cz_ref is
absent. The permutation approach handles the small-n regime correctly; the
asymptotic p-values become even less trustworthy and should be reported
only with the explicit n in the caption.

#### Holm-Bonferroni for paired Wilcoxon

Per-test-ref Wilcoxon yields one p-value per reference (six values, or
five for Schirrmeister). Multiple comparisons need correction. Holm
step-down preferred over plain Bonferroni because it's strictly more
powerful at the same FWER. The "pooled" row in the output is uncorrected
— different question ("is there an overall effect" vs "which test_refs
differ").

#### REST condition number

`build_graph(..., include_rest=True)` reports a condition number around
1.4e9 for IV-2a's 22-channel layout. Looks alarming but is normal for
spherical-model REST: the Yao 2001 toolbox uses `rcond=1e-4` (which we
match) and similar conditioning is reported in the published REST code.
The reference invariance property `REST(V + a·1) = REST(V)` holds
numerically to ~1e-5 on float32, which is good enough for our purposes.

---

### Things I considered and rejected

#### Implementing per-sample jitter on GPU

The current implementation does CPU round-trip per batch. ~30s overhead per
200-epoch training on T4. A GPU implementation would need separate kernels
for each operator and validation against the numpy reference. Not worth it
at current scale (training cost dominates).

#### Caching post-reference tensors

Cache key in `refshift.data` includes preprocessing params but NOT the
post-window reference operator. So all six reference variants share one
cache entry per (subject, preprocess_params). Considered caching
post-reference too; rejected because (a) reference operators are cheap
relative to braindecode preprocessing, (b) it would 6× the cache storage.

The pre-EMS reference IS in the cache key (`pre_ems_reference`,
`pre_ems_laplacian_k`, `pre_ems_montage`), because it's part of the
preprocessing chain and changing it changes the cached `.npz`.

#### LeaveOneSubjectOut as the headline split

I want to defend a 6×6 reference mismatch matrix per dataset, not a cross-
subject generalisation result. LOSO is a different question (which I might
add later as a secondary analysis). Within-subject, train/test split as
above keeps the variance source clearly attributable to the reference.

---

## Add new entries above this line.
