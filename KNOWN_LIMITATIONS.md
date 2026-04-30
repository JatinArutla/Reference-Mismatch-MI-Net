# Known limitations and methodological caveats

This document collects every place the codebase deviates from "use the
upstream libraries exactly as documented" and explains why. It also lists
methodological choices a reviewer or user should be aware of when
interpreting results.

If you're writing a paper using `refshift`, the relevant items here belong
in your methods section.

---

## Reference-operator set (v0.10 redesign after peer review)

Three rounds of peer review on the v0.9 reference set led to a
6-operator scheme in v0.10. The changes, in order of importance:

**1. `gs` was dropped.** The natural data-dependent Gram-Schmidt
projection is not a fixed C×C linear operator; it is per-trial,
per-channel, depends on the input time-series, and behaves nonlinearly
under input scaling. It does not fit the operator-shift framework the
paper advances. We did not replace `gs` with a linear "leave-one-out
mean" alternative because LOO_i = (C/(C−1)) · CAR_i — they are a
constant scaling of each other and produce identical decoder outputs
under any scale-invariant decoder (CSP+LDA's eigenvalue problem and
batch-normalised neural networks both qualify). LOO would appear as a
duplicate of CAR in every empirical result.

**2. `bipolar` was renamed to `nn_diff`.** The operation `X_i − X_{nn(i)}`
is a dimension-preserving nearest-neighbour local-difference operator,
*not* a clinical bipolar montage. Clinical bipolar montages use
predefined electrode pairs and typically reduce the channel count.
Calling our operator "bipolar" overloaded a clinical term; "NN-diff" is
honest about what we compute. The implementation is unchanged.

**3. NN-diff rank diagnostic.** When nearest-neighbour pairs are mutual
(e.g. C3↔CP3 are each other's nearest neighbours), the operator
destroys two dimensions instead of one. `build_graph` now computes the
rank of `(I − P)` where P is the channel-permutation matrix from
`nn_diff_idx`, and stores it on `DatasetGraph.nn_diff_rank` /
`nn_diff_nullity`. Logged at run start so the operator's effective
rank is on record per dataset.

**4. REST regularization.** REST's pseudo-inverse uses `rcond=1e-4`
explicitly, matching the realistic-head-model REST literature. The
default numpy `pinv` rcond depends on the largest singular value and
can be too aggressive for well-conditioned but small leadfields.

**5. Pre-EMS reference control (`run_pre_ems_diagonal`).** Exponential
moving standardization in the deep-learning pipeline runs *before*
reference operators in `run_mismatch`, because EMS happens in
`load_dl_data` and reference operators are applied to the windowed X
array. EMS is per-channel and adaptive; it does not commute with
channel-mixing reference operators. The standard pipeline therefore
measures "reference applied to EMS-standardized signals" rather than
"reference applied to raw filtered signals, then standardized." The
function `run_pre_ems_diagonal` runs the corresponding control: for
each reference r, preprocess with r applied *before* EMS, train and
test on the same r, return a 6-element diagonal. Compare to the
diagonal of `run_mismatch` to verify EMS-after-reference is not
materially distorting per-reference accuracies. Use only as an
ablation; the headline matrix uses the standard pipeline.

**6. Naming throughout.** "kNN local Laplacian (not formal CSD)";
"REST-like spherical-model re-reference (not validated against a
canonical REST implementation)"; "median as robustness control, not a
mainstream MI reference"; "NN-diff (not clinical bipolar)". The paper
should mirror these qualifications.

---

## v0.11 changes (current)

These changes consolidate the codebase against issues that surfaced
during a deep code review prior to the multi-decoder, multi-dataset
re-run.

**1. DL pipeline resamples to a common rate.** Previously the DL path ran
each dataset at its native acquisition rate (IV-2a 250 Hz, OpenBMI
1000 Hz, Cho2017/Dreyer2023 512 Hz, Schirrmeister 500 Hz), so
ShallowFBCSPNet's `filter_time_length=25` corresponded to wildly
different physical-time receptive fields (~25 ms to ~100 ms). The DL
pipeline now resamples every dataset to a common rate
(`load_dl_data(resample=250.0)` default), with `resample` part of the
disk-cache key so different rates get separate cache entries. The
CSP+LDA path is left at MOABB's native paradigm settings (only
Schirrmeister has `paradigm.resample=250.0`); CSP-based decoders are
not sensitive to absolute sample rate, so the path inconsistency does
not affect cross-dataset CSP+LDA comparability.

**2. Uniform EEGNet learning rate.** `make_dl_model("eegnet", ...)` now
defaults to `lr=5e-4` for all datasets — Lawhern et al. 2018's
recommendation for small-data MI. The previous default of 1e-3
produced chance-level results on Cho2017 specifically (EEGNet has
~3,000 parameters; 80 train trials per subject under stratified 80/20
overshoots at higher LRs). The earlier per-dataset override of 5e-4
for Cho2017 only is removed.

**3. Bug fixes that prevented prior releases from running:**

  - `run_pre_ems_diagonal` (the EMS-control ablation, listed as the
    highest-priority unrun experiment in earlier handoffs) was calling
    `make_dl_model` with parameter names `name=` / `n_chans=` /
    `input_window_samples=` against a function whose signature is
    `model=` / `n_channels=` / `n_times=`. It would TypeError on the
    first iteration and never produced a result. Fixed.

  - `run_mismatch_jitter` referenced a variable `paradigm` that was
    bound as `_paradigm` two lines earlier. NameError on every default
    call (because `REFERENCE_MODES` always includes graph-requiring
    spatial modes). Fixed.

  - `dl.py`'s Schirrmeister branch used `pick_channels(ordered=False)`
    while `_get_eeg_channel_names` returns `paradigm.channels` in
    user-supplied order (because MOABB's `RawToEpochs` calls
    `pick_channels(ordered=True)`). The graph and X-axis-1 channel
    orders therefore disagreed; the runtime assertion
    `list(ch_names_subj) == graph.ch_names` would have caught it and
    crashed. Now `ordered=True`.

  - The `load_dl_data` cache-key tuple `_CACHE_KEY_PARAMS` did not
    include `pre_ems_reference`, even though the function set the value
    in `params`. Calls with `pre_ems_reference=X` would silently return
    cached results from `pre_ems_reference=None`. Fixed; the cache key
    now includes both `pre_ems_reference` and `resample`. Old caches
    are invalidated by the key change (this is intentional — the
    contents change too).

**4. Documentation alignment.** The previous `dl.py` module docstring
asserted that applying linear references to the EMS-standardized
windowed tensor was "numerically equivalent to applying them in
raw-space." This was false; CAR(EMS(X)) and EMS(CAR(X)) differ because
EMS divides each channel by its own running standard deviation, so
per-channel scales are not identical at the moment CAR sums them. The
docstring is now consistent with this section's caveat (§5 above).

**5. Operator-distance statistics tightened.** With 6 operators we have
n=15 pairs in the upper triangle; the asymptotic Spearman/Pearson
p-values used in v0.9/v0.10 were not reliable at this n. The function
`operator_distance_correlation` now returns: a bootstrap 95% confidence
interval over pairs (resampling pair indices with replacement); a
permutation p-value computed by shuffling operator labels of the gap
matrix while keeping the distance matrix fixed; and an averaged
linear-operator estimate over multiple Gaussian probes (default 8) to
reduce variance in the median operator's linear-tangent estimate. The
asymptotic values are still returned for completeness; the bootstrap CI
and permutation p are the values to report in the paper.

**6. Two new experiment runners.** `run_lofo_matrix` wraps
`run_mismatch_jitter(condition='lofo', ...)` in a loop over each
reference in `REFERENCE_MODES`, producing the full
holdout-by-test-reference table in one call. `run_bandpass_mismatch`
trains under one bandpass and tests under others (default train
8-32 Hz, test 6-32 Hz and 8-30 Hz, reference held fixed at native);
the resulting accuracy drops give a baseline for "how big is the drop
under any preprocessing change of comparable magnitude" against which
the reference-mismatch gap can be compared. Both are DL-only.

---

## Upstream-library workarounds

### MOABB Lee2019_MI session-filter inconsistency

**Affects:** OpenBMI dataset, both CSP+LDA and DL paths.

**Symptom under default usage.** `Lee2019_MI()` returns 100 trials per
subject (calibration run from one session) instead of the documented 200
that MOABB's own benchmark paper uses (Chevallier et al. 2024,
Table 1: 50 trials/class × 2 classes × 2 sessions = 200/subject).

**Cause.** `Lee2019.__init__` stores the user-facing `sessions=(1, 2)` as
`_selected_sessions` for a filter in `BaseDataset.get_data`, but
`_get_single_subject_data` writes session keys as zero-indexed strings
(`'0'`, `'1'`). The filter then drops every key not in `{'1', '2'}`,
silently throwing away session `'0'` on every call.

**Workaround.** `refshift.compat.make_openbmi_dataset()` constructs
`Lee2019_MI()` with the default `test_run=False` (matching the MOABB
benchmark paper protocol) and clears `_selected_sessions` to `None`.
Both the CSP+LDA path (`_resolve_dataset` in `experiments.py`) and the
DL path (`make_braindecode_dataset` in `compat.py`) use this shim.
After the shim, OpenBMI returns 200 trials per subject from the
calibration runs of both sessions.

**Why we don't include the test phase.** MOABB's docstring on
`Lee2019_MI.test_run` explicitly warns that *"test_run for MI and SSVEP
do not have labels associated with trials: these runs could not be used
in classification tasks."* Although the labels technically exist in the
.mat files, they reflect the cued direction during real-time classifier
feedback rather than the subject's intent. The MOABB benchmark paper
deliberately excludes the test phase for this reason; we follow the same
protocol.

**Tracking.** Remove the `_selected_sessions = None` workaround once
MOABB fixes the session-label inconsistency in `Lee2019`.

### Dreyer2023 BIDS lock-file handling on read-only filesystems

**Affects:** Dreyer2023 dataset on Kaggle (and any read-only
filesystem).

**Symptom.** `mne_bids` writes lock files into the dataset directory. On
Kaggle's read-only `/kaggle/input` mount, this raises a `PermissionError`
on first access.

**Workaround.** `refshift.env._setup_dreyer_symlinks` mirrors the dataset
directory with per-file symlinks under `/kaggle/working/mne_data/...`,
which is writable. A monkey-patch of MOABB's `download_by_subject` skips
the unzip step that would otherwise re-download files we've symlinked.

**Tracking.** This is a Kaggle environment limitation, not a MOABB bug.
On a normal filesystem, no workaround is needed.

### EEGNet uses an EEGNetv4 alias from braindecode

**Affects:** `make_dl_model("eegnet", ...)` — emits one `FutureWarning`
per call.

**Reason.** Braindecode 1.12 renamed `EEGNetv4` to `EEGNet`; the alias
will be removed in v1.14.

**Action.** Cosmetic; will be addressed when braindecode forces the
rename.

---

## Dataset-level methodological choices

### OpenBMI subject 29 is excluded by default

**Reason.** The `.mat` file for subject 29 in the GigaDB release is
truncated; `scipy.io.loadmat` raises "could not read bytes" mid-stream.
This is a defect in the public dataset, not in our code.

**Behavior.** `_resolve_dataset("openbmi")` returns 53/54 subjects by
default. Pass `subjects=[...]` explicitly to override.

### Schirrmeister2017 uses the canonical 44-channel motor subset

**Affects:** Schirrmeister2017, both CSP+LDA and DL paths.

**What we do.** The pipeline restricts Schirrmeister2017 to the 44 motor
channels published in Schirrmeister et al. 2017 (Section 2.7.1) and the
canonical ``high-gamma-dataset`` example code (FC*, C*, CP* and their
h-suffix high-density variants; Cz is excluded as the recording
reference). The full 128-channel cap is not used.

**Why.** Schirrmeister et al. (2017) report that this 44-channel subset
gave *better* accuracy than using all 128 channels for both ConvNets
and FBCSP — using all electrodes "led to worse accuracies" per their
Section A.7. CSP also scales as O(C^3) in the channel count, so the
restriction additionally cuts per-subject CSP+LDA runtime from ~13 min
to ~1 min on CPU. There is no loss of relevant signal: the dropped
channels are over occipital, frontal-pole, and temporal sites that
don't contribute to motor decoding.

**Implication.** Our channel selection matches the published HGD
protocol exactly. Other deviations from the canonical Schirrmeister
2017 pipeline — bandpass to 8–32 Hz instead of 0–125 Hz, trial-wise
training instead of cropped, no FBCSP feature selection — are
deliberate cross-dataset standardization choices for this paper, not
mistakes; they are why our absolute accuracies (~70% diagonal CSP+LDA)
are below their published 91.2% but they do not affect the
reference-shift effect we measure.


### Train/test split protocol differs across datasets

| Dataset | Sessions / structure | Split protocol |
|---|---|---|
| IV-2a (BNCI2014_001) | 2 sessions (different days) | Cross-session: session 0 train, session 1 test |
| OpenBMI (Lee2019_MI) | 2 sessions (different days, with our compat shim) | Cross-session: session 0 train, session 1 test |
| Cho2017 | 1 session | Stratified 80/20 within session |
| Dreyer2023 | 1 session | Stratified 80/20 within session |
| Schirrmeister2017 | 1 session, 2 runs (`0train` ~880 trials + `1test` ~160 trials) | Run-based: `0train` train, `1test` test (matches the original Schirrmeister 2017 paper) |

**Implication.** The reference-shift effect is being tested against:

- a held-out *recording day* on IV-2a and OpenBMI (strongest evidence of
  robustness to natural inter-session variation),
- a held-out *trial subset* on Cho2017 and Dreyer2023 (single-session
  datasets in their MOABB releases),
- a held-out *run* on Schirrmeister2017 (within the same recording day,
  but matching the dataset's documented evaluation protocol).

This is documented in the methods section of any paper using these
results. To address the obvious reviewer objection that the multi-dataset
gap variation might be partly attributable to protocol heterogeneity,
the paper should also report an 80/20 sensitivity check on IV-2a and
OpenBMI: rerun with `split_strategy='stratify'` and compare the mean
gap to the cross-session result. If they agree to within seed noise,
the cluster structure is not driven by the split protocol.

---

## Decoder-specific notes

### Single-seed DL on large datasets

**Disclosure.** Phase 2 DL experiments on OpenBMI, Cho2017, and
Dreyer2023 have historically been single-seed (seed=0) due to compute
budget; IV-2a uses 3 seeds across all conditions. With v0.11's resample
standardization the DL preprocessing cache is invalidated, so the
re-run sweep is the right time to bring at least one secondary dataset
to 3 seeds (Schirrmeister is the natural choice given the largest
absolute gap reported in v0.9).

**Justification.** On IV-2a we have measured seed-level variability at
~2 pt vs subject-level variability at ~13 pt. Subject variability
dominates seed variability by ~6×, so single-seed runs on the larger
datasets are methodologically defensible for headline numbers but cost
statistical power on per-test-ref Wilcoxon tests.

### CSP+LDA path runs at native sample rates (except Schirrmeister)

**Affects:** CSP+LDA on OpenBMI, Cho2017, Dreyer2023.

**What we do.** Only Schirrmeister2017 sets `paradigm.resample=250.0`
in `_resolve_dataset`. The other paradigms inherit MOABB's defaults
(no resampling), so OpenBMI runs at 1000 Hz, Cho2017 and Dreyer2023 at
512 Hz, IV-2a at its native 250 Hz.

**Why.** CSP's covariance-based decoder is insensitive to absolute
sample rate (the eigenvalue problem is scale-invariant; the bandpass
strips content above 32 Hz on every dataset). The IV-2a 65.99% MOABB
calibration target was computed at native 250 Hz, and we want to
preserve direct comparability to MOABB's published results where
possible. Adding resample=250.0 universally would silently invalidate
that comparability for non-IV-2a datasets.

**Implication for the DL path.** The DL path *does* resample to a
common 250 Hz on every dataset (see `dl_resample` in `run_mismatch`).
The CSP+LDA and DL paths therefore see slightly different inputs on
non-IV-2a datasets. This is documented in methods; CSP+LDA and DL
results are not directly comparable on the same dataset because of
this and other architectural differences (covariance vs convolutional
features), so the path inconsistency is not a confound.

---

## Known cosmetic warnings

The following warnings appear during normal operation and can be
ignored:

- `RuntimeWarning: Setting non-standard config type: "MNE_DATASETS_..."` —
  MOABB writing its config; benign.
- `Could not read the /root/.mne/mne-python.json json file...` — first-run
  MNE config bootstrap on a fresh filesystem; benign.
- `pick_types() is a legacy function` — MNE deprecation warning emitted
  inside braindecode's preprocess; cosmetic until braindecode updates.
- `apply_on_array can only be True if fn is a callable function` — emitted
  in some braindecode versions when introspecting a `Preprocessor` step;
  cosmetic, no behavioural impact (we already use a named function, not a
  lambda).
- `EEGNetv4() is a deprecated class` — see "EEGNet uses an EEGNetv4 alias"
  above.
