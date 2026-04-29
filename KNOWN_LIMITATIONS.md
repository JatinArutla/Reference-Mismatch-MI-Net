# Known limitations and methodological caveats

This document collects every place the codebase deviates from "use the
upstream libraries exactly as documented" and explains why. It also lists
methodological choices a reviewer or user should be aware of when
interpreting results.

If you're writing a paper using `refshift`, the relevant items here belong
in your methods section.

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
results.

---

## Decoder-specific notes

### EEGNet on Cho2017 underperforms at the default learning rate

**Symptom.** `model="eegnet"` on Cho2017 with default `dl_lr=1e-3`
yields a diagonal accuracy of ~55% (chance is 50% for 2-class), making
reference-shift effects on this row of the result matrix uninterpretable.

**Cause.** EEGNet has ~3,000 parameters; on small per-subject training
sets (Cho2017 has 80 train trials per subject under stratified 80/20),
the default learning rate of 1e-3 is too aggressive. Lawhern et al.
(2018) recommend 5e-4 for small-data MI.

**Workaround.** Pass `dl_lr=5e-4` for EEGNet on Cho2017. Other
architectures and other datasets do not need this adjustment.

**Disclosure.** When comparing decoders, report the per-architecture
learning rate explicitly:
- ShallowFBCSPNet: 6.25e-4 (braindecode's MOABB-example default,
  Schirrmeister 2017)
- EEGNet: 1e-3 default; 5e-4 on small-data datasets
  (Lawhern et al. 2018)

### Single-seed DL on large datasets

**Disclosure.** Phase 2 DL experiments on OpenBMI, Cho2017, and
Dreyer2023 are single-seed (seed=0) due to compute budget. IV-2a uses 3
seeds across all conditions.

**Justification.** On IV-2a we have measured seed-level variability at
~2pt vs subject-level variability at ~13pt. Subject variability dominates
seed variability by ~6×, so single-seed runs on the larger datasets are
methodologically defensible. Where this matters for a result, we report
it.

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
