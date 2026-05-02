"""refshift.experiments — calibration and mismatch-matrix runners.

Five entry points:

    calibrate_csp_lda(dataset_id, ...)   - MOABB WithinSession calibration (Phase 1)
    run_mismatch(dataset_id, ...)        - 6x6 mismatch matrix, CSP+LDA or DL
    run_mismatch_jitter(dataset_id, ...) - DL with per-sample jitter (full or LOFO)
    run_lofo_matrix(dataset_id, ...)     - LOFO sweep across all 6 hold-outs
    run_pre_ems_diagonal(dataset_id, ...) - EMS-control ablation
    run_bandpass_mismatch(dataset_id, ...) - bandpass-mismatch preprocessing control
    mismatch_matrix(df, ...)             - pivot long-form -> 6x6 table

``run_mismatch`` dispatches on ``model``: ``'csp_lda'`` uses MOABB's paradigm
interface (Phase 1); ``'eegnet'`` / ``'shallow'`` use ``refshift.dl`` wrappers
around braindecode's canonical MOABB loader.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score

from refshift.pipelines import make_csp_lda_pipeline
from refshift.reference import REFERENCE_MODES, _GRAPH_MODES, apply_reference, build_graph


# ---------------------------------------------------------------------------
# Dataset registry (lazy MOABB imports so `from refshift import *` is cheap)
# ---------------------------------------------------------------------------

DATASET_IDS = ("iv2a", "openbmi", "cho2017", "dreyer2023", "schirrmeister2017")


# Subjects with known data-quality issues in the public dataset release.
# These are excluded from the default ``subject_list`` returned by
# ``_resolve_dataset`` so that ``run_mismatch(dataset_id, subjects=None, ...)``
# never silently hits a corrupt file mid-run. Callers can still pass
# ``subjects=[...]`` explicitly to include or exclude any subjects they want;
# this only changes the default.
#
# - openbmi (Lee2019_MI) subject 29: corrupt .mat in the GigaDB release;
#   loadmat raises "could not read bytes" mid-stream. Confirmed against
#   Phase 1 results (53/54 subjects, 29 absent).
_KNOWN_BAD_SUBJECTS: dict = {
    "openbmi": frozenset({29}),
}


# Datasets where the train/test split is run-based (not session-based).
# The dataset returns a single MOABB session with two runs ('0train' / '1test'),
# and we want to honour that natural split rather than fall back to stratified
# 80/20. ``_split_train_test`` consults this set when strategy='auto'.
_RUN_SPLIT_DATASETS = frozenset({"schirrmeister2017"})


# Motor-cortex channel subset for Schirrmeister2017. This is the canonical
# 44-channel list from the original Schirrmeister 2017 (Hum. Brain Mapp.)
# paper and the public ``high-gamma-dataset`` example code. Cz is excluded
# because it served as the recording reference electrode in the original
# acquisition (Section 2.7.1 of Schirrmeister et al. 2017: "all central
# electrodes (45), except the Cz electrode which served as the recording
# reference electrode"). The list is 20 standard 10-20 motor channels +
# 24 high-density h-suffix channels available in the 128-channel cap.
#
# Restricting to this subset reduces CSP+LDA per-subject runtime from
# ~13 min to ~1 min on CPU (CSP is O(C^3) in the channel count). Note
# that Schirrmeister et al. also report that using all 128 channels gave
# *worse* accuracy than this 44-channel subset, so the restriction
# matches the published protocol on both efficiency and accuracy grounds.
_SCHIRRMEISTER_MOTOR_CHANNELS = (
    # 20 standard motor channels (Cz excluded — recording reference)
    "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6",
    "C5", "C3", "C1", "C2", "C4", "C6",
    "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
    # 24 high-density h-channels
    "FFC5h", "FFC3h", "FFC1h", "FFC2h", "FFC4h", "FFC6h",
    "FCC5h", "FCC3h", "FCC1h", "FCC2h", "FCC4h", "FCC6h",
    "CCP5h", "CCP3h", "CCP1h", "CCP2h", "CCP4h", "CCP6h",
    "CPP5h", "CPP3h", "CPP1h", "CPP2h", "CPP4h", "CPP6h",
)


def _resolve_dataset(
    dataset_id: str,
    classes: Optional[Sequence[str]] = None,
):
    """Return (dataset, paradigm) for a short dataset_id.

    The dataset's ``subject_list`` is filtered to drop any IDs in
    ``_KNOWN_BAD_SUBJECTS[dataset_id]``. This is the *default* subject list
    used when the caller passes ``subjects=None``; explicit ``subjects=``
    overrides bypass the filter entirely.

    OpenBMI requires a configured ``Lee2019_MI`` to expose all 400
    trials/subject; the configuration is in ``refshift.compat`` so the
    DL path can share it.

    Parameters
    ----------
    dataset_id : str
        One of ``DATASET_IDS``.
    classes : sequence of str or None
        Class subset to load. Defaults to None (= dataset's full class set:
        4 classes for iv2a/schirrmeister2017, 2 classes for the others).
        When provided, must be a non-empty subset of the dataset's classes.
        For iv2a and schirrmeister2017 this is wired into MOABB's
        ``MotorImagery(events=...)`` argument so the paradigm only loads
        trials with these labels. For LeftRightImagery datasets (openbmi,
        cho2017, dreyer2023) the only valid non-default value is
        ``("left_hand", "right_hand")``, which is a no-op (already
        the paradigm's full set); any other class set raises ``ValueError``
        because LeftRightImagery datasets don't contain other classes.

    Notes
    -----
    Used for the binary-reduction ablation: passing
    ``classes=("left_hand", "right_hand")`` on iv2a or schirrmeister2017
    produces a binary version of the 4-class paradigm, useful for
    isolating task-complexity effects on reference-mismatch gaps from
    other dataset-specific factors.
    """
    dataset_id = dataset_id.lower()
    classes_t = tuple(classes) if classes is not None else None

    # Per-paradigm validation of the classes argument.
    _IV2A_CLASSES = ("left_hand", "right_hand", "feet", "tongue")
    _SCHIRR_CLASSES = ("left_hand", "right_hand", "feet", "rest")
    _LR_CLASSES = ("left_hand", "right_hand")

    def _validate_classes(allowed: Tuple[str, ...]) -> None:
        if classes_t is None:
            return
        if len(classes_t) == 0:
            raise ValueError("classes=() is empty; pass None for default.")
        unknown = [c for c in classes_t if c not in allowed]
        if unknown:
            raise ValueError(
                f"Unknown classes for {dataset_id}: {unknown}. "
                f"Allowed: {allowed}"
            )
        if len(set(classes_t)) < 2:
            raise ValueError(
                f"classes={classes_t} has fewer than 2 distinct labels; "
                f"need at least 2 for a classification task."
            )

    if dataset_id == "iv2a":
        from moabb.datasets import BNCI2014_001
        from moabb.paradigms import MotorImagery
        _validate_classes(_IV2A_CLASSES)
        if classes_t is None:
            paradigm = MotorImagery(n_classes=4)
        else:
            paradigm = MotorImagery(events=list(classes_t))
        ds = BNCI2014_001()
    elif dataset_id == "openbmi":
        from moabb.paradigms import LeftRightImagery
        from refshift.compat import make_openbmi_dataset
        _validate_classes(_LR_CLASSES)
        ds, paradigm = make_openbmi_dataset(), LeftRightImagery()
    elif dataset_id == "cho2017":
        from moabb.datasets import Cho2017
        from moabb.paradigms import LeftRightImagery
        _validate_classes(_LR_CLASSES)
        ds, paradigm = Cho2017(), LeftRightImagery()
    elif dataset_id == "dreyer2023":
        from moabb.datasets import Dreyer2023
        from moabb.paradigms import LeftRightImagery
        _validate_classes(_LR_CLASSES)
        ds, paradigm = Dreyer2023(), LeftRightImagery()
    elif dataset_id == "schirrmeister2017":
        from moabb.datasets import Schirrmeister2017
        from moabb.paradigms import MotorImagery
        # 4-class MI (left_hand, right_hand, feet, rest), single session per
        # subject with a natural train/test run split (~880 train + ~160
        # test trials per subject). The run-level split is honoured via
        # ``_RUN_SPLIT_DATASETS``.
        #
        # Channel subset: 44 motor channels (FC*/C*/CP* + h-suffix
        # high-density variants). See ``_SCHIRRMEISTER_MOTOR_CHANNELS`` for
        # the full list and rationale.
        #
        # Resample to 250 Hz to match (a) IV-2a's native rate and (b) the
        # canonical HGD pipeline (Schirrmeister 2017 example.py:
        # ``resample_cnt(cnt, 250.0)``). HGD is recorded at 500 Hz; given
        # the 8-32 Hz bandpass we apply downstream, 250 Hz is well above
        # Nyquist and incurs no signal loss. Resampling halves the per-trial
        # sample count (2000 -> 1000), keeping Shallow's ``filter_time_length``
        # in the same physical-time regime as on IV-2a (~100 ms).
        _validate_classes(_SCHIRR_CLASSES)
        ds = Schirrmeister2017()
        paradigm_kwargs = dict(
            channels=_SCHIRRMEISTER_MOTOR_CHANNELS,
            resample=250.0,
        )
        if classes_t is None:
            paradigm = MotorImagery(n_classes=4, **paradigm_kwargs)
        else:
            paradigm = MotorImagery(events=list(classes_t), **paradigm_kwargs)
    else:
        raise ValueError(
            f"Unknown dataset_id: {dataset_id!r}. Known: {DATASET_IDS}"
        )

    bad = _KNOWN_BAD_SUBJECTS.get(dataset_id, frozenset())
    if bad:
        ds.subject_list = [s for s in ds.subject_list if s not in bad]
    return ds, paradigm


# ---------------------------------------------------------------------------
# Small private helpers
# ---------------------------------------------------------------------------

def _get_eeg_channel_names(
    dataset, subject: Optional[int] = None, paradigm=None,
) -> List[str]:
    """Return the EEG channel names that match the data the paradigm will
    deliver, in the same order as the channel axis of the X array.

    If ``paradigm.channels`` is set (e.g. Schirrmeister2017's 44-channel
    motor subset), the returned list is *that* subset in the order the
    user supplied it. MOABB's ``RawToEpochs`` step calls
    ``mne.pick_channels(..., include=self.channels, ordered=True)``,
    which preserves the order of the ``include`` list — so the X array
    has channels in ``paradigm.channels`` order, not raw-channel order.
    Returning that order here keeps the neighbour graph aligned with the
    channel axis of X.

    If ``paradigm.channels`` is unset, all EEG channels in the raw are
    returned in raw-channel order.

    The neighbour graph for spatial-derivative operators (kNN-Laplacian)
    and REST is built from this list, so it is critical that it matches
    the channel axis of the X array the paradigm produces. Mismatches
    here surface as ``IndexError: index N is out of bounds`` inside
    ``_laplacian``.
    """
    if paradigm is not None and getattr(paradigm, "channels", None):
        # MOABB picks with ordered=True, which preserves include-list
        # order. We do not need to peek at the raw to determine order.
        return list(paradigm.channels)
    if subject is None:
        subject = dataset.subject_list[0]
    raws = dataset.get_data(subjects=[subject])
    raw = next(iter(next(iter(raws[subject].values())).values()))
    types = raw.get_channel_types()
    return [ch for ch, t in zip(raw.ch_names, types) if t == "eeg"]


def _build_cache_config(path: Optional[str] = None):
    """MOABB CacheConfig that saves/reads the final ndarray output."""
    from moabb.datasets.base import CacheConfig
    return CacheConfig(save_array=True, use=True, path=path)


def _encode_labels(y: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """String labels -> contiguous ints [0, n_classes)."""
    classes = sorted(np.unique(y).tolist())
    to_int = {c: i for i, c in enumerate(classes)}
    return np.asarray([to_int[v] for v in y], dtype=np.int64), classes


def _split_train_test(
    X: np.ndarray, y: np.ndarray, metadata: pd.DataFrame,
    *, strategy: str = "auto", test_size: float = 0.2, seed: int = 0,
    dataset_id: Optional[str] = None,
):
    """(X_tr, y_tr, X_te, y_te) for a single subject.

    Strategy resolution under ``'auto'``:
      - dataset is in ``_RUN_SPLIT_DATASETS`` -> split on ``metadata['run']``,
        with ``'0train'`` -> train and ``'1test'`` -> test.
      - >1 session in ``metadata['session']`` -> cross-session
        (first session train, second test).
      - otherwise -> stratified ``test_size`` within the single session.
    """
    sessions = sorted(metadata["session"].unique())
    if strategy == "auto":
        if dataset_id is not None and dataset_id in _RUN_SPLIT_DATASETS:
            effective = "run"
        elif len(sessions) > 1:
            effective = "session"
        else:
            effective = "stratify"
    else:
        effective = strategy

    if effective == "session":
        train_mask = (metadata["session"] == sessions[0]).to_numpy()
        return X[train_mask], y[train_mask], X[~train_mask], y[~train_mask]
    if effective == "run":
        if "run" not in metadata.columns:
            raise ValueError(
                "split strategy 'run' requires a 'run' column in metadata"
            )
        runs = sorted(metadata["run"].unique())
        if len(runs) < 2:
            raise ValueError(
                f"split strategy 'run' needs >=2 runs; got {runs}"
            )
        # Convention: first run alphabetically is train, last is test.
        # For Schirrmeister2017 this gives '0train' -> train, '1test' -> test.
        train_mask = (metadata["run"] == runs[0]).to_numpy()
        return X[train_mask], y[train_mask], X[~train_mask], y[~train_mask]
    if effective == "stratify":
        from sklearn.model_selection import StratifiedShuffleSplit
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=seed,
        )
        tr, te = next(splitter.split(X, y))
        return X[tr], y[tr], X[te], y[te]
    raise ValueError(f"Unknown split strategy: {strategy!r}")


def _free_cuda():
    """Best-effort CUDA cache release between DL model trainings."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Shared DL-runner scaffolding
# ---------------------------------------------------------------------------
#
# Three of the four DL runners (run_mismatch DL branch, run_mismatch_jitter,
# run_bandpass_mismatch train side) share the same setup and per-(subject,
# seed) iteration: resolve dataset, build neighbour graph if any reference
# mode needs one, tqdm over (subject, seed) jobs with subject-level data
# caching, and per-iteration train/test split. The two helpers below
# extract that scaffolding so each runner expresses only its own
# train-once-evaluate-many logic.
#
# run_pre_ems_diagonal does NOT use these helpers because its iteration
# is per-(subject, seed, ref) and each ref triggers a fresh load_dl_data
# call (the reference is part of the preprocessing). The CSP+LDA branch
# of run_mismatch does NOT use these helpers either because it goes
# through paradigm.get_data, not load_dl_data. Both are kept inline.

from dataclasses import dataclass


@dataclass
class _DLRunContext:
    """Shared state for the DL runners.

    Built once by ``_setup_dl_run`` per call to a runner. Holds the
    resolved MOABB dataset id and code, the per-dataset neighbour graph
    (if any of the run's reference modes need one), and the subject and
    seed lists.
    """
    dataset_id: str
    dataset_code: str
    subjects: List[int]
    seeds: List[int]
    graph: 'Optional["DatasetGraph"]'  # forward ref; built lazily


def _setup_dl_run(
    dataset_id: str,
    *,
    subjects: Optional[List[int]],
    seeds: List[int],
    reference_modes_for_graph: tuple,
    laplacian_k: int = 4,
    montage: str = "standard_1005",
    progress: bool = True,
) -> _DLRunContext:
    """Resolve a dataset and build the neighbour graph if any of the
    declared reference modes need one.

    ``reference_modes_for_graph`` is the set of operators the run will
    apply (train side and test side). The graph is built iff any of
    them require neighbour indices, REST, or Cz indexing (i.e. any mode
    in ``_GRAPH_MODES``). REST is included in the graph iff 'rest' is
    among the modes; this avoids the spherical-model construction cost
    when REST isn't used. Logs the REST condition number (when
    applicable) once per run.
    """
    dataset, paradigm = _resolve_dataset(dataset_id)
    if subjects is None:
        subjects = list(dataset.subject_list)

    needs_graph = any(m in _GRAPH_MODES for m in reference_modes_for_graph)
    needs_rest = "rest" in reference_modes_for_graph
    graph = None
    if needs_graph:
        ch_names = _get_eeg_channel_names(
            dataset, subject=subjects[0], paradigm=paradigm,
        )
        graph = build_graph(
            ch_names, k=laplacian_k, montage=montage, include_rest=needs_rest,
        )
        if progress:
            cz_msg = (
                f", cz_idx={graph.cz_idx}"
                if graph.cz_idx is not None else ", cz_idx=None (no Cz channel)"
            )
            rest_msg = (
                f", REST cond={graph.rest_cond:.2e}"
                if graph.rest_cond is not None else ""
            )
            print(
                f"[{dataset.code}] graph: C={len(graph.ch_names)}"
                f"{cz_msg}{rest_msg}"
            )

    return _DLRunContext(
        dataset_id=dataset_id,
        dataset_code=dataset.code,
        subjects=list(subjects),
        seeds=list(seeds),
        graph=graph,
    )


def _iter_per_subject_dl_jobs(
    ctx: _DLRunContext,
    *,
    split_strategy: str = "auto",
    desc: str = "",
    progress: bool = True,
    dl_resample: float = 250.0,
    dl_l_freq: float = 8.0,
    dl_h_freq: float = 32.0,
    dl_trial_start_offset_s: float = 0.0,
    dl_trial_stop_offset_s: float = 0.0,
    dl_cache_dir: Optional[str] = None,
):
    """Generator yielding ``(subject, seed, X_tr, y_tr, X_te, y_te, sfreq)``
    per (subject, seed) job, with subject-level data caching.

    The underlying ``load_dl_data`` call happens only when the subject
    changes; subsequent seeds reuse the in-memory tensor. The
    channel-order assertion against ``ctx.graph`` runs once per subject
    reload so a downstream operator that indexes into X with graph
    indices cannot silently apply to the wrong channels.

    Intended for the train-once-evaluate-many runners (``run_mismatch``
    DL branch, ``run_mismatch_jitter``, ``run_bandpass_mismatch`` train
    side). Other iteration shapes call ``load_dl_data`` directly.
    """
    from refshift.dl import load_dl_data

    try:
        from tqdm.auto import tqdm as _tqdm
    except ImportError:  # pragma: no cover
        def _tqdm(it, **kwargs):
            return it

    jobs = [(s, k) for s in ctx.subjects for k in ctx.seeds]
    iterator = _tqdm(
        jobs, desc=desc or f"[{ctx.dataset_code}]",
        disable=not progress, leave=True,
    )

    last_subject: Optional[int] = None
    X = y_int = metadata = None
    sfreq: Optional[float] = None

    for subject, seed in iterator:
        if subject != last_subject:
            X, y_int, metadata, sfreq, ch_names_subj = load_dl_data(
                ctx.dataset_id, subject,
                resample=dl_resample,
                l_freq=dl_l_freq, h_freq=dl_h_freq,
                trial_start_offset_s=dl_trial_start_offset_s,
                trial_stop_offset_s=dl_trial_stop_offset_s,
                cache_dir=dl_cache_dir,
            )
            if ctx.graph is not None:
                assert list(ch_names_subj) == ctx.graph.ch_names, (
                    f"Channel order mismatch for subject {subject}: "
                    f"data has {ch_names_subj[:5]}... but graph was "
                    f"built from {ctx.graph.ch_names[:5]}..."
                )
            last_subject = subject

        X_tr, y_tr, X_te, y_te = _split_train_test(
            X, y_int, metadata,
            strategy=split_strategy, seed=seed, dataset_id=ctx.dataset_id,
        )
        yield subject, seed, X_tr, y_tr, X_te, y_te, sfreq


# ---------------------------------------------------------------------------
# Calibration (Phase 1 CSP+LDA MOABB match)
# ---------------------------------------------------------------------------

IV2A_CSP_LDA_TARGET = 65.99
IV2A_CSP_LDA_TOL = 2.0
IDENTITY_TOL = 0.5


def calibrate_csp_lda(
    dataset_id: str = "iv2a",
    *,
    subjects: Optional[List[int]] = None,
    random_state: int = 42,
    overwrite: bool = True,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, bool]:
    """Run MOABB WithinSession CSP+LDA calibration.

    Runs two pipelines: bare MOABB canonical and canonical prepended with
    ReferenceTransformer('native'). Identity-pipeline per-fold scores must
    match bare to within 0.5%.
    """
    from moabb.evaluations import WithinSessionEvaluation

    dataset, paradigm = _resolve_dataset(dataset_id)
    if subjects is not None:
        dataset.subject_list = list(subjects)

    pipelines = {
        "CSP+LDA (bare)": make_csp_lda_pipeline(reference_mode=None),
        "CSP+LDA (ReferenceTransformer='native')":
            make_csp_lda_pipeline(reference_mode="native"),
    }

    evaluation = WithinSessionEvaluation(
        paradigm=paradigm, datasets=[dataset],
        overwrite=overwrite, random_state=random_state,
    )
    results = evaluation.process(pipelines)

    summary = (
        results.groupby("pipeline")["score"]
               .agg(["mean", "std", "count"])
               .assign(mean=lambda d: 100 * d["mean"],
                       std=lambda d: 100 * d["std"])
               .round(2)
    )

    bare_mean = 100 * results[
        results["pipeline"] == "CSP+LDA (bare)"
    ]["score"].mean()
    ident_mean = 100 * results[
        results["pipeline"] == "CSP+LDA (ReferenceTransformer='native')"
    ]["score"].mean()

    moabb_ok = (
        abs(bare_mean - IV2A_CSP_LDA_TARGET) <= IV2A_CSP_LDA_TOL
        if dataset_id.lower() == "iv2a" else True
    )
    identity_ok = abs(ident_mean - bare_mean) <= IDENTITY_TOL
    passed = bool(moabb_ok and identity_ok)

    if verbose:
        print()
        print("Per-pipeline summary (mean +/- std across subjects x sessions):")
        for name, row in summary.iterrows():
            print(f"  {name:42s}  {row['mean']:5.2f} +/- {row['std']:5.2f}")
        print()
        if dataset_id.lower() == "iv2a":
            print(
                f"Target 1 (MOABB {IV2A_CSP_LDA_TARGET}% +/- {IV2A_CSP_LDA_TOL}%): "
                f"got {bare_mean:.2f}% --> {'PASS' if moabb_ok else 'FAIL'}"
            )
        print(
            f"Target 2 (identity within {IDENTITY_TOL}%): "
            f"delta={ident_mean - bare_mean:+.2f}% --> "
            f"{'PASS' if identity_ok else 'FAIL'}"
        )

    return results, summary, passed


# ---------------------------------------------------------------------------
# Mismatch matrix — CSP+LDA or DL via braindecode
# ---------------------------------------------------------------------------

def run_mismatch(
    dataset_id: str,
    *,
    model: str = "csp_lda",
    subjects: Optional[List[int]] = None,
    seeds: List[int] = (0,),
    reference_modes: tuple = REFERENCE_MODES,
    classes: Optional[Sequence[str]] = None,
    split_strategy: str = "auto",
    n_filters: int = 6,
    laplacian_k: int = 4,
    montage: str = "standard_1005",
    cache: bool = True,
    progress: bool = True,
    # --- Phase 2 (DL) options ---
    dl_max_epochs: int = 200,
    dl_batch_size: int = 32,
    dl_lr: Optional[float] = None,
    dl_weight_decay: float = 0.0,
    dl_device: Optional[str] = None,
    dl_verbose: int = 0,
    dl_l_freq: float = 8.0,
    dl_h_freq: float = 32.0,
    dl_resample: float = 250.0,
    dl_trial_start_offset_s: float = 0.0,
    dl_trial_stop_offset_s: float = 0.0,
    dl_cache_dir: 'Optional[str]' = None,
) -> pd.DataFrame:
    """Run the 6x6 mismatch matrix on a dataset.

    For each (subject, seed):
      1. Load epoched data (CSP path: MOABB paradigm; DL path: braindecode).
      2. Split train/test (session split if >1 session, else 80/20 stratified).
      3. Pre-compute all 6 test variants once.
      4. For each train_ref, train one model; score on all 6 test variants.

    Parameters
    ----------
    dataset_id : {'iv2a', 'openbmi', 'cho2017', 'dreyer2023', 'schirrmeister2017'}
    model : {'csp_lda', 'eegnet', 'shallow'}
        ``csp_lda`` uses the MOABB paradigm path (Phase 1).
        ``eegnet`` / ``shallow`` use ``refshift.dl`` (Phase 2).
    subjects : list of int or None
        None -> all subjects in the dataset, with known-bad subjects excluded
        (currently: OpenBMI subject 29, due to a corrupt .mat in the GigaDB
        release; see ``_KNOWN_BAD_SUBJECTS`` in this module). Pass an
        explicit list to override (this bypasses the bad-subject filter).
    seeds : list of int
        For stratified-split datasets and for DL training. For CSP+LDA on
        session-split datasets, seeds are near-redundant.
    reference_modes : tuple of str
        Subset of REFERENCE_MODES to evaluate. Order is preserved.
    classes : sequence of str or None, default None
        Class subset for the paradigm. ``None`` uses the dataset's full
        class set (4 classes for iv2a/schirrmeister2017, 2 classes for
        openbmi/cho2017/dreyer2023). Used for the binary-reduction
        ablation: passing ``classes=("left_hand", "right_hand")`` on
        iv2a or schirrmeister2017 produces a binary version of the
        4-class paradigm, which isolates task-complexity effects from
        other dataset-specific factors when comparing reference-mismatch
        gaps across datasets. Currently supported for ``model='csp_lda'``
        only; the DL path raises NotImplementedError because plumbing
        the class subset through ``refshift.dl.load_dl_data`` is a
        separate change.
    split_strategy : {'auto', 'session', 'stratify'}
        'auto' picks 'session' if the subject has >1 session, else 'stratify' 80/20.
    n_filters, laplacian_k, montage : Phase 1 knobs (CSP+LDA only / graph build).
    cache : bool
        MOABB paradigm cache (CSP path). Ignored by the DL path.
    progress : bool
        Show tqdm progress bar over (subject, seed) jobs.

    DL options (``dl_``-prefixed) are ignored when ``model='csp_lda'``.
    ``dl_resample`` is the sample rate used for every dataset on the DL path
    (default 250 Hz, matching IV-2a's native rate). Standardising this
    across datasets keeps the time-domain receptive field of every model
    identical regardless of native acquisition rate.

    Returns
    -------
    pandas.DataFrame with columns
        ``dataset, subject, seed, train_ref, test_ref, accuracy, kappa,
        n_train, n_test``.
    """
    model_lc = model.lower()
    if model_lc == "csp_lda":
        is_dl = False
    else:
        from refshift.dl import SUPPORTED_DL_MODELS
        if model_lc not in SUPPORTED_DL_MODELS:
            raise NotImplementedError(
                f"model={model!r} is not supported. "
                f"Known: 'csp_lda', {SUPPORTED_DL_MODELS}."
            )
        is_dl = True

    if classes is not None and is_dl:
        raise NotImplementedError(
            f"classes={classes!r} is currently only supported for "
            f"model='csp_lda', not for {model!r}. Threading the class "
            f"subset through the DL data loader (refshift.dl.load_dl_data) "
            f"is a separate change. Run the binary-reduction ablation "
            f"with model='csp_lda' for now."
        )

    modes = tuple(reference_modes)
    rows: List[dict] = []

    if is_dl:
        # DL path: helpers handle dataset resolution, graph build, tqdm,
        # subject-level caching, and the train/test split.
        from refshift.dl import make_dl_model

        ctx = _setup_dl_run(
            dataset_id, subjects=subjects, seeds=seeds,
            reference_modes_for_graph=modes,
            laplacian_k=laplacian_k, montage=montage, progress=progress,
        )
        for subject, seed, X_tr, y_tr, X_te, y_te, sfreq in _iter_per_subject_dl_jobs(
            ctx, split_strategy=split_strategy,
            desc=f"[{ctx.dataset_code}] {model_lc} mismatch",
            progress=progress,
            dl_resample=dl_resample,
            dl_l_freq=dl_l_freq, dl_h_freq=dl_h_freq,
            dl_trial_start_offset_s=dl_trial_start_offset_s,
            dl_trial_stop_offset_s=dl_trial_stop_offset_s,
            dl_cache_dir=dl_cache_dir,
        ):
            X_te_by_ref = {
                m: apply_reference(X_te, m, graph=ctx.graph) for m in modes
            }
            n_classes = int(max(int(y_tr.max()), int(y_te.max()))) + 1

            for train_ref in modes:
                X_tr_ref = apply_reference(X_tr, train_ref, graph=ctx.graph)
                pipe = make_dl_model(
                    model=model_lc,
                    n_channels=X_tr_ref.shape[1],
                    n_classes=n_classes,
                    n_times=X_tr_ref.shape[2],
                    sfreq=float(sfreq),
                    seed=int(seed),
                    max_epochs=dl_max_epochs,
                    batch_size=dl_batch_size,
                    lr=dl_lr,
                    weight_decay=dl_weight_decay,
                    device=dl_device,
                    verbose=dl_verbose,
                )
                pipe.fit(X_tr_ref, y_tr)
                for test_ref in modes:
                    y_pred = pipe.predict(X_te_by_ref[test_ref])
                    rows.append({
                        "dataset":   ctx.dataset_code,
                        "subject":   subject,
                        "seed":      seed,
                        "train_ref": train_ref,
                        "test_ref":  test_ref,
                        "accuracy":  float(accuracy_score(y_te, y_pred)),
                        "kappa":     float(cohen_kappa_score(y_te, y_pred)),
                        "n_train":   int(len(y_tr)),
                        "n_test":    int(len(y_te)),
                    })
                del pipe
                _free_cuda()

        return pd.DataFrame(rows)

    # CSP+LDA path: paradigm.get_data is its own data loader and the loop
    # is simple enough to keep inline. The helpers above target the DL
    # path specifically.
    dataset, paradigm = _resolve_dataset(dataset_id, classes=classes)
    if subjects is None:
        subjects = list(dataset.subject_list)
    seeds = list(seeds)

    needs_graph = any(m in _GRAPH_MODES for m in modes)
    needs_rest = "rest" in modes
    graph = None
    if needs_graph:
        ch_names = _get_eeg_channel_names(
            dataset, subject=subjects[0], paradigm=paradigm,
        )
        graph = build_graph(
            ch_names, k=laplacian_k, montage=montage, include_rest=needs_rest,
        )
        if progress:
            cz_msg = (
                f", cz_idx={graph.cz_idx}"
                if graph.cz_idx is not None else ", cz_idx=None (no Cz channel)"
            )
            rest_msg = (
                f", REST cond={graph.rest_cond:.2e}"
                if graph.rest_cond is not None else ""
            )
            print(
                f"[{dataset.code}] graph: C={len(graph.ch_names)}"
                f"{cz_msg}{rest_msg}"
            )

    cache_config = _build_cache_config() if cache else None
    cache_kwargs = {"cache_config": cache_config} if cache_config else {}

    try:
        from tqdm.auto import tqdm as _tqdm
    except ImportError:  # pragma: no cover
        def _tqdm(it, **kwargs):
            return it

    jobs = [(s, k) for s in subjects for k in seeds]
    iterator = _tqdm(
        jobs, desc=f"[{dataset.code}] {model_lc} mismatch",
        disable=not progress, leave=True,
    )

    last_subject: Optional[int] = None
    X = y_int = metadata = None

    for subject, seed in iterator:
        if subject != last_subject:
            X, y_raw, metadata = paradigm.get_data(
                dataset=dataset, subjects=[subject], **cache_kwargs,
            )
            y_int, _ = _encode_labels(y_raw)
            if graph is not None:
                assert X.shape[1] == len(graph.ch_names), (
                    f"Channel count mismatch for subject {subject}: "
                    f"data has {X.shape[1]} channels but graph has "
                    f"{len(graph.ch_names)}."
                )
            last_subject = subject

        X_tr, y_tr, X_te, y_te = _split_train_test(
            X, y_int, metadata, strategy=split_strategy, seed=seed,
            dataset_id=dataset_id,
        )
        X_te_by_ref = {
            m: apply_reference(X_te, m, graph=graph) for m in modes
        }

        for train_ref in modes:
            X_tr_ref = apply_reference(X_tr, train_ref, graph=graph)
            pipe = make_csp_lda_pipeline(
                reference_mode=None, n_filters=n_filters,
            )
            pipe.fit(X_tr_ref, y_tr)
            for test_ref in modes:
                y_pred = pipe.predict(X_te_by_ref[test_ref])
                rows.append({
                    "dataset":   dataset.code,
                    "subject":   subject,
                    "seed":      seed,
                    "train_ref": train_ref,
                    "test_ref":  test_ref,
                    "accuracy":  float(accuracy_score(y_te, y_pred)),
                    "kappa":     float(cohen_kappa_score(y_te, y_pred)),
                    "n_train":   int(len(y_tr)),
                    "n_test":    int(len(y_te)),
                })

    return pd.DataFrame(rows)


def mismatch_matrix(
    df: pd.DataFrame,
    *,
    metric: str = "accuracy",
    aggregate: str = "mean",
) -> pd.DataFrame:
    """Pivot long-form results into a train_ref x test_ref table.

    Aggregates across (subject, seed). ``aggregate`` is 'mean' or 'std'.
    """
    grouped = df.groupby(["train_ref", "test_ref"])[metric]
    if aggregate == "mean":
        return grouped.mean().unstack("test_ref")
    if aggregate == "std":
        return grouped.std().unstack("test_ref")
    raise ValueError(f"Unknown aggregate: {aggregate!r}")


# ---------------------------------------------------------------------------
# Reference-jitter augmentation runner (Phase 2 intervention)
# ---------------------------------------------------------------------------

def run_mismatch_jitter(
    dataset_id: str,
    *,
    model: str,
    condition: str = "full",
    holdout_ref: str = "cz_ref",
    subjects: Optional[List[int]] = None,
    seeds: List[int] = (0,),
    test_reference_modes: tuple = REFERENCE_MODES,
    split_strategy: str = "auto",
    laplacian_k: int = 4,
    montage: str = "standard_1005",
    progress: bool = True,
    dl_max_epochs: int = 200,
    dl_batch_size: int = 32,
    dl_lr: Optional[float] = None,
    dl_weight_decay: float = 0.0,
    dl_device: Optional[str] = None,
    dl_verbose: int = 0,
    dl_l_freq: float = 8.0,
    dl_h_freq: float = 32.0,
    dl_resample: float = 250.0,
    dl_trial_start_offset_s: float = 0.0,
    dl_trial_stop_offset_s: float = 0.0,
    dl_cache_dir: 'Optional[str]' = None,
) -> pd.DataFrame:
    """Train one model per (subject, seed) with reference-jitter augmentation,
    evaluate on all test references.

    Two conditions:

      condition='full':
          Each training sample independently gets a reference drawn uniformly
          from all modes in REFERENCE_MODES. Tests at-distribution
          generalization across all references the model has seen.

      condition='lofo':
          Each training sample independently gets a reference from
          REFERENCE_MODES \\ {holdout_ref}. The model never sees holdout_ref
          during training. Test-time accuracy on holdout_ref is the cleanest
          probe of operator-invariance — the model has to generalize to a
          previously-unseen reference.

    Parameters
    ----------
    dataset_id : {'iv2a', 'openbmi', 'cho2017', 'dreyer2023', 'schirrmeister2017'}
    model : {'eegnet', 'shallow'}
        DL only — jitter is meaningful for end-to-end models, not for the
        CSP+LDA pipeline (which has no batched training loop to inject
        per-sample reference randomness into).
    condition : {'full', 'lofo'}
    holdout_ref : str
        Used only when condition='lofo'. Default 'cz_ref' (a global
        single-electrode reference, structurally distinct from the
        symmetric-globals cluster). Must be a member of REFERENCE_MODES.
        For the full LOFO matrix across all references, see
        ``run_lofo_matrix`` which loops this function and concatenates.
    test_reference_modes : tuple of str
        References to evaluate on at test time. Defaults to all of
        REFERENCE_MODES.
    Other parameters : see ``run_mismatch``.

    Returns
    -------
    DataFrame with columns
        ``dataset, subject, seed, condition, holdout_ref, train_modes,
        test_ref, accuracy, kappa, n_train, n_test``.

    There is no ``train_ref`` column because each sample sees a different
    reference; ``train_modes`` is the comma-joined set the sampler drew from.
    """
    from refshift.dl import SUPPORTED_DL_MODELS, make_dl_model
    from refshift.jitter import make_random_reference_transform

    model_lc = model.lower()
    if model_lc not in SUPPORTED_DL_MODELS:
        raise ValueError(
            f"run_mismatch_jitter requires a DL model. "
            f"Got model={model!r}; supported: {SUPPORTED_DL_MODELS}."
        )
    cond = condition.lower()
    if cond not in ("full", "lofo"):
        raise ValueError(f"Unknown condition: {condition!r}. Use 'full' or 'lofo'.")
    if cond == "lofo" and holdout_ref not in REFERENCE_MODES:
        raise ValueError(
            f"holdout_ref={holdout_ref!r} not in REFERENCE_MODES={REFERENCE_MODES}"
        )

    test_modes = tuple(test_reference_modes)
    if cond == "full":
        train_modes = tuple(REFERENCE_MODES)
        holdout_label = ""
    else:
        train_modes = tuple(m for m in REFERENCE_MODES if m != holdout_ref)
        holdout_label = holdout_ref
    train_modes_str = ",".join(train_modes)

    # Graph must cover both train-time sampler modes and test-time apply_reference modes.
    ctx = _setup_dl_run(
        dataset_id, subjects=subjects, seeds=seeds,
        reference_modes_for_graph=tuple(set(train_modes) | set(test_modes)),
        laplacian_k=laplacian_k, montage=montage, progress=progress,
    )

    rows: List[dict] = []
    for subject, seed, X_tr, y_tr, X_te, y_te, sfreq in _iter_per_subject_dl_jobs(
        ctx, split_strategy=split_strategy,
        desc=f"[{ctx.dataset_code}] {model_lc} jitter-{cond}",
        progress=progress,
        dl_resample=dl_resample,
        dl_l_freq=dl_l_freq, dl_h_freq=dl_h_freq,
        dl_trial_start_offset_s=dl_trial_start_offset_s,
        dl_trial_stop_offset_s=dl_trial_stop_offset_s,
        dl_cache_dir=dl_cache_dir,
    ):
        # Pre-compute every test variant once. Under LOFO we still compute
        # the held-out test variant so the row layout matches full-jitter.
        X_te_by_ref = {
            m: apply_reference(X_te, m, graph=ctx.graph) for m in test_modes
        }
        n_classes = int(max(int(y_tr.max()), int(y_te.max()))) + 1

        # Per-sample random-reference transform. Seeded from (subject, seed)
        # so reproducibility is preserved across re-runs.
        rng_seed = int(1_000_003 * int(seed) + 7919 * int(subject))
        ref_transform = make_random_reference_transform(
            allowed_modes=train_modes,
            graph=ctx.graph,
            probability=1.0,
            random_state=rng_seed,
        )

        pipe = make_dl_model(
            model=model_lc,
            n_channels=X_tr.shape[1],
            n_classes=n_classes,
            n_times=X_tr.shape[2],
            sfreq=float(sfreq),
            seed=int(seed),
            max_epochs=dl_max_epochs,
            batch_size=dl_batch_size,
            lr=dl_lr,
            weight_decay=dl_weight_decay,
            device=dl_device,
            verbose=dl_verbose,
            transforms=[ref_transform],
        )
        # Train data is passed in native form; the transform re-references
        # each sample at batch-time. Applying a reference here too would
        # double-transform.
        pipe.fit(X_tr, y_tr)

        for test_ref in test_modes:
            y_pred = pipe.predict(X_te_by_ref[test_ref])
            rows.append({
                "dataset":     ctx.dataset_code,
                "subject":     subject,
                "seed":        seed,
                "condition":   cond,
                "holdout_ref": holdout_label,
                "train_modes": train_modes_str,
                "test_ref":    test_ref,
                "accuracy":    float(accuracy_score(y_te, y_pred)),
                "kappa":       float(cohen_kappa_score(y_te, y_pred)),
                "n_train":     int(len(y_tr)),
                "n_test":      int(len(y_te)),
            })

        del pipe
        _free_cuda()

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# EMS-control diagonal experiment
# ---------------------------------------------------------------------------

def run_pre_ems_diagonal(
    dataset_id: str,
    *,
    model: str = "shallow",
    subjects: Optional[List[int]] = None,
    seeds: Iterable[int] = (0,),
    reference_modes: Iterable[str] = REFERENCE_MODES,
    split_strategy: str = "auto",
    laplacian_k: int = 4,
    montage: str = "standard_1005",
    progress: bool = True,
    dl_max_epochs: int = 200,
    dl_batch_size: int = 32,
    dl_lr: float = 6.25e-4,
    dl_weight_decay: float = 0.0,
    dl_device: Optional[str] = None,
    dl_verbose: int = 0,
    dl_l_freq: float = 8.0,
    dl_h_freq: float = 32.0,
    dl_resample: float = 250.0,
    dl_trial_start_offset_s: float = 0.0,
    dl_trial_stop_offset_s: float = 0.0,
    dl_cache_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Diagonal-only EMS-control experiment for the deep-learning pipeline.

    Motivation. The standard pipeline applies exponential moving
    standardization (EMS) before the reference operator, because EMS
    runs in ``load_dl_data`` and reference operators are applied to the
    windowed X array by ``run_mismatch``. EMS is per-channel and
    adaptive; it does not commute with channel-mixing reference
    operators. So the standard pipeline measures "reference operators
    applied to EMS-standardized signals," not "reference operators
    applied to raw filtered signals, then standardized."

    This function provides the corresponding control: for each reference
    r in ``reference_modes``, preprocess the raw signal with r applied
    *before* EMS (via ``load_dl_data(pre_ems_reference=r)``), train a
    fresh model on the resulting data, and evaluate on the same-reference
    test split. The output is a 6-element diagonal that can be compared
    directly with the diagonal of the standard ``run_mismatch`` matrix.
    If the two diagonals match closely (within seed noise), the EMS-
    after-reference order doesn't materially affect the per-reference
    accuracy — and by extension, the off-diagonal cluster structure is
    unlikely to be an artifact of operator/EMS non-commutativity.

    Returns
    -------
    pd.DataFrame with columns:
        subject, seed, reference, accuracy, kappa, n_train, n_test
    one row per (subject, seed, reference) cell.
    """
    model_lc = model.lower()
    if model_lc == "csp_lda":
        raise ValueError(
            "run_pre_ems_diagonal is a DL-only ablation. CSP+LDA does not "
            "use exponential moving standardization, so the EMS-control "
            "question doesn't apply. Use run_mismatch with model='csp_lda' "
            "for the standard pipeline."
        )

    from refshift.dl import SUPPORTED_DL_MODELS, load_dl_data, make_dl_model
    if model_lc not in SUPPORTED_DL_MODELS:
        raise ValueError(
            f"Unknown DL model {model!r}; expected one of {SUPPORTED_DL_MODELS}"
        )

    modes = tuple(reference_modes)
    dataset, paradigm = _resolve_dataset(dataset_id)
    if subjects is None:
        subjects = list(dataset.subject_list)
    seeds = list(seeds)

    try:
        from tqdm.auto import tqdm as _tqdm
    except ImportError:  # pragma: no cover
        def _tqdm(it, **kwargs):
            return it

    jobs = [(s, k, r) for s in subjects for k in seeds for r in modes]
    iterator = _tqdm(
        jobs, desc=f"[{dataset.code}] {model_lc} pre-EMS diagonal",
        disable=not progress, leave=True,
    )

    rows: List[dict] = []
    for subject, seed, ref in iterator:
        # Each (subject, ref) pair gets its own preprocess pass with the
        # reference applied to the continuous Raw before EMS. The cache
        # key in load_dl_data includes pre_ems_reference, so repeated
        # calls with the same (subject, ref) reuse the cache.
        X, y_int, metadata, sfreq, ch_names_subj = load_dl_data(
            dataset_id, subject,
            resample=dl_resample,
            l_freq=dl_l_freq, h_freq=dl_h_freq,
            trial_start_offset_s=dl_trial_start_offset_s,
            trial_stop_offset_s=dl_trial_stop_offset_s,
            cache_dir=dl_cache_dir,
            pre_ems_reference=ref,
        )

        X_tr, y_tr, X_te, y_te = _split_train_test(
            X, y_int, metadata, strategy=split_strategy, seed=seed,
            dataset_id=dataset_id,
        )
        n_classes = int(max(int(y_tr.max()), int(y_te.max()))) + 1

        net = make_dl_model(
            model=model_lc,
            n_channels=X_tr.shape[1],
            n_classes=n_classes,
            n_times=X_tr.shape[2],
            sfreq=float(sfreq),
            seed=int(seed),
            max_epochs=dl_max_epochs,
            batch_size=dl_batch_size,
            lr=dl_lr,
            weight_decay=dl_weight_decay,
            device=dl_device,
            verbose=dl_verbose,
        )
        net.fit(X_tr.astype(np.float32, copy=False), y_tr.astype(np.int64, copy=False))
        y_pred = net.predict(X_te.astype(np.float32, copy=False))

        rows.append({
            "subject":   int(subject),
            "seed":      int(seed),
            "reference": ref,
            "accuracy":  float(accuracy_score(y_te, y_pred)),
            "kappa":     float(cohen_kappa_score(y_te, y_pred)),
            "n_train":   int(len(y_tr)),
            "n_test":    int(len(y_te)),
        })

        del net
        _free_cuda()

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Multi-holdout LOFO sweep
# ---------------------------------------------------------------------------

def run_lofo_matrix(
    dataset_id: str,
    *,
    model: str,
    holdout_modes: tuple = REFERENCE_MODES,
    seeds: List[int] = (0,),
    subjects: Optional[List[int]] = None,
    progress: bool = True,
    **jitter_kwargs,
) -> pd.DataFrame:
    """Run leave-one-reference-out for every reference in ``holdout_modes``.

    Equivalent to looping ``run_mismatch_jitter(condition='lofo',
    holdout_ref=h, ...)`` over each ``h`` and concatenating, but with a
    single tqdm-friendly entry point. Used to produce the full LOFO-by-
    held-out-reference table for the C3 (jitter is distribution-matching,
    not invariance) claim.

    Parameters
    ----------
    dataset_id : str
        Dataset id (see ``DATASET_IDS``).
    model : {'eegnet', 'shallow'}
        DL only.
    holdout_modes : tuple of str
        References to hold out one at a time. Defaults to all of
        ``REFERENCE_MODES``. Each pass trains a fresh model with the
        remaining 5 jittered, evaluated on all 6 test references; the
        held-out test reference is the cell of interest, and the other 5
        are reported for completeness so the seen-vs-unseen comparison is
        on the same axis.
    seeds, subjects, progress : forwarded to ``run_mismatch_jitter``.
    **jitter_kwargs : forwarded to ``run_mismatch_jitter``.

    Returns
    -------
    pd.DataFrame
        Long-form concatenation of per-holdout outputs. Same columns as
        ``run_mismatch_jitter``. The ``holdout_ref`` column distinguishes
        the rows corresponding to each LOFO pass.
    """
    frames: List[pd.DataFrame] = []
    for h in holdout_modes:
        if h not in REFERENCE_MODES:
            raise ValueError(
                f"holdout {h!r} not in REFERENCE_MODES={REFERENCE_MODES}"
            )
        df_h = run_mismatch_jitter(
            dataset_id,
            model=model,
            condition="lofo",
            holdout_ref=h,
            seeds=seeds,
            subjects=subjects,
            progress=progress,
            **jitter_kwargs,
        )
        frames.append(df_h)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Bandpass-mismatch control
# ---------------------------------------------------------------------------

def run_bandpass_mismatch(
    dataset_id: str,
    *,
    model: str = "shallow",
    train_band: Tuple[float, float] = (8.0, 32.0),
    test_bands: Tuple[Tuple[float, float], ...] = ((6.0, 32.0), (8.0, 30.0)),
    reference_mode: str = "native",
    subjects: Optional[List[int]] = None,
    seeds: List[int] = (0,),
    split_strategy: str = "auto",
    progress: bool = True,
    dl_max_epochs: int = 200,
    dl_batch_size: int = 32,
    dl_lr: Optional[float] = None,
    dl_weight_decay: float = 0.0,
    dl_device: Optional[str] = None,
    dl_verbose: int = 0,
    dl_resample: float = 250.0,
    dl_trial_start_offset_s: float = 0.0,
    dl_trial_stop_offset_s: float = 0.0,
    dl_cache_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Preprocessing-mismatch control: train under one bandpass, test under another.

    The reference-shift paper claims the off-diagonal collapse in
    ``run_mismatch`` is specifically structured (operator-distance
    predicts it; family clustering organizes it; jitter recovers it),
    not generic preprocessing brittleness. To rule out the obvious
    confounder, this function trains a model under ``train_band`` on a
    fixed reference (default native) and evaluates the same model on
    test data preprocessed with each of the ``test_bands``. The
    expectation is that bandpass mismatch produces a much smaller
    accuracy drop (typically <5 pts on IV-2a Shallow) than reference
    mismatch (typically 20+ pts). Reporting both gives the reader a
    quantitative baseline for "how big a drop is normal under any
    preprocessing change" against which the reference-mismatch gap can
    be compared.

    The test bands are processed independently: each one triggers a
    full ``load_dl_data`` re-preprocess on the test split. The train
    split is preprocessed once. Reference operator is held fixed at
    ``reference_mode`` throughout (not jittered, not mismatched) so the
    only varying factor is the bandpass.

    Parameters
    ----------
    dataset_id : str
    model : {'eegnet', 'shallow'}
        DL only — bandpass mismatch on CSP+LDA is uninteresting because
        CSP is band-power agnostic up to scale.
    train_band : (l_freq, h_freq)
        Bandpass used for training.
    test_bands : tuple of (l_freq, h_freq)
        Bandpasses used for testing. The matched-band condition (=
        ``train_band``) is added automatically as the diagonal control.
    reference_mode : str
        Reference operator applied to both train and test (always the
        same one). Default 'native'.
    Other parameters : as ``run_mismatch``.

    Returns
    -------
    pd.DataFrame with columns
        ``dataset, subject, seed, reference, train_band, test_band,
        accuracy, kappa, n_train, n_test``.
        ``train_band`` and ``test_band`` are stored as "Lf-Hf" strings
        (e.g. "8.0-32.0") so the result is csv-friendly.
    """
    from refshift.dl import SUPPORTED_DL_MODELS, load_dl_data, make_dl_model

    model_lc = model.lower()
    if model_lc not in SUPPORTED_DL_MODELS:
        raise ValueError(
            f"run_bandpass_mismatch is DL-only. "
            f"Got model={model!r}; supported: {SUPPORTED_DL_MODELS}."
        )
    if reference_mode not in REFERENCE_MODES:
        raise ValueError(
            f"reference_mode={reference_mode!r} not in REFERENCE_MODES"
        )

    # Train side uses the shared helpers; the graph only needs to support
    # `reference_mode` (the single ref applied to both train and test).
    ctx = _setup_dl_run(
        dataset_id, subjects=subjects, seeds=seeds,
        reference_modes_for_graph=(reference_mode,),
        montage="standard_1005", progress=progress,
    )

    # All bands to evaluate (train_band always included as the diagonal).
    all_test_bands = (train_band,) + tuple(b for b in test_bands if b != train_band)
    bands_str = lambda b: f"{b[0]:.1f}-{b[1]:.1f}"

    rows: List[dict] = []
    for subject, seed, X_tr_split, y_tr, _Xdrop, _ydrop, sfreq in _iter_per_subject_dl_jobs(
        ctx, split_strategy=split_strategy,
        desc=f"[{ctx.dataset_code}] {model_lc} bandpass",
        progress=progress,
        dl_resample=dl_resample,
        dl_l_freq=train_band[0], dl_h_freq=train_band[1],
        dl_trial_start_offset_s=dl_trial_start_offset_s,
        dl_trial_stop_offset_s=dl_trial_stop_offset_s,
        dl_cache_dir=dl_cache_dir,
    ):
        X_tr_ref = apply_reference(X_tr_split, reference_mode, graph=ctx.graph)
        n_classes = int(max(int(y_tr.max()), int(_ydrop.max()))) + 1

        net = make_dl_model(
            model=model_lc,
            n_channels=X_tr_ref.shape[1],
            n_classes=n_classes,
            n_times=X_tr_ref.shape[2],
            sfreq=float(sfreq),
            seed=int(seed),
            max_epochs=dl_max_epochs,
            batch_size=dl_batch_size,
            lr=dl_lr,
            weight_decay=dl_weight_decay,
            device=dl_device,
            verbose=dl_verbose,
        )
        net.fit(X_tr_ref, y_tr)

        # For each test band, re-preprocess with that band and evaluate.
        # The helper can't drive this loop because the bandpass changes
        # per iteration (different cache key, different load).
        for tb in all_test_bands:
            X_te_raw, y_te_raw, meta_te, _sfreq_te, _chs = load_dl_data(
                dataset_id, subject,
                resample=dl_resample,
                l_freq=tb[0], h_freq=tb[1],
                trial_start_offset_s=dl_trial_start_offset_s,
                trial_stop_offset_s=dl_trial_stop_offset_s,
                cache_dir=dl_cache_dir,
            )
            _Xdrop2, _ydrop2, X_te_split, y_te = _split_train_test(
                X_te_raw, y_te_raw, meta_te,
                strategy=split_strategy, seed=seed, dataset_id=dataset_id,
            )
            X_te_ref = apply_reference(X_te_split, reference_mode, graph=ctx.graph)
            y_pred = net.predict(X_te_ref)

            rows.append({
                "dataset":    ctx.dataset_code,
                "subject":    int(subject),
                "seed":       int(seed),
                "reference":  reference_mode,
                "train_band": bands_str(train_band),
                "test_band":  bands_str(tb),
                "accuracy":   float(accuracy_score(y_te, y_pred)),
                "kappa":      float(cohen_kappa_score(y_te, y_pred)),
                "n_train":    int(len(y_tr)),
                "n_test":     int(len(y_te)),
            })

        del net
        _free_cuda()

    return pd.DataFrame(rows)
