"""refshift.experiments — calibration and mismatch-matrix runners.

Three entry points:

    calibrate_csp_lda(dataset_id, ...)  - MOABB WithinSession calibration (Phase 1)
    run_mismatch(dataset_id, ...)       - 7x7 mismatch matrix, CSP+LDA or DL
    mismatch_matrix(df, ...)            - pivot long-form -> 7x7 table

``run_mismatch`` dispatches on ``model``: ``'csp_lda'`` uses MOABB's paradigm
interface (Phase 1); ``'eegnet'`` / ``'shallow'`` use ``refshift.dl`` wrappers
around braindecode's canonical MOABB loader.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score

from refshift.pipelines import make_csp_lda_pipeline
from refshift.reference import REFERENCE_MODES, apply_reference, build_graph


# ---------------------------------------------------------------------------
# Dataset registry (lazy MOABB imports so `from refshift import *` is cheap)
# ---------------------------------------------------------------------------

DATASET_IDS = ("iv2a", "openbmi", "cho2017", "dreyer2023")


def _resolve_dataset(dataset_id: str):
    """Return (dataset, paradigm) for a short dataset_id."""
    dataset_id = dataset_id.lower()
    if dataset_id == "iv2a":
        from moabb.datasets import BNCI2014_001
        from moabb.paradigms import MotorImagery
        return BNCI2014_001(), MotorImagery(n_classes=4)
    if dataset_id == "openbmi":
        from moabb.datasets import Lee2019_MI
        from moabb.paradigms import LeftRightImagery
        return Lee2019_MI(), LeftRightImagery()
    if dataset_id == "cho2017":
        from moabb.datasets import Cho2017
        from moabb.paradigms import LeftRightImagery
        return Cho2017(), LeftRightImagery()
    if dataset_id == "dreyer2023":
        from moabb.datasets import Dreyer2023
        from moabb.paradigms import LeftRightImagery
        return Dreyer2023(), LeftRightImagery()
    raise ValueError(
        f"Unknown dataset_id: {dataset_id!r}. Known: {DATASET_IDS}"
    )


# ---------------------------------------------------------------------------
# Small private helpers
# ---------------------------------------------------------------------------

def _get_eeg_channel_names(dataset, subject: Optional[int] = None) -> List[str]:
    """Return EEG channel names in MOABB's native order, peeking at one subject."""
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
):
    """(X_tr, y_tr, X_te, y_te) for a single subject.

    'auto': cross-session if >1 session, else stratified 80/20.
    """
    sessions = sorted(metadata["session"].unique())
    effective = strategy if strategy != "auto" else (
        "session" if len(sessions) > 1 else "stratify"
    )
    if effective == "session":
        train_mask = (metadata["session"] == sessions[0]).to_numpy()
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
    dl_trial_start_offset_s: float = 0.0,
    dl_trial_stop_offset_s: float = 0.0,
) -> pd.DataFrame:
    """Run the 7x7 mismatch matrix on a dataset.

    For each (subject, seed):
      1. Load epoched data (CSP path: MOABB paradigm; DL path: braindecode).
      2. Split train/test (session split if >1 session, else 80/20 stratified).
      3. Pre-compute all 7 test variants once.
      4. For each train_ref, train one model; score on all 7 test variants.

    Parameters
    ----------
    dataset_id : {'iv2a', 'openbmi', 'cho2017', 'dreyer2023'}
    model : {'csp_lda', 'eegnet', 'shallow'}
        ``csp_lda`` uses the MOABB paradigm path (Phase 1).
        ``eegnet`` / ``shallow`` use ``refshift.dl`` (Phase 2).
    subjects : list of int or None
        None -> all subjects in the dataset. For OpenBMI pass
        ``[s for s in range(1, 55) if s != 29]`` (subject 29 is corrupt).
    seeds : list of int
        For stratified-split datasets and for DL training. For CSP+LDA on
        session-split datasets, seeds are near-redundant.
    reference_modes : tuple of str
        Subset of REFERENCE_MODES to evaluate. Order is preserved.
    split_strategy : {'auto', 'session', 'stratify'}
        'auto' picks 'session' if the subject has >1 session, else 'stratify' 80/20.
    n_filters, laplacian_k, montage : Phase 1 knobs (CSP+LDA only / graph build).
    cache : bool
        MOABB paradigm cache (CSP path). Ignored by the DL path.
    progress : bool
        Show tqdm progress bar over (subject, seed) jobs.

    DL options (``dl_``-prefixed) are ignored when ``model='csp_lda'``.

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

    modes = tuple(reference_modes)
    dataset, paradigm = _resolve_dataset(dataset_id)
    if subjects is None:
        subjects = list(dataset.subject_list)
    seeds = list(seeds)

    # Build the neighbor graph once per dataset (only if needed).
    needs_graph = any(m in ("laplacian", "bipolar", "rest") for m in modes)
    needs_rest = "rest" in modes
    graph = None
    if needs_graph:
        ch_names = _get_eeg_channel_names(dataset, subject=subjects[0])
        graph = build_graph(
            ch_names, k=laplacian_k, montage=montage,
            include_rest=needs_rest,
        )

    cache_config = _build_cache_config() if (cache and not is_dl) else None
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

    rows: List[dict] = []
    last_subject: Optional[int] = None
    X = y_int = metadata = None
    sfreq: Optional[float] = None

    for subject, seed in iterator:
        # Reload data only when subject changes (shared across seeds).
        if subject != last_subject:
            if is_dl:
                from refshift.dl import load_dl_data
                X, y_int, metadata, sfreq, _ = load_dl_data(
                    dataset_id, subject,
                    l_freq=dl_l_freq, h_freq=dl_h_freq,
                    trial_start_offset_s=dl_trial_start_offset_s,
                    trial_stop_offset_s=dl_trial_stop_offset_s,
                )
            else:
                X, y_raw, metadata = paradigm.get_data(
                    dataset=dataset, subjects=[subject], **cache_kwargs,
                )
                y_int, _ = _encode_labels(y_raw)
            last_subject = subject

        X_tr, y_tr, X_te, y_te = _split_train_test(
            X, y_int, metadata, strategy=split_strategy, seed=seed,
        )

        # Pre-compute all 7 test variants once.
        X_te_by_ref = {
            m: apply_reference(X_te, m, graph=graph) for m in modes
        }

        n_classes = int(max(int(y_tr.max()), int(y_te.max()))) + 1

        for train_ref in modes:
            X_tr_ref = apply_reference(X_tr, train_ref, graph=graph)

            if is_dl:
                from refshift.dl import make_dl_model
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
            else:
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

            # Release GPU memory between cells to keep Kaggle P100 stable.
            if is_dl:
                del pipe
                _free_cuda()

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
    holdout_ref: str = "bipolar",
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
    dl_trial_start_offset_s: float = 0.0,
    dl_trial_stop_offset_s: float = 0.0,
) -> pd.DataFrame:
    """Train one model per (subject, seed) with reference-jitter augmentation,
    evaluate on all 7 test references.

    Two conditions:

      condition='full':
          Each training sample independently gets a reference drawn uniformly
          from all 7 modes. Tests at-distribution generalization across all
          references the model has seen.

      condition='lofo':
          Each training sample independently gets a reference from
          REFERENCE_MODES \\ {holdout_ref}. The model never sees holdout_ref
          during training. Test-time accuracy on holdout_ref is the cleanest
          probe of operator-invariance — the model has to generalize to a
          previously-unseen reference.

    Parameters
    ----------
    dataset_id : {'iv2a', 'openbmi', 'cho2017', 'dreyer2023'}
    model : {'eegnet', 'shallow'}
        DL only — jitter is meaningful for end-to-end models, not for the
        CSP+LDA pipeline (which has no batched training loop to inject
        per-sample reference randomness into).
    condition : {'full', 'lofo'}
    holdout_ref : str
        Used only when condition='lofo'. Default 'bipolar' (the structurally
        most distinct operator and the strongest test of generalization).
    test_reference_modes : tuple of str
        References to evaluate on at test time. Defaults to all 7.
    Other parameters : see ``run_mismatch``.

    Returns
    -------
    DataFrame with columns
        ``dataset, subject, seed, condition, holdout_ref, train_modes,
        test_ref, accuracy, kappa, n_train, n_test``.

    There is no ``train_ref`` column because each sample sees a different
    reference; ``train_modes`` is the comma-joined set the sampler drew from.
    """
    from refshift.dl import SUPPORTED_DL_MODELS, load_dl_data, make_dl_model
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

    dataset, _paradigm = _resolve_dataset(dataset_id)
    if subjects is None:
        subjects = list(dataset.subject_list)
    seeds = list(seeds)
    test_modes = tuple(test_reference_modes)

    # Reference set for the train-time sampler
    if cond == "full":
        train_modes = tuple(REFERENCE_MODES)
        holdout_label = ""
    else:
        train_modes = tuple(m for m in REFERENCE_MODES if m != holdout_ref)
        holdout_label = holdout_ref
    train_modes_str = ",".join(train_modes)

    # Graph for spatial / REST modes (needed by the train-time sampler if
    # any of laplacian/bipolar/rest is in train_modes, AND/OR by test-time
    # apply_reference for any spatial test mode).
    needs_graph = any(
        m in ("laplacian", "bipolar", "rest")
        for m in set(train_modes) | set(test_modes)
    )
    needs_rest = ("rest" in train_modes) or ("rest" in test_modes)
    graph = None
    if needs_graph:
        ch_names = _get_eeg_channel_names(dataset, subject=subjects[0])
        graph = build_graph(
            ch_names, k=laplacian_k, montage=montage, include_rest=needs_rest,
        )

    try:
        from tqdm.auto import tqdm as _tqdm
    except ImportError:
        def _tqdm(it, **kwargs):
            return it

    jobs = [(s, k) for s in subjects for k in seeds]
    iterator = _tqdm(
        jobs,
        desc=f"[{dataset.code}] {model_lc} jitter-{cond}",
        disable=not progress, leave=True,
    )

    rows: List[dict] = []
    last_subject: Optional[int] = None
    X = y_int = metadata = None
    sfreq: Optional[float] = None

    for subject, seed in iterator:
        if subject != last_subject:
            X, y_int, metadata, sfreq, _ = load_dl_data(
                dataset_id, subject,
                l_freq=dl_l_freq, h_freq=dl_h_freq,
                trial_start_offset_s=dl_trial_start_offset_s,
                trial_stop_offset_s=dl_trial_stop_offset_s,
            )
            last_subject = subject

        X_tr, y_tr, X_te, y_te = _split_train_test(
            X, y_int, metadata, strategy=split_strategy, seed=seed,
        )

        # Pre-compute every test variant once. We intentionally compute all
        # 7 even under LOFO so the table includes the held-out test-ref
        # accuracy in the same row layout as the full-jitter table.
        X_te_by_ref = {
            m: apply_reference(X_te, m, graph=graph) for m in test_modes
        }
        n_classes = int(max(int(y_tr.max()), int(y_te.max()))) + 1

        # Build the per-sample random-reference transform. random_state is
        # tied to the (subject, seed, condition) triple so reproducibility
        # is preserved across re-runs.
        rng_seed = int(1_000_003 * int(seed) + 7919 * int(subject))
        ref_transform = make_random_reference_transform(
            allowed_modes=train_modes,
            graph=graph,
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

        # The training data is intentionally passed in *native* form. The
        # transform re-references each sample at batch-time; if we apply a
        # reference here too, we'd be transforming twice.
        pipe.fit(X_tr, y_tr)

        for test_ref in test_modes:
            y_pred = pipe.predict(X_te_by_ref[test_ref])
            rows.append({
                "dataset":     dataset.code,
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
