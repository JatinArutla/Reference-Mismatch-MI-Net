"""refshift.experiments — calibration and mismatch-matrix runners.

Three entry points:

    calibrate_csp_lda(dataset_id, ...)  - MOABB WithinSession calibration
    run_mismatch(dataset_id, ...)       - 6x6 mismatch matrix
    mismatch_matrix(df, ...)            - pivot long-form -> 6x6 table

Plus helpers used only inside this module, kept private (leading underscore).
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


# ---------------------------------------------------------------------------
# Calibration (unchanged from 0.1.2 — MOABB WithinSessionEvaluation)
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

    Returns
    -------
    results : pd.DataFrame
        Per-subject/session/pipeline scores from MOABB.
    summary : pd.DataFrame
        Two rows: mean and std per pipeline.
    passed : bool
        True iff both calibration targets (MOABB-match on IV-2a and
        identity-match) are satisfied.
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
# Mismatch matrix (quiet: tqdm progress bar only, no per-subject prints)
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
) -> pd.DataFrame:
    """Run the 6x6 mismatch matrix on a dataset.

    For each (subject, seed):
      1. Load epoched data via MOABB's paradigm.get_data (filter-on-raw,
         correct epoching, scaling).
      2. Split into train/test (session split if >1 session, else 80/20).
      3. Pre-compute 6 test variants; train one CSP+LDA per train_ref;
         score each fitted model on all 6 test variants.

    The function is silent by default apart from an optional tqdm progress
    bar over (subject, seed) pairs. There is no per-subject logging. To
    inspect the completed matrix, call ``mismatch_matrix(df)`` on the
    returned DataFrame or ``plot_mismatch_matrix(df, ...)`` to render it.

    Parameters
    ----------
    dataset_id : {'iv2a', 'openbmi', 'cho2017', 'dreyer2023'}
    model : {'csp_lda'}
        Phase 1 only. 'atcnet'/'eegnet'/'shallow' land in Phase 2.
    subjects : list of int or None
        None -> all subjects in the dataset.
    seeds : list of int
        Seeds for stratified-split datasets. CSP+LDA is nearly
        deterministic on session-split datasets, so [0] is usually
        sufficient there.
    reference_modes : tuple of str
        Subset of REFERENCE_MODES to evaluate. Order is preserved.
    split_strategy : {'auto', 'session', 'stratify'}
        'auto' picks 'session' if the subject has >1 session,
        otherwise 'stratify' 80/20.
    n_filters : int
        CSP spatial filters. MOABB default is 6.
    laplacian_k : int
        Nearest neighbors used for the Laplacian reference. Default 4.
    montage : str
        MNE montage used to compute spatial neighbor indices.
    cache : bool
        If True (default), MOABB caches the epoched array output to disk.
        First subject load is slow; repeats are near-instant.
    progress : bool
        Show a tqdm progress bar over (subject, seed) jobs. Default True.

    Returns
    -------
    pd.DataFrame
        Long-form results with columns:
        dataset, subject, seed, train_ref, test_ref, accuracy, kappa,
        n_train, n_test.
    """
    if model != "csp_lda":
        raise NotImplementedError(
            f"model={model!r} is Phase 2. Phase 1 supports 'csp_lda' only."
        )

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
        ch_names = _get_eeg_channel_names(dataset)
        graph = build_graph(
            ch_names, k=laplacian_k, montage=montage,
            include_rest=needs_rest,
        )

    cache_config = _build_cache_config() if cache else None
    cache_kwargs = {"cache_config": cache_config} if cache_config else {}

    # Progress bar (falls back to a no-op iterable if tqdm isn't installed).
    try:
        from tqdm.auto import tqdm as _tqdm
    except ImportError:  # pragma: no cover
        def _tqdm(it, **kwargs):
            return it

    jobs = [(s, k) for s in subjects for k in seeds]
    iterator = _tqdm(
        jobs, desc=f"[{dataset.code}] mismatch",
        disable=not progress, leave=True,
    )

    rows: List[dict] = []
    last_subject: Optional[int] = None
    X = y_int = metadata = None

    for subject, seed in iterator:
        # Reuse the loaded tensor across seeds for the same subject.
        if subject != last_subject:
            X, y_raw, metadata = paradigm.get_data(
                dataset=dataset, subjects=[subject], **cache_kwargs,
            )
            y_int, _ = _encode_labels(y_raw)
            last_subject = subject

        X_tr, y_tr, X_te, y_te = _split_train_test(
            X, y_int, metadata, strategy=split_strategy, seed=seed,
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
