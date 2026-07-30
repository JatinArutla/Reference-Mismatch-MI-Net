"""The four experiments and the calibration check.

Every runner takes ``dataset_id`` first and returns one long-form row per
(subject, seed, ...). Pass ``results_dir`` and a runner will write its table
there and reload it on a rerun instead of retraining, so an interrupted
session resumes where it stopped.

run_mismatch         train on each reference, test on every reference -> an
                     N x N matrix whose diagonal is matched and off-diagonal
                     is mismatched. A large gap means the model is
                     reference-fragile. CSP+LDA or a deep net, optional EA.
run_mismatch_jitter  train ONE deep net with per-sample reference jitter, then
                     test on every reference.
run_loro_matrix      leave one reference out of the jitter mix, one at a time.
run_lofo_matrix      leave a whole family out (see references.FAMILIES).
calibrate_csp_lda    check bare CSP+LDA reproduces MOABB's published baseline.

All models read the same preprocessed windows from preprocess.load_windows.
References are applied to those windows; the deep nets then z-score per trial,
CSP+LDA does not (it is covariance-based and calibrated against MOABB).
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score

from refshift.datasets import moabb_dataset, subject_list
from refshift.models import SUPPORTED_DL_MODELS, make_csp_lda_pipeline, make_dl_model
from refshift.preprocess import load_windows, split_train_test
from refshift.references import (
    FAMILIES,
    GRAPH_MODES,
    REFERENCE_MODES,
    apply_reference_then_ea,
    build_graph,
    canonical_mode_tuple,
    reference_modes_for_dataset,
)

MONTAGE = "standard_1005"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tqdm(iterable, **kwargs):
    from tqdm.auto import tqdm
    return tqdm(iterable, **kwargs)


def _free_cuda():
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _cache_path(results_dir: Optional[str], name: str) -> Optional[str]:
    """Where this experiment's table lives, or None when caching is off."""
    if results_dir is None:
        return None
    os.makedirs(results_dir, exist_ok=True)
    return os.path.join(results_dir, f"{name}.csv")


def _load_cached(path: Optional[str], name: str):
    if path and os.path.exists(path):
        df = pd.read_csv(path)
        print(f"[cached] {name}: {len(df)} rows")
        return df
    return None


def _save(df: pd.DataFrame, path: Optional[str]) -> pd.DataFrame:
    if path:
        df.to_csv(path, index=False)
    return df


def _resolve_modes(dataset_id, reference_modes):
    if reference_modes is None:
        return reference_modes_for_dataset(dataset_id)
    return canonical_mode_tuple(reference_modes)


def _make_graph(modes, ch_names, *, progress=True):
    """Build a DatasetGraph iff a mode needs one; include REST only if used."""
    if not any(m in GRAPH_MODES for m in modes):
        return None
    graph = build_graph(ch_names, montage=MONTAGE, include_rest=("rest" in modes))
    if progress:
        rest = f", REST cond={graph.rest_cond:.2e}" if graph.rest_cond else ""
        print(f"graph: C={len(graph.ch_names)}, cz_idx={graph.cz_idx}{rest}")
    return graph


def _row(subject, seed, train_ref, test_ref, y_true, y_pred, n_train, n_test):
    return {
        "subject": subject, "seed": seed,
        "train_ref": train_ref, "test_ref": test_ref,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "n_train": int(n_train), "n_test": int(n_test),
    }


# ---------------------------------------------------------------------------
# run_mismatch
# ---------------------------------------------------------------------------

def run_mismatch(
    dataset_id: str,
    *,
    model: str = "csp_lda",
    subjects: Optional[List[int]] = None,
    seeds: Sequence[int] = (0,),
    reference_modes: Optional[Sequence[str]] = None,
    apply_ea: bool = False,
    zscore: Optional[bool] = None,
    max_epochs: int = 200,
    batch_size: int = 32,
    progress: bool = True,
    cache_dir: Optional[str] = None,
    results_dir: Optional[str] = None,
) -> pd.DataFrame:
    """N x N train-reference by test-reference matrix.

    ``model`` is 'csp_lda', 'shallow', 'eegnet' or 'atcnet'. With apply_ea=True
    each block is Euclidean-aligned after referencing. CSP+LDA is
    deterministic, so pass a single seed for it.

    ``zscore`` defaults to True for the deep nets and False for CSP+LDA, which
    is the only pipeline difference between them. Set it explicitly to test
    whether a result is about referencing or about standardisation -- the two
    model families disagree most on 'native', where the shared common-mode
    component is largest.
    """
    model = model.lower()
    is_dl = model != "csp_lda"
    if is_dl and model not in SUPPORTED_DL_MODELS:
        raise ValueError(f"Unknown model {model!r}; use 'csp_lda' or {SUPPORTED_DL_MODELS}")

    zscore = is_dl if zscore is None else bool(zscore)
    name = (f"{dataset_id}_{model}_{'EA' if apply_ea else 'noEA'}"
            + ("" if zscore == is_dl else f"_zscore{int(zscore)}"))
    path = _cache_path(results_dir, name)
    cached = _load_cached(path, name)
    if cached is not None:
        return cached

    modes = _resolve_modes(dataset_id, reference_modes)
    subjects = list(subjects) if subjects is not None else subject_list(dataset_id)
    seeds = list(seeds)

    _, _, _, _, ch_names = load_windows(dataset_id, subjects[0], cache_dir=cache_dir)
    graph = _make_graph(modes, ch_names, progress=progress)

    rows = []
    jobs = [(s, k) for s in subjects for k in seeds]
    last_subject = X = y = metadata = sfreq = None

    for subject, seed in _tqdm(jobs, desc=f"[{dataset_id}] {model} mismatch",
                               disable=not progress):
        if subject != last_subject:
            X, y, metadata, sfreq, ch = load_windows(
                dataset_id, subject, cache_dir=cache_dir)
            if graph is not None and ch != graph.ch_names:
                raise RuntimeError(f"subject {subject}: channel order differs from graph")
            last_subject = subject

        X_tr, y_tr, X_te, y_te = split_train_test(X, y, metadata, dataset_id)
        n_classes = int(max(y_tr.max(), y_te.max())) + 1
        X_te_by_ref = {
            m: apply_reference_then_ea(X_te, m, graph=graph, zscore=zscore,
                                       apply_ea=apply_ea)
            for m in modes
        }
        for train_ref in modes:
            X_tr_ref = apply_reference_then_ea(
                X_tr, train_ref, graph=graph, zscore=zscore, apply_ea=apply_ea)

            if is_dl:
                pipe = make_dl_model(
                    model, n_channels=X_tr_ref.shape[1], n_classes=n_classes,
                    n_times=X_tr_ref.shape[2], sfreq=sfreq, seed=seed,
                    max_epochs=max_epochs, batch_size=batch_size)
            else:
                pipe = make_csp_lda_pipeline()
            pipe.fit(X_tr_ref, y_tr)

            for test_ref in modes:
                y_pred = pipe.predict(X_te_by_ref[test_ref])
                rows.append(_row(subject, seed, train_ref, test_ref,
                                 y_te, y_pred, len(y_tr), len(y_te)))
            del pipe
            if is_dl:
                _free_cuda()

    return _save(pd.DataFrame(rows), path)


# ---------------------------------------------------------------------------
# run_mismatch_jitter
# ---------------------------------------------------------------------------

def run_mismatch_jitter(
    dataset_id: str,
    *,
    model: str,
    condition: str = "full",
    holdout_ref: str = "cz_ref",
    subjects: Optional[List[int]] = None,
    seeds: Sequence[int] = (0,),
    reference_modes: Optional[Sequence[str]] = None,
    test_reference_modes: Optional[Sequence[str]] = None,
    max_epochs: int = 200,
    batch_size: int = 32,
    progress: bool = True,
    cache_dir: Optional[str] = None,
    results_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Train one deep net with per-sample reference jitter; test on every ref.

    condition='full' jitters over every reference; condition='loro' jitters
    over all but ``holdout_ref``.
    """
    from refshift.jitter import make_random_reference_transform

    model = model.lower()
    if model not in SUPPORTED_DL_MODELS:
        raise ValueError(f"jitter needs a deep model; got {model!r}")
    if condition not in ("full", "loro"):
        raise ValueError(f"Unknown condition {condition!r}; use 'full' or 'loro'")

    universe = _resolve_modes(dataset_id, reference_modes)
    if condition == "loro" and holdout_ref not in universe:
        raise ValueError(f"holdout_ref={holdout_ref!r} not in {universe}")
    test_modes = (canonical_mode_tuple(test_reference_modes)
                  if test_reference_modes is not None else universe)
    train_modes = (universe if condition == "full"
                   else tuple(m for m in universe if m != holdout_ref))
    all_modes = canonical_mode_tuple(set(train_modes) | set(test_modes))

    # Name the run by what it never trains on. run_lofo_matrix caches its own
    # per-family tables, so these names never collide with a family holdout.
    held = [m for m in all_modes if m not in train_modes]
    name = f"{dataset_id}_{model}_jitter_" + ("full" if not held else "hold-" + "+".join(held))
    path = _cache_path(results_dir, name)
    cached = _load_cached(path, name)
    if cached is not None:
        return cached

    subjects = list(subjects) if subjects is not None else subject_list(dataset_id)
    seeds = list(seeds)
    _, _, _, _, ch_names = load_windows(dataset_id, subjects[0], cache_dir=cache_dir)
    graph = _make_graph(all_modes, ch_names, progress=progress)

    rows = []
    jobs = [(s, k) for s in subjects for k in seeds]
    last_subject = X = y = metadata = sfreq = None

    for subject, seed in _tqdm(jobs, desc=f"[{dataset_id}] {model} jitter-{condition}",
                               disable=not progress):
        if subject != last_subject:
            X, y, metadata, sfreq, ch = load_windows(
                dataset_id, subject, cache_dir=cache_dir)
            if graph is not None and ch != graph.ch_names:
                raise RuntimeError(f"subject {subject}: channel order differs from graph")
            last_subject = subject

        X_tr, y_tr, X_te, y_te = split_train_test(X, y, metadata, dataset_id)
        n_classes = int(max(y_tr.max(), y_te.max())) + 1
        X_te_by_ref = {m: apply_reference_then_ea(X_te, m, graph=graph, zscore=True)
                       for m in test_modes}

        transform = make_random_reference_transform(
            train_modes, graph=graph, random_state=1_000_003 * seed + 7919 * subject)
        pipe = make_dl_model(
            model, n_channels=X_tr.shape[1], n_classes=n_classes,
            n_times=X_tr.shape[2], sfreq=sfreq, seed=seed,
            max_epochs=max_epochs, batch_size=batch_size, transforms=[transform])
        # Training data stays native: the transform re-references and z-scores
        # each sample at batch time, so it must NOT be pre-referenced here.
        pipe.fit(X_tr, y_tr)

        for test_ref in test_modes:
            y_pred = pipe.predict(X_te_by_ref[test_ref])
            rows.append({
                "subject": subject, "seed": seed, "condition": condition,
                "holdout_ref": holdout_ref if condition == "loro" else "",
                "train_modes": ",".join(train_modes), "test_ref": test_ref,
                "accuracy": float(accuracy_score(y_te, y_pred)),
                "kappa": float(cohen_kappa_score(y_te, y_pred)),
                "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
            })
        del pipe
        _free_cuda()

    return _save(pd.DataFrame(rows), path)


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------

def run_loro_matrix(
    dataset_id: str,
    *,
    model: str,
    reference_modes: Optional[Sequence[str]] = None,
    seeds: Sequence[int] = (0,),
    subjects: Optional[List[int]] = None,
    max_epochs: int = 200,
    batch_size: int = 32,
    progress: bool = True,
    cache_dir: Optional[str] = None,
    results_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Leave one reference out of the jitter mix, one at a time.

    Each holdout caches separately, so an interrupted sweep resumes.
    """
    universe = _resolve_modes(dataset_id, reference_modes)
    return pd.concat([
        run_mismatch_jitter(
            dataset_id, model=model, condition="loro", holdout_ref=h,
            reference_modes=universe, seeds=seeds, subjects=subjects,
            max_epochs=max_epochs, batch_size=batch_size, progress=progress,
            cache_dir=cache_dir, results_dir=results_dir)
        for h in universe
    ], ignore_index=True)


def run_lofo_matrix(
    dataset_id: str,
    *,
    model: str,
    families: dict = None,
    reference_modes: Optional[Sequence[str]] = None,
    seeds: Sequence[int] = (0,),
    subjects: Optional[List[int]] = None,
    max_epochs: int = 200,
    batch_size: int = 32,
    progress: bool = True,
    cache_dir: Optional[str] = None,
    results_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Hold out a whole family at a time, train on the rest, test on every ref.

    'native' is not in any family (it is the no-op baseline), so it is not part
    of the LOFO universe. Rows are tagged with holdout_family and test_family.
    """
    families = families or FAMILIES
    mode_to_family = {}
    for fam, members in families.items():
        for m in canonical_mode_tuple(members):
            if m in mode_to_family:
                raise ValueError(f"mode {m!r} is in two families; that is not allowed")
            mode_to_family[m] = fam

    available = reference_modes_for_dataset(dataset_id)
    universe = (canonical_mode_tuple(reference_modes) if reference_modes
                else tuple(m for m in REFERENCE_MODES
                           if m in mode_to_family and m in available))

    frames = []
    for fam in families:
        held = tuple(m for m in universe if mode_to_family.get(m) == fam)
        if not held:                        # e.g. cz_ref on Schirrmeister
            continue
        train_modes = tuple(m for m in universe if m not in held)
        if not train_modes:
            raise ValueError(f"holding out {fam!r} leaves nothing to train on")

        name = f"{dataset_id}_{model}_lofo_{fam}"
        path = _cache_path(results_dir, name)
        df = _load_cached(path, name)
        if df is None:
            df = _save(run_mismatch_jitter(
                dataset_id, model=model, condition="full",
                reference_modes=train_modes, test_reference_modes=universe,
                seeds=seeds, subjects=subjects, max_epochs=max_epochs,
                batch_size=batch_size, progress=progress,
                cache_dir=cache_dir, results_dir=None), path)
        df = df.copy()
        df["condition"] = "lofo"
        df["holdout_ref"] = ",".join(held)
        df["holdout_family"] = fam
        df["test_family"] = df["test_ref"].map(mode_to_family)
        frames.append(df)

    if not frames:
        raise ValueError("No non-empty families to sweep.")
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

IV2A_CSP_LDA_TARGET = 65.99      # MOABB's published CSP+LDA accuracy on IV-2a
IV2A_CSP_LDA_TOL = 2.0


def calibrate_csp_lda(dataset_id: str = "iv2a", *, subjects: Optional[List[int]] = None):
    """Check bare CSP+LDA reproduces MOABB's own baseline.

    Runs through MOABB's WithinSessionEvaluation, not this package's loaders,
    so it is an independent check. For IV-2a the target is 65.99% accuracy over
    all nine subjects; pass ``subjects`` only to smoke-test, since a subset
    will not match the published mean. Returns (results, mean_score, passed).
    """
    from moabb.evaluations import WithinSessionEvaluation

    dataset, paradigm = moabb_dataset(dataset_id)
    if subjects is not None:
        dataset.subject_list = list(subjects)

    evaluation = WithinSessionEvaluation(
        paradigm=paradigm, datasets=[dataset], overwrite=True, random_state=42)
    results = evaluation.process({"CSP+LDA": make_csp_lda_pipeline()})
    score = 100 * results["score"].mean()

    passed = True
    if dataset_id.lower() == "iv2a" and subjects is None:
        passed = abs(score - IV2A_CSP_LDA_TARGET) <= IV2A_CSP_LDA_TOL
        print(f"MOABB target {IV2A_CSP_LDA_TARGET}%: got {score:.2f}% -> "
              f"{'PASS' if passed else 'FAIL'}")
    else:
        print(f"mean score = {score:.2f}% (no published target to compare against)")
    return results, score, passed
