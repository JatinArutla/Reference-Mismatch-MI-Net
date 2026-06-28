"""The four experiments and the calibration check, in one place.

Every runner takes ``dataset_id`` first (one of datasets.DATASET_IDS).

run_mismatch
    Train a model on each reference, test on every reference -> an N x N matrix
    whose diagonal is matched and off-diagonal is mismatched. A large gap means
    the model is reference-fragile. CSP+LDA or the three deep nets, optional EA.

run_mismatch_jitter
    Train ONE deep net with per-sample reference jitter, then test on every
    reference. condition='full' jitters over all references; condition='loro'
    jitters over all-but-one (used by run_loro_matrix).

run_loro_matrix  (Leave-One-Reference-Out)
    Sweep the jitter runner, holding out one reference at a time.

run_lofo_matrix  (Leave-One-Family-Out)
    Hold out a whole family (see references.FAMILIES), train on the others,
    test on every reference; tag each row with held-out and test family.

calibrate_csp_lda
    Confirm bare CSP+LDA reproduces the dataset's MOABB baseline and that a
    no-op reference transformer changes nothing.

Two data sources, on purpose:
  * Deep nets use braindecode windows from preprocess.load_windows (bandpassed
    microvolts, not z-scored). References are applied to those windowed trials,
    then per-trial z-score (see apply_reference_then_ea, zscore=True).
  * CSP+LDA uses MOABB's paradigm output directly, which is what the
    calibration check compares against.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score

from refshift.datasets import (
    classes_for,
    moabb_dataset,
    moabb_paradigm_channels,
    split_strategy_for,
    subject_list,
)
from refshift.models import SUPPORTED_DL_MODELS, make_csp_lda_pipeline, make_dl_model
from refshift.preprocess import load_windows, split_train_test
from refshift.references import (
    FAMILIES,
    GRAPH_MODES,
    apply_reference,
    apply_reference_then_ea,
    build_graph,
    canonical_mode_tuple,
    reference_modes_for_dataset,
)


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------

def _tqdm(iterable, **kwargs):
    try:
        from tqdm.auto import tqdm
        return tqdm(iterable, **kwargs)
    except ImportError:
        return iterable


def _free_cuda():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _resolve_modes(dataset_id: str, reference_modes: Optional[Sequence[str]]) -> tuple:
    """Canonicalise the requested modes, defaulting to the dataset-safe set."""
    if reference_modes is None:
        return reference_modes_for_dataset(dataset_id)
    return canonical_mode_tuple(reference_modes)


def _build_graph_if_needed(modes, ch_names, *, montage, k_small, k_large_skip,
                           k_large_use, progress):
    """Build a DatasetGraph iff any mode needs one; include REST only if used."""
    if not any(m in GRAPH_MODES for m in modes):
        return None
    graph = build_graph(
        ch_names, k_small=k_small, k_large_skip=k_large_skip,
        k_large_use=k_large_use, montage=montage, include_rest=("rest" in modes),
    )
    if progress:
        rest_msg = (
            f", REST cond={graph.rest_cond:.2e}" if graph.rest_cond is not None else ""
        )
        print(f"graph: C={len(graph.ch_names)}, cz_idx={graph.cz_idx}{rest_msg}")
    return graph


def _score_row(subject, seed, train_ref, test_ref, y_te, y_pred, n_train, n_test):
    return {
        "subject": subject, "seed": seed,
        "train_ref": train_ref, "test_ref": test_ref,
        "accuracy": float(accuracy_score(y_te, y_pred)),
        "kappa": float(cohen_kappa_score(y_te, y_pred)),
        "n_train": int(n_train), "n_test": int(n_test),
    }


def _moabb_split_csp(X, y, metadata, dataset_id):
    """Train/test split for the CSP path, mirroring the dataset's strategy.

    MOABB's metadata uses 'session' / 'run' columns just like the braindecode
    path, so the same first-session / first-run logic applies.
    """
    strategy = split_strategy_for(dataset_id)
    if strategy == "session":
        sessions = sorted(metadata["session"].unique())
        train_mask = (metadata["session"] == sessions[0]).to_numpy()
    else:  # run
        runs = sorted(metadata["run"].unique())
        train_mask = (metadata["run"] == runs[0]).to_numpy()
    return X[train_mask], y[train_mask], X[~train_mask], y[~train_mask]


def _load_csp_subject(dataset_id, subject):
    """Load one subject via MOABB's paradigm: (X, y_int, metadata)."""
    dataset, paradigm = moabb_dataset(dataset_id)
    X, y_raw, metadata = paradigm.get_data(dataset=dataset, subjects=[subject])
    to_int = {name: i for i, name in enumerate(classes_for(dataset_id))}
    y = np.asarray([to_int[v] for v in y_raw], dtype=np.int64)
    return X.astype(np.float32, copy=False), y, metadata


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
    ea_eps: float = 1e-12,
    n_filters: int = 6,
    montage: str = "standard_1005",
    laplacian_k: int = 4,
    k_large_skip: int = 4,
    k_large_use: int = 4,
    progress: bool = True,
    dl_max_epochs: int = 200,
    dl_batch_size: int = 32,
    dl_lr: Optional[float] = None,
    dl_device: Optional[str] = None,
    dl_verbose: int = 0,
    cache_dir: Optional[str] = None,
) -> pd.DataFrame:
    """N x N train-reference by test-reference matrix on one dataset.

    ``model`` is 'csp_lda', 'shallow', 'eegnet', or 'atcnet'. With
    apply_ea=True each split is Euclidean-aligned after referencing (both
    pipelines). Returns long-form rows: subject, seed, train_ref, test_ref,
    accuracy, kappa, n_train, n_test.
    """
    model_lc = model.lower()
    is_dl = model_lc != "csp_lda"
    if is_dl and model_lc not in SUPPORTED_DL_MODELS:
        raise ValueError(f"Unknown model {model!r}; use 'csp_lda' or {SUPPORTED_DL_MODELS}")

    modes = _resolve_modes(dataset_id, reference_modes)
    subjects = list(subjects) if subjects is not None else subject_list(dataset_id)
    seeds = list(seeds)
    rows: List[dict] = []

    if is_dl:
        _, _, _, _, ch0 = load_windows(dataset_id, subjects[0], cache_dir=cache_dir)
        graph = _build_graph_if_needed(
            modes, ch0, montage=montage, k_small=laplacian_k,
            k_large_skip=k_large_skip, k_large_use=k_large_use, progress=progress,
        )
        jobs = [(s, k) for s in subjects for k in seeds]
        last_subject = None
        X = y = metadata = sfreq = None
        for subject, seed in _tqdm(jobs, desc=f"[{dataset_id}] {model_lc} mismatch",
                                   disable=not progress):
            if subject != last_subject:
                X, y, metadata, sfreq, ch = load_windows(
                    dataset_id, subject, cache_dir=cache_dir,
                )
                if graph is not None:
                    assert ch == graph.ch_names, "channel order mismatch vs graph"
                last_subject = subject
            X_tr, y_tr, X_te, y_te = split_train_test(X, y, metadata, dataset_id)
            n_classes = int(max(y_tr.max(), y_te.max())) + 1
            X_te_by_ref = {
                m: apply_reference_then_ea(X_te, m, graph=graph, zscore=True,
                                           apply_ea=apply_ea, ea_eps=ea_eps)
                for m in modes
            }
            for train_ref in modes:
                X_tr_ref = apply_reference_then_ea(
                    X_tr, train_ref, graph=graph, zscore=True,
                    apply_ea=apply_ea, ea_eps=ea_eps,
                )
                pipe = make_dl_model(
                    model=model_lc, n_channels=X_tr_ref.shape[1],
                    n_classes=n_classes, n_times=X_tr_ref.shape[2],
                    sfreq=float(sfreq), seed=int(seed),
                    max_epochs=dl_max_epochs, batch_size=dl_batch_size,
                    lr=dl_lr, device=dl_device, verbose=dl_verbose,
                )
                pipe.fit(X_tr_ref, y_tr)
                for test_ref in modes:
                    y_pred = pipe.predict(X_te_by_ref[test_ref])
                    rows.append(_score_row(subject, seed, train_ref, test_ref,
                                           y_te, y_pred, len(y_tr), len(y_te)))
                del pipe
                _free_cuda()
        return pd.DataFrame(rows)

    # CSP+LDA path (MOABB paradigm data, no z-score)
    graph = _build_graph_if_needed(
        modes, moabb_paradigm_channels(dataset_id, subjects[0]), montage=montage,
        k_small=laplacian_k, k_large_skip=k_large_skip, k_large_use=k_large_use,
        progress=progress,
    )
    jobs = [(s, k) for s in subjects for k in seeds]
    last_subject = None
    X = y = metadata = None
    for subject, seed in _tqdm(jobs, desc=f"[{dataset_id}] csp_lda mismatch",
                               disable=not progress):
        if subject != last_subject:
            X, y, metadata = _load_csp_subject(dataset_id, subject)
            last_subject = subject
        X_tr, y_tr, X_te, y_te = _moabb_split_csp(X, y, metadata, dataset_id)
        X_te_by_ref = {
            m: apply_reference_then_ea(X_te, m, graph=graph,
                                       apply_ea=apply_ea, ea_eps=ea_eps)
            for m in modes
        }
        for train_ref in modes:
            X_tr_ref = apply_reference_then_ea(
                X_tr, train_ref, graph=graph, apply_ea=apply_ea, ea_eps=ea_eps,
            )
            pipe = make_csp_lda_pipeline(reference_mode=None, n_filters=n_filters)
            pipe.fit(X_tr_ref, y_tr)
            for test_ref in modes:
                y_pred = pipe.predict(X_te_by_ref[test_ref])
                rows.append(_score_row(subject, seed, train_ref, test_ref,
                                       y_te, y_pred, len(y_tr), len(y_te)))
    return pd.DataFrame(rows)


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
    montage: str = "standard_1005",
    laplacian_k: int = 4,
    k_large_skip: int = 4,
    k_large_use: int = 4,
    progress: bool = True,
    dl_max_epochs: int = 200,
    dl_batch_size: int = 32,
    dl_lr: Optional[float] = None,
    dl_device: Optional[str] = None,
    dl_verbose: int = 0,
    cache_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Train one deep net with per-sample reference jitter; test on every ref.

    condition='full' jitters over all references; condition='loro' jitters over
    all-but-``holdout_ref``. Deep nets only. Returns long-form rows: subject,
    seed, condition, holdout_ref, train_modes, test_ref, accuracy, kappa,
    n_train, n_test.
    """
    from refshift.jitter import make_random_reference_transform

    model_lc = model.lower()
    if model_lc not in SUPPORTED_DL_MODELS:
        raise ValueError(f"jitter needs a deep model; got {model!r}")
    cond = condition.lower()
    if cond not in ("full", "loro"):
        raise ValueError(f"Unknown condition {condition!r}; use 'full' or 'loro'")

    universe = _resolve_modes(dataset_id, reference_modes)
    if cond == "loro" and holdout_ref not in universe:
        raise ValueError(f"holdout_ref={holdout_ref!r} not in {universe}")
    test_modes = (
        canonical_mode_tuple(test_reference_modes)
        if test_reference_modes is not None else universe
    )

    if cond == "full":
        train_modes = universe
        holdout_label = ""
    else:
        train_modes = tuple(m for m in universe if m != holdout_ref)
        holdout_label = holdout_ref
    train_modes_str = ",".join(train_modes)

    subjects = list(subjects) if subjects is not None else subject_list(dataset_id)
    seeds = list(seeds)

    all_modes = canonical_mode_tuple(set(train_modes) | set(test_modes))
    _, _, _, _, ch0 = load_windows(dataset_id, subjects[0], cache_dir=cache_dir)
    graph = _build_graph_if_needed(
        all_modes, ch0, montage=montage, k_small=laplacian_k,
        k_large_skip=k_large_skip, k_large_use=k_large_use, progress=progress,
    )

    rows: List[dict] = []
    jobs = [(s, k) for s in subjects for k in seeds]
    last_subject = None
    X = y = metadata = sfreq = None
    for subject, seed in _tqdm(jobs, desc=f"[{dataset_id}] {model_lc} jitter-{cond}",
                               disable=not progress):
        if subject != last_subject:
            X, y, metadata, sfreq, ch = load_windows(
                dataset_id, subject, cache_dir=cache_dir,
            )
            if graph is not None:
                assert ch == graph.ch_names, "channel order mismatch vs graph"
            last_subject = subject
        X_tr, y_tr, X_te, y_te = split_train_test(X, y, metadata, dataset_id)
        n_classes = int(max(y_tr.max(), y_te.max())) + 1
        X_te_by_ref = {
            m: apply_reference_then_ea(X_te, m, graph=graph, zscore=True)
            for m in test_modes
        }

        rng_seed = int(1_000_003 * int(seed) + 7919 * int(subject))
        ref_transform = make_random_reference_transform(
            allowed_modes=train_modes, graph=graph,
            probability=1.0, random_state=rng_seed,
        )
        pipe = make_dl_model(
            model=model_lc, n_channels=X_tr.shape[1], n_classes=n_classes,
            n_times=X_tr.shape[2], sfreq=float(sfreq), seed=int(seed),
            max_epochs=dl_max_epochs, batch_size=dl_batch_size,
            lr=dl_lr, device=dl_device, verbose=dl_verbose,
            transforms=[ref_transform],
        )
        # Training data stays native; the transform re-references and z-scores
        # each sample at batch time, so we must NOT pre-reference here.
        pipe.fit(X_tr, y_tr)

        for test_ref in test_modes:
            y_pred = pipe.predict(X_te_by_ref[test_ref])
            rows.append({
                "subject": subject, "seed": seed, "condition": cond,
                "holdout_ref": holdout_label, "train_modes": train_modes_str,
                "test_ref": test_ref,
                "accuracy": float(accuracy_score(y_te, y_pred)),
                "kappa": float(cohen_kappa_score(y_te, y_pred)),
                "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
            })
        del pipe
        _free_cuda()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# run_loro_matrix  (leave-one-reference-out)
# ---------------------------------------------------------------------------

def run_loro_matrix(
    dataset_id: str,
    *,
    model: str,
    holdout_modes: Optional[Sequence[str]] = None,
    reference_modes: Optional[Sequence[str]] = None,
    seeds: Sequence[int] = (0,),
    subjects: Optional[List[int]] = None,
    progress: bool = True,
    **jitter_kwargs,
) -> pd.DataFrame:
    """Sweep run_mismatch_jitter(condition='loro') over each held-out reference."""
    universe = _resolve_modes(dataset_id, reference_modes)
    holdouts = (
        canonical_mode_tuple(holdout_modes)
        if holdout_modes is not None else universe
    )
    frames: List[pd.DataFrame] = []
    for h in holdouts:
        if h not in universe:
            raise ValueError(f"holdout {h!r} not in universe={universe}")
        frames.append(run_mismatch_jitter(
            dataset_id, model=model, condition="loro", holdout_ref=h,
            reference_modes=universe, seeds=seeds, subjects=subjects,
            progress=progress, **jitter_kwargs,
        ))
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# run_lofo_matrix  (leave-one-family-out)
# ---------------------------------------------------------------------------

def run_lofo_matrix(
    dataset_id: str,
    *,
    model: str,
    families: dict = None,
    reference_modes: Optional[Sequence[str]] = None,
    holdout_families: Optional[Sequence[str]] = None,
    seeds: Sequence[int] = (0,),
    subjects: Optional[List[int]] = None,
    progress: bool = True,
    **jitter_kwargs,
) -> pd.DataFrame:
    """Hold out a whole family of references at a time, test on every reference.

    ``families`` is a {name: [modes]} dict; defaults to references.FAMILIES. For
    each held-out family, train full jitter over the union of the others, test
    on the whole universe. Rows are tagged with holdout_family and test_family.
    """
    if families is None:
        families = FAMILIES

    fam_modes: dict = {}
    mode_to_family: dict = {}
    for fam, members in families.items():
        canon = canonical_mode_tuple(members) if members else ()
        fam_modes[fam] = canon
        for m in canon:
            if m in mode_to_family:
                raise ValueError(
                    f"mode {m!r} is in families {mode_to_family[m]!r} and {fam!r}; "
                    "a mode may belong to at most one family."
                )
            mode_to_family[m] = fam

    from refshift.references import REFERENCE_MODES
    family_union = tuple(m for m in REFERENCE_MODES if m in mode_to_family)
    if not family_union:
        raise ValueError("families is empty or all families have no modes.")

    if reference_modes is None:
        universe = tuple(m for m in family_union
                         if m in reference_modes_for_dataset(dataset_id))
    else:
        universe = canonical_mode_tuple(reference_modes)
        missing = set(family_union) - set(universe)
        if missing:
            raise ValueError(
                f"reference_modes must contain every family mode; missing {sorted(missing)}"
            )

    sweep = list(holdout_families) if holdout_families is not None else list(fam_modes)
    for fam in sweep:
        if fam not in fam_modes:
            raise ValueError(f"holdout_family {fam!r} not in families {list(fam_modes)}")

    frames: List[pd.DataFrame] = []
    for fam in sweep:
        held = tuple(m for m in fam_modes[fam] if m in universe)
        if not held:
            continue
        train_modes = tuple(m for m in universe if m not in held)
        if not train_modes:
            raise ValueError(f"holding out {fam!r} leaves no training modes.")
        df_fam = run_mismatch_jitter(
            dataset_id, model=model, condition="full",
            reference_modes=train_modes, test_reference_modes=universe,
            seeds=seeds, subjects=subjects, progress=progress, **jitter_kwargs,
        )
        df_fam["condition"] = "lofo"
        df_fam["holdout_ref"] = ",".join(held)
        df_fam["holdout_family"] = fam
        df_fam["test_family"] = df_fam["test_ref"].map(
            lambda m: mode_to_family.get(m, "unassigned")
        )
        frames.append(df_fam)
    if not frames:
        raise ValueError("No non-empty families to sweep.")
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# calibrate_csp_lda
# ---------------------------------------------------------------------------

IV2A_CSP_LDA_TARGET = 65.99
IV2A_CSP_LDA_TOL = 2.0
IDENTITY_TOL = 0.5


def calibrate_csp_lda(
    dataset_id: str = "iv2a",
    *,
    subjects: Optional[List[int]] = None,
    random_state: int = 42,
    verbose: bool = True,
):
    """Confirm bare CSP+LDA matches the MOABB baseline and that a no-op
    reference transformer changes nothing.

    For IV-2a the target is the published 65.99%. For other datasets only the
    identity-equivalence check is enforced. Returns (results, summary, passed).
    """
    from moabb.evaluations import WithinSessionEvaluation

    dataset, paradigm = moabb_dataset(dataset_id)
    if subjects is not None:
        dataset.subject_list = list(subjects)

    pipelines = {
        "CSP+LDA (bare)": make_csp_lda_pipeline(reference_mode=None),
        "CSP+LDA (native)": make_csp_lda_pipeline(reference_mode="native"),
    }
    evaluation = WithinSessionEvaluation(
        paradigm=paradigm, datasets=[dataset],
        overwrite=True, random_state=random_state,
    )
    results = evaluation.process(pipelines)

    summary = (
        results.groupby("pipeline")["score"]
               .agg(["mean", "std", "count"])
               .assign(mean=lambda d: 100 * d["mean"], std=lambda d: 100 * d["std"])
               .round(2)
    )
    bare = 100 * results[results["pipeline"] == "CSP+LDA (bare)"]["score"].mean()
    ident = 100 * results[results["pipeline"] == "CSP+LDA (native)"]["score"].mean()
    moabb_ok = (
        abs(bare - IV2A_CSP_LDA_TARGET) <= IV2A_CSP_LDA_TOL
        if dataset_id.lower() == "iv2a" else True
    )
    identity_ok = abs(ident - bare) <= IDENTITY_TOL
    passed = bool(moabb_ok and identity_ok)

    if verbose:
        print("\nPer-pipeline mean +/- std across subjects x sessions:")
        for name, row in summary.iterrows():
            print(f"  {name:18s} {row['mean']:5.2f} +/- {row['std']:5.2f}")
        if dataset_id.lower() == "iv2a":
            print(f"\nMOABB target {IV2A_CSP_LDA_TARGET}%: got {bare:.2f}% -> "
                  f"{'PASS' if moabb_ok else 'FAIL'}")
        print(f"Identity within {IDENTITY_TOL}%: delta={ident - bare:+.2f}% -> "
              f"{'PASS' if identity_ok else 'FAIL'}")
    return results, summary, passed
