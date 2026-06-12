"""Cross-subject (leave-one-subject-out) reference mismatch, CSP+LDA.

The within-subject EA sweep (v0.19) showed EA's R_bar from 5 target trials/class
already closes the reference-mismatch gap to ~2 points. The remaining open
question is the zero-target-calibration regime: when the target subject's R_bar
cannot be estimated from target data and must be borrowed, does the reference
mismatch reappear?

This runner trains CSP+LDA on a POOL of source subjects (everyone except the
held-out target) and tests on the target subject, under three EA regimes:

  ea_regime="within"     : R_bar from the target subject's OWN test block
                           (the v0.19 floor; sanity anchor, uses target data).
  ea_regime="source"     : R_bar from the pooled SOURCE subjects' (referenced)
                           training data; ZERO target calibration. The true
                           deployment case.
  ea_regime="target_k"   : R_bar from k target trials/class carved out of the
                           target test block (excluded from scoring); the
                           realistic small-calibration middle ground.

The headline metric is NOT absolute accuracy (cross-subject CSP+LDA is just
harder, so the diagonal drops regardless of references). It is the TRANSFER GAP:
matched-reference diagonal minus off-diagonal. If the gap stays ~2 across
regimes, references are solved even at zero target calibration. If the gap
balloons under "source", reference mismatch survives cross-subject transfer and
there is a real problem for a reference-robust method to attack.

Train side: the source pool is referenced under train_ref and EA-aligned with
the SOURCE pool's own R_bar (a model trained on heterogeneous source subjects
needs its own alignment). Only the TEST-side R_bar varies by ea_regime.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score

from refshift.experiments._datasets import (
    build_cache_config,
    get_eeg_channel_names,
    resolve_dataset,
)
from refshift.experiments._split import encode_labels, split_train_test
from refshift.model import make_csp_lda_pipeline
from refshift.reference import (
    _GRAPH_MODES,
    _ea_apply,
    _ea_fit,
    apply_reference,
    build_graph,
    canonical_mode_tuple,
    reference_modes_for_dataset,
    stratified_calibration_index,
    validate_reference_modes,
)

EA_REGIMES = ("within", "source", "target_k")


def run_cross_subject_mismatch(
    dataset_id: str,
    *,
    ea_regime: str = "source",
    subjects: Optional[List[int]] = None,
    seeds: Sequence[int] = (0,),
    reference_modes: Optional[Sequence[str]] = None,
    classes: Optional[Sequence[str]] = None,
    split_strategy: str = "auto",
    ea_eps: float = 1e-12,
    ea_target_k: int = 5,
    n_filters: int = 6,
    csp_trace_normalize: bool = False,
    laplacian_k: int = 4,
    k_large_skip: int = 4,
    k_large_use: int = 4,
    montage: str = "standard_1005",
    cache: bool = True,
    progress: bool = True,
) -> pd.DataFrame:
    """Leave-one-subject-out reference mismatch with CSP+LDA.

    For each held-out target subject t:
      - source pool = all other subjects' TRAIN splits, concatenated.
      - for each train_ref: reference the pool, EA-align it with the pool's own
        R_bar, fit CSP+LDA.
      - for each test_ref: reference the target's TEST split, EA-align it with
        the regime's R_bar, score.

    ea_regime in {"within","source","target_k"} (see module docstring).
    Returns long-form rows with columns: dataset, target_subject, seed,
    ea_regime, train_ref, test_ref, accuracy, kappa, n_source_train, n_target_test.

    Note: only CSP+LDA. The source-pool R_bar for the train side is computed
    once per (train_ref) since the pool is fixed for a given target; the
    test-side R_bar is recomputed per the regime.
    """
    if ea_regime not in EA_REGIMES:
        raise ValueError(f"ea_regime={ea_regime!r} not in {EA_REGIMES}")

    if reference_modes is None:
        modes = reference_modes_for_dataset(dataset_id)
    else:
        modes = canonical_mode_tuple(reference_modes)

    dataset, paradigm = resolve_dataset(dataset_id, classes=classes)
    if subjects is None:
        subjects = list(dataset.subject_list)
    subjects = list(subjects)
    seeds = list(seeds)

    if len(subjects) < 2:
        raise ValueError(
            f"cross-subject needs >=2 subjects; got {len(subjects)} for {dataset_id}"
        )

    # Build the operator graph once from the first subject's channels. Within a
    # single dataset the montage is shared across subjects (asserted per-subject
    # on load below), so one graph is valid for all. Cross-DATASET would need a
    # per-dataset graph and is intentionally out of scope here.
    needs_graph = any(m in _GRAPH_MODES for m in modes)
    needs_rest = "rest" in modes
    needs_csd = "csd" in modes
    graph = None
    if needs_graph:
        ch_names = get_eeg_channel_names(dataset, subject=subjects[0], paradigm=paradigm)
        graph = build_graph(
            ch_names, k_small=laplacian_k,
            k_large_skip=k_large_skip, k_large_use=k_large_use,
            montage=montage, include_rest=needs_rest, include_csd=needs_csd,
        )
        if progress:
            print(f"[{dataset.code}] graph: C={len(graph.ch_names)}")
    validate_reference_modes(modes, graph, dataset_id=dataset_id)

    cache_kwargs = {"cache_config": build_cache_config()} if cache else {}

    # Pre-load every subject's split once; reuse across the LOSO loop and seeds.
    # Keyed by (subject, seed) because the within-session stratified split is
    # seed-dependent. For session/run split datasets the seed is inert.
    def _load_subject(subject):
        X, y_raw, metadata = paradigm.get_data(
            dataset=dataset, subjects=[subject], **cache_kwargs,
        )
        y_int, _ = encode_labels(y_raw)
        if graph is not None:
            ch_subj = get_eeg_channel_names(dataset, subject=subject, paradigm=paradigm)
            assert list(ch_subj) == graph.ch_names, (
                f"Channel order mismatch subj {subject}: "
                f"{ch_subj[:5]} vs {graph.ch_names[:5]}"
            )
            assert X.shape[1] == len(graph.ch_names)
        return X, y_int, metadata

    try:
        from tqdm.auto import tqdm as _tqdm
    except ImportError:
        def _tqdm(it, **kwargs):
            return it

    rows: List[dict] = []
    raw_cache: dict = {}

    jobs = [(t, s) for t in subjects for s in seeds]
    for target, seed in _tqdm(
        jobs, desc=f"[{dataset.code}] cross-subject {ea_regime}",
        disable=not progress, leave=True,
    ):
        # Assemble the source training pool: every other subject's TRAIN split.
        src_X_list, src_y_list = [], []
        for s in subjects:
            if s == target:
                continue
            if s not in raw_cache:
                raw_cache[s] = _load_subject(s)
            Xs, ys, ms = raw_cache[s]
            X_tr, y_tr, _, _ = split_train_test(
                Xs, ys, ms, strategy=split_strategy, seed=seed,
                dataset_id=dataset_id,
            )
            src_X_list.append(X_tr)
            src_y_list.append(y_tr)
        X_src = np.concatenate(src_X_list, axis=0)
        y_src = np.concatenate(src_y_list, axis=0)

        # Target test split.
        if target not in raw_cache:
            raw_cache[target] = _load_subject(target)
        Xt, yt, mt = raw_cache[target]
        _, _, X_te, y_te = split_train_test(
            Xt, yt, mt, strategy=split_strategy, seed=seed, dataset_id=dataset_id,
        )

        # Resolve the test-side scoring subset + the target-derived R_bar source.
        if ea_regime == "target_k":
            calib_idx = stratified_calibration_index(y_te, int(ea_target_k), seed=seed)
            mask = np.zeros(len(y_te), dtype=bool)
            mask[calib_idx] = True
            score_idx = np.flatnonzero(~mask)
        else:
            calib_idx = None
            score_idx = np.arange(len(y_te))
        y_te_scored = y_te[score_idx]

        # Pre-reference + EA-align the target test variants per the regime.
        X_te_by_ref = {}
        for m in modes:
            Xr = apply_reference(X_te, m, graph=graph)
            if ea_regime == "within":
                whit = _ea_fit(Xr[score_idx], eps=ea_eps)        # target's own
            elif ea_regime == "target_k":
                whit = _ea_fit(Xr[calib_idx], eps=ea_eps)        # k target trials
            else:  # "source": R_bar from the source pool under the SAME ref
                Xsrc_ref = apply_reference(X_src, m, graph=graph)
                whit = _ea_fit(Xsrc_ref, eps=ea_eps)
            X_te_by_ref[m] = _ea_apply(Xr[score_idx], whit)

        # Train one model per train_ref on the source pool (source-aligned).
        for train_ref in modes:
            X_src_ref = apply_reference(X_src, train_ref, graph=graph)
            src_whit = _ea_fit(X_src_ref, eps=ea_eps)
            X_src_aligned = _ea_apply(X_src_ref, src_whit)
            pipe = make_csp_lda_pipeline(
                reference_mode=None, n_filters=n_filters,
                trace_normalize=csp_trace_normalize,
            )
            pipe.fit(X_src_aligned, y_src)
            for test_ref in modes:
                y_pred = pipe.predict(X_te_by_ref[test_ref])
                rows.append({
                    "dataset": dataset.code,
                    "target_subject": target,
                    "seed": seed,
                    "ea_regime": ea_regime,
                    "train_ref": train_ref,
                    "test_ref": test_ref,
                    "accuracy": float(accuracy_score(y_te_scored, y_pred)),
                    "kappa": float(cohen_kappa_score(y_te_scored, y_pred)),
                    "n_source_train": int(len(y_src)),
                    "n_target_test": int(len(y_te_scored)),
                })
            del pipe

    return pd.DataFrame(rows)
