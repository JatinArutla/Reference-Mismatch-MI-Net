"""EMS-control diagonal: each (subject, ref) cell preprocesses with the
reference applied *before* EMS (via load_dl_data(pre_ems_reference=ref)),
trains a fresh model, and tests on the same reference.

The standard run_mismatch pipeline applies references to the EMS-standardised
windowed tensor. EMS is per-channel and adaptive, so it does not commute with
channel-mixing reference operators. This function provides the corresponding
control: a per-reference diagonal that should match run_mismatch's diagonal
within seed noise if the EMS-after-reference order is not driving cluster
structure.

In v0.14 the pre_ems_reference is applied to the bandpass-filtered raw
(was: broadband raw before filter, which mattered for the median operator).

Default reference_modes is auto-resolved from dataset_id, so cz_ref is
dropped on Schirrmeister automatically.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score

from refshift.experiments._datasets import (
    get_eeg_channel_names,
    resolve_dataset,
)
from refshift.experiments._dl_runner import free_cuda
from refshift.experiments._split import split_train_test
from refshift.reference import (
    REFERENCE_MODES,
    _GRAPH_MODES,
    build_graph,
    canonical_mode_tuple,
    reference_modes_for_dataset,
    validate_reference_modes,
)


def run_pre_ems_diagonal(
    dataset_id: str,
    *,
    model: str = "shallow",
    subjects: Optional[List[int]] = None,
    seeds: Iterable[int] = (0,),
    reference_modes: Optional[Iterable[str]] = None,
    classes: Optional[Iterable[str]] = None,
    split_strategy: str = "auto",
    normalization: str = "ems",
    laplacian_k: int = 4,
    k_large_skip: int = 4,
    k_large_use: int = 4,
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
    dl_cache_dir: Optional[str] = None,
) -> pd.DataFrame:
    """One row per (subject, seed, reference) with same-reference train/test,
    reference applied to filtered raw before the normalization step.

    laplacian_k, k_large_skip, k_large_use, and montage are threaded into
    load_dl_data's pre-normalization graph build and are part of the cache key,
    so non-default values produce distinct cache entries.

    normalization in {"zscore", "ems", "none"}; default "ems". The reference
    is applied to the filtered raw BEFORE this standardisation step. The default
    is "ems" (not "zscore" like the main runners) because this is specifically
    the EMS-ordering control: EMS is adaptive, so reference-before-EMS differs
    from reference-after-EMS in a time-varying way that is the whole point of
    the ablation. To probe ordering under a different normalization, set this
    AND run_mismatch with the SAME value -- an ordering comparison is only valid
    when both arms share a normalization.
    """
    from refshift.data import load_dl_data
    from refshift.model import SUPPORTED_DL_MODELS, make_dl_model

    model_lc = model.lower()
    if model_lc == "csp_lda":
        raise ValueError(
            "run_pre_ems_diagonal is DL-only. CSP+LDA does not use EMS, so "
            "the EMS-control question doesn't apply."
        )
    if model_lc not in SUPPORTED_DL_MODELS:
        raise ValueError(f"Unknown DL model {model!r}; expected one of {SUPPORTED_DL_MODELS}")

    from refshift.data import NORMALIZATIONS
    if normalization not in NORMALIZATIONS:
        raise ValueError(f"normalization={normalization!r} not in {NORMALIZATIONS}")

    if reference_modes is None:
        modes = reference_modes_for_dataset(dataset_id)
    else:
        modes = canonical_mode_tuple(reference_modes)

    dataset, paradigm = resolve_dataset(dataset_id)
    if subjects is None:
        subjects = list(dataset.subject_list)
    seeds = list(seeds)

    # Build a probe graph for early validation. Using the first subject's
    # channel names is fine because `paradigm.channels` (when set) is the
    # same across subjects, and the canonical EEG montage is fixed.
    needs_graph = any(m in _GRAPH_MODES for m in modes)
    probe_graph = None
    if needs_graph:
        ch_names = get_eeg_channel_names(
            dataset, subject=subjects[0], paradigm=paradigm,
        )
        probe_graph = build_graph(
            ch_names, k_small=laplacian_k,
            k_large_skip=k_large_skip, k_large_use=k_large_use,
            montage=montage,
            include_rest=("rest" in modes),
            include_csd=("csd" in modes),
        )
    validate_reference_modes(modes, probe_graph, dataset_id=dataset_id)

    try:
        from tqdm.auto import tqdm as _tqdm
    except ImportError:
        def _tqdm(it, **kwargs):
            return it

    jobs = [(s, k, r) for s in subjects for k in seeds for r in modes]
    iterator = _tqdm(
        jobs, desc=f"[{dataset.code}] {model_lc} pre-EMS diagonal",
        disable=not progress, leave=True,
    )

    rows: List[dict] = []
    for subject, seed, ref in iterator:
        # Each (subject, ref) gets its own preprocess pass with the reference
        # applied to the filtered raw before EMS. The cache key in load_dl_data
        # includes pre_ems_reference, pre_ems_laplacian_k, pre_ems_montage, so
        # repeated calls with the same triple reuse the cache.
        X, y_int, metadata, sfreq, _ = load_dl_data(
            dataset_id, subject,
            resample=dl_resample,
            l_freq=dl_l_freq, h_freq=dl_h_freq,
            normalization=normalization,
            trial_start_offset_s=dl_trial_start_offset_s,
            trial_stop_offset_s=dl_trial_stop_offset_s,
            cache_dir=dl_cache_dir,
            pre_ems_reference=ref,
            pre_ems_laplacian_k=laplacian_k,
            pre_ems_k_large_skip=k_large_skip,
            pre_ems_k_large_use=k_large_use,
            pre_ems_montage=montage,
            classes=classes,
        )

        X_tr, y_tr, X_te, y_te = split_train_test(
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
            "dataset":   dataset.code,
            "subject":   int(subject),
            "seed":      int(seed),
            "reference": ref,
            "accuracy":  float(accuracy_score(y_te, y_pred)),
            "kappa":     float(cohen_kappa_score(y_te, y_pred)),
            "n_train":   int(len(y_tr)),
            "n_test":    int(len(y_te)),
        })

        del net
        free_cuda()

    return pd.DataFrame(rows)
