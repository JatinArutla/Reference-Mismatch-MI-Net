"""Bandpass-mismatch control: train under one bandpass, test under another.

The mismatch matrix's off-diagonal collapse needs to be shown to be specific
to reference operators, not generic preprocessing brittleness. This runner
trains under train_band on a fixed reference (default native) and evaluates
the same model on each of test_bands. Expected: bandpass mismatch yields
a much smaller drop (<5 pts on IV-2a Shallow) than reference mismatch (20+).

The diagonal (matched-band test) is added automatically as a control.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score

from refshift.experiments._dl_runner import (
    free_cuda,
    iter_per_subject_dl_jobs,
    setup_dl_run,
)
from refshift.experiments._split import split_train_test
from refshift.reference import REFERENCE_MODES, apply_reference


def run_bandpass_mismatch(
    dataset_id: str,
    *,
    model: str = "shallow",
    train_band: Tuple[float, float] = (8.0, 32.0),
    test_bands: Tuple[Tuple[float, float], ...] = ((6.0, 32.0), (8.0, 30.0)),
    reference_mode: str = "native",
    subjects: Optional[List[int]] = None,
    seeds: List[int] = (0,),
    classes: Optional[Tuple[str, ...]] = None,
    split_strategy: str = "auto",
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
    dl_resample: float = 250.0,
    dl_trial_start_offset_s: float = 0.0,
    dl_trial_stop_offset_s: float = 0.0,
    dl_cache_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Train at train_band, test at each band in test_bands plus train_band itself.

    laplacian_k and montage are only consulted if reference_mode is graph-requiring
    (laplacian, rest, cz_ref). For the default reference_mode='native' they are unused.
    """
    from refshift.data import load_dl_data
    from refshift.model import SUPPORTED_DL_MODELS, make_dl_model

    model_lc = model.lower()
    if model_lc not in SUPPORTED_DL_MODELS:
        raise ValueError(
            f"run_bandpass_mismatch is DL-only. Got {model!r}; supported: {SUPPORTED_DL_MODELS}"
        )
    from refshift.reference import _resolve_alias
    reference_mode = _resolve_alias(reference_mode)
    if reference_mode not in REFERENCE_MODES:
        raise ValueError(f"reference_mode={reference_mode!r} not in REFERENCE_MODES")

    # Graph only needs to support the single fixed reference.
    ctx = setup_dl_run(
        dataset_id, subjects=subjects, seeds=seeds,
        reference_modes_for_graph=(reference_mode,),
        laplacian_k=laplacian_k,
        k_large_skip=k_large_skip, k_large_use=k_large_use,
        montage=montage, progress=progress,
    )

    all_test_bands = (train_band,) + tuple(b for b in test_bands if b != train_band)
    bands_str = lambda b: f"{b[0]:.1f}-{b[1]:.1f}"

    rows: List[dict] = []
    for subject, seed, X_tr_split, y_tr, _Xdrop, y_drop, sfreq in iter_per_subject_dl_jobs(
        ctx, split_strategy=split_strategy,
        desc=f"[{ctx.dataset_code}] {model_lc} bandpass",
        progress=progress,
        dl_resample=dl_resample,
        dl_l_freq=train_band[0], dl_h_freq=train_band[1],
        dl_trial_start_offset_s=dl_trial_start_offset_s,
        dl_trial_stop_offset_s=dl_trial_stop_offset_s,
        dl_cache_dir=dl_cache_dir,
        classes=classes,
    ):
        X_tr_ref = apply_reference(X_tr_split, reference_mode, graph=ctx.graph)
        n_classes = int(max(int(y_tr.max()), int(y_drop.max()))) + 1

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

        # For each test band, re-preprocess with that band and evaluate. The
        # iter helper can't drive this loop because the bandpass changes per
        # iteration (different cache key, different load).
        for tb in all_test_bands:
            X_te_raw, y_te_raw, meta_te, _, _ = load_dl_data(
                dataset_id, subject,
                resample=dl_resample,
                l_freq=tb[0], h_freq=tb[1],
                trial_start_offset_s=dl_trial_start_offset_s,
                trial_stop_offset_s=dl_trial_stop_offset_s,
                cache_dir=dl_cache_dir,
                classes=classes,
            )
            _, _, X_te_split, y_te = split_train_test(
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
        free_cuda()

    return pd.DataFrame(rows)
