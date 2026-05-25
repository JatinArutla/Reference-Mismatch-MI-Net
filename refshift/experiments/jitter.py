"""Per-sample reference jitter (Phase 2 intervention).

run_mismatch_jitter trains one model per (subject, seed) where each training
sample independently gets a reference drawn uniformly from the allowed set,
then evaluates on every test reference.

Two conditions:
    full: train_modes = reference_modes (the universe).
          Tests in-distribution generalisation.
    lofo: train_modes = reference_modes \\ {holdout_ref}; tests invariance
          to a previously-unseen reference. test_modes still includes the
          holdout (that's the whole point: scoring on what we never trained on).

run_lofo_matrix sweeps holdout_ref over every reference in `holdout_modes`
and concatenates the long-form output.

Default behaviour: reference_modes is auto-resolved from dataset_id via
reference_modes_for_dataset, which drops cz_ref for Schirrmeister2017.
Pass an explicit reference_modes tuple to override.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score

from refshift.experiments._dl_runner import (
    free_cuda,
    iter_per_subject_dl_jobs,
    setup_dl_run,
)
from refshift.reference import (
    REFERENCE_MODES,
    apply_reference,
    canonical_mode_tuple,
    reference_modes_for_dataset,
    validate_reference_modes,
)


def run_mismatch_jitter(
    dataset_id: str,
    *,
    model: str,
    condition: str = "full",
    holdout_ref: str = "cz_ref",
    subjects: Optional[List[int]] = None,
    seeds: List[int] = (0,),
    reference_modes: Optional[Sequence[str]] = None,
    test_reference_modes: Optional[Sequence[str]] = None,
    classes: Optional[tuple] = None,
    split_strategy: str = "auto",
    normalization: str = "zscore",
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
    """Train with per-sample reference jitter; evaluate on every test reference.

    Returns a long-form DataFrame with columns:
        dataset, subject, seed, condition, holdout_ref, train_modes,
        test_ref, accuracy, kappa, n_train, n_test.
    No train_ref column — each training sample sees a different reference.

    reference_modes / test_reference_modes accept any iterable (set, tuple,
    list, frozenset); both are canonicalised to REFERENCE_MODES order.

    normalization in {"zscore", "ems", "none"}; default "zscore". Selects the
    per-channel standardisation in load_dl_data (DL-only runner).
    """
    from refshift.jitter import make_random_reference_transform
    from refshift.model import SUPPORTED_DL_MODELS, make_dl_model

    model_lc = model.lower()
    if model_lc not in SUPPORTED_DL_MODELS:
        raise ValueError(
            f"run_mismatch_jitter requires a DL model. "
            f"Got {model!r}; supported: {SUPPORTED_DL_MODELS}"
        )
    cond = condition.lower()
    if cond not in ("full", "lofo"):
        raise ValueError(f"Unknown condition: {condition!r}. Use 'full' or 'lofo'")
    from refshift.data import NORMALIZATIONS
    if normalization not in NORMALIZATIONS:
        raise ValueError(f"normalization={normalization!r} not in {NORMALIZATIONS}")

    # Universe of references for this run. Auto-resolved per dataset by default.
    if reference_modes is None:
        universe = reference_modes_for_dataset(dataset_id)
    else:
        universe = canonical_mode_tuple(reference_modes)
    if cond == "lofo" and holdout_ref not in universe:
        raise ValueError(
            f"holdout_ref={holdout_ref!r} not in reference_modes universe={universe}. "
            "For Schirrmeister2017 this typically means cz_ref was already excluded."
        )

    # Test universe defaults to the same; LOFO keeps the holdout in test_modes
    # by design (that's the whole point of scoring transfer to an unseen ref).
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

    # Graph must cover both train-time sampler modes and test-time apply_reference modes.
    ctx = setup_dl_run(
        dataset_id, subjects=subjects, seeds=seeds,
        reference_modes_for_graph=tuple(set(train_modes) | set(test_modes)),
        laplacian_k=laplacian_k,
        k_large_skip=k_large_skip, k_large_use=k_large_use,
        montage=montage, progress=progress,
    )
    validate_reference_modes(train_modes, ctx.graph, dataset_id=dataset_id)
    validate_reference_modes(test_modes, ctx.graph, dataset_id=dataset_id)

    rows: List[dict] = []
    for subject, seed, X_tr, y_tr, X_te, y_te, sfreq in iter_per_subject_dl_jobs(
        ctx, split_strategy=split_strategy,
        desc=f"[{ctx.dataset_code}] {model_lc} jitter-{cond}",
        progress=progress,
        normalization=normalization,
        dl_resample=dl_resample,
        dl_l_freq=dl_l_freq, dl_h_freq=dl_h_freq,
        dl_trial_start_offset_s=dl_trial_start_offset_s,
        dl_trial_stop_offset_s=dl_trial_stop_offset_s,
        dl_cache_dir=dl_cache_dir,
        classes=classes,
    ):
        X_te_by_ref = {m: apply_reference(X_te, m, graph=ctx.graph) for m in test_modes}
        n_classes = int(max(int(y_tr.max()), int(y_te.max()))) + 1

        # Seed from (subject, seed) so re-runs are reproducible.
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
        # Train data passed in native form: the transform re-references each
        # sample at batch-time. Applying a reference here too would double-transform.
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
        free_cuda()

    return pd.DataFrame(rows)


def run_lofo_matrix(
    dataset_id: str,
    *,
    model: str,
    holdout_modes: Optional[Sequence[str]] = None,
    reference_modes: Optional[Sequence[str]] = None,
    seeds: List[int] = (0,),
    subjects: Optional[List[int]] = None,
    normalization: str = "zscore",
    progress: bool = True,
    **jitter_kwargs,
) -> pd.DataFrame:
    """Sweep run_mismatch_jitter(condition='lofo', holdout_ref=h) over h.

    Defaults: holdout_modes = reference_modes_for_dataset(dataset_id), so on
    Schirrmeister2017 cz_ref is dropped from both the universe and the
    holdout sweep. Explicit holdout_modes overrides; in that case the
    universe still defaults to reference_modes_for_dataset unless
    reference_modes= is also passed.

    holdout_modes and reference_modes accept any iterable (set, tuple, list).
    """
    universe = (
        reference_modes_for_dataset(dataset_id)
        if reference_modes is None else canonical_mode_tuple(reference_modes)
    )
    holdouts = (
        canonical_mode_tuple(holdout_modes)
        if holdout_modes is not None else universe
    )
    frames: List[pd.DataFrame] = []
    for h in holdouts:
        if h not in universe:
            raise ValueError(
                f"holdout {h!r} not in universe={universe}. Either add it to "
                f"reference_modes= or remove it from holdout_modes."
            )
        df_h = run_mismatch_jitter(
            dataset_id, model=model, condition="lofo", holdout_ref=h,
            reference_modes=universe,
            seeds=seeds, subjects=subjects, normalization=normalization,
            progress=progress, **jitter_kwargs,
        )
        frames.append(df_h)
    return pd.concat(frames, ignore_index=True)
