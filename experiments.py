"""
refshift.experiments — experiment runners for the paper's two main analyses.

Two entry points:

    run_mismatch_matrix(dataset, subject, ...) -> DataFrame
        Runs the 6x6 (train_ref x test_ref) mismatch matrix for one subject.
        Trains ONE model per train_ref and evaluates it against all 6
        test_refs (6 trainings + 36 evaluations per subject). This is the
        core paper result.

    run_jitter(dataset, subject, training_refs, ...) -> DataFrame
        Trains ONE model with per-trial reference jitter (each batch
        samples a reference uniformly from `training_refs`) and evaluates
        it against each reference mode in test. Shows whether exposing
        the model to multiple references during training improves
        reference-invariance at test time.

Outputs are long-form DataFrames one row per (train_ref, test_ref) cell.
For session-split datasets (iv2a, openbmi), uses the dataset's own split.
For single-session datasets (cho2017, dreyer2023), creates a stratified
80/20 split per (subject, seed).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from loader import ALL_SUBJECTS, load_subject, SubjectData
from preprocessing import bandpass_subject_data, DEFAULT_BANDPASS_HZ
from reference_ops import (
    REFERENCE_MODES,
    apply_reference,
    build_graph,
)
from standardization import (
    apply_standardizer,
    fit_standardizer,
    standardize_mechanistic,
)
from models import build_atcnet
from training import (
    evaluate_atcnet,
    evaluate_csp_lda,
    fit_atcnet,
    fit_csp_lda,
    make_jitter_batch_fn,
)


# ============================================================================
# Helpers
# ============================================================================

def _get_train_test(data: SubjectData, seed: int = 0, test_frac: float = 0.2):
    """Return (X_tr, y_tr, X_te, y_te).

    Session-split datasets: use the native train/test split.
    Single-session datasets: stratified shuffle split, 80/20 by default.
    """
    if data.has_session_split():
        return data.X_train, data.y_train, data.X_test, data.y_test
    from sklearn.model_selection import StratifiedShuffleSplit
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_frac,
                                      random_state=seed)
    tr_idx, te_idx = next(splitter.split(data.X_all, data.y_all))
    return (data.X_all[tr_idx], data.y_all[tr_idx],
            data.X_all[te_idx], data.y_all[te_idx])


def _standardize_train_and_test(
    X_tr_ref: np.ndarray, X_te_ref: np.ndarray, protocol: str
) -> Tuple[np.ndarray, np.ndarray, Optional[Tuple[np.ndarray, np.ndarray]]]:
    """Apply standardization per protocol. Returns (X_tr_std, X_te_std, fit_stats).

    mechanistic: per-trial per-channel z-score on each set independently.
                 fit_stats is None.
    deployment:  per-channel mu/sd fitted on X_tr_ref only, applied to both.
                 fit_stats is (mu, sd).
    """
    if protocol == "mechanistic":
        return standardize_mechanistic(X_tr_ref), standardize_mechanistic(X_te_ref), None
    if protocol == "deployment":
        mu, sd = fit_standardizer(X_tr_ref)
        return apply_standardizer(X_tr_ref, mu, sd), apply_standardizer(X_te_ref, mu, sd), (mu, sd)
    raise ValueError(f"Unknown standardization protocol: {protocol!r}")


# ============================================================================
# Mismatch matrix runner (main paper result)
# ============================================================================

def run_mismatch_matrix(
    dataset_id: str,
    subject: int,
    *,
    reference_modes: Optional[List[str]] = None,
    standardization: str = "mechanistic",
    model_type: str = "atcnet",
    bandpass: Tuple[float, float] = DEFAULT_BANDPASS_HZ,
    laplacian_k: int = 4,
    n_epochs: int = 200,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    seed: int = 0,
    device: str = "auto",
    verbose: bool = False,
) -> pd.DataFrame:
    """Run the full 6x6 train_ref x test_ref mismatch matrix for one subject.

    For each train_ref, trains one model and evaluates it against all six
    test_refs. Total: 6 trainings + 36 evaluations per subject.

    Args:
        dataset_id:      'iv2a' | 'openbmi' | 'cho2017' | 'dreyer2023'
        subject:         subject ID
        reference_modes: defaults to all 6 modes in reference_ops.REFERENCE_MODES
        standardization: 'mechanistic' or 'deployment'
        model_type:      'atcnet' or 'csp_lda' (csp_lda skips standardization)
        bandpass:        (low_hz, high_hz)
        laplacian_k:     neighbors per channel for laplacian reference
        n_epochs:        training epochs (ATCNet only)
        batch_size:      batch size (ATCNet only)
        lr:              learning rate (ATCNet only)
        weight_decay:    AdamW weight decay (ATCNet only)
        seed:            RNG seed
        device:          'auto' | 'cuda' | 'cpu' (ATCNet only)
        verbose:         print progress per train_ref

    Returns:
        DataFrame with columns: dataset, subject, train_ref, test_ref,
        accuracy, kappa, n_test, model, standardization, seed.
    """
    if reference_modes is None:
        reference_modes = list(REFERENCE_MODES)
    if standardization not in ("mechanistic", "deployment"):
        raise ValueError(f"standardization: {standardization!r}")
    if model_type not in ("atcnet", "csp_lda"):
        raise ValueError(f"model_type: {model_type!r}")

    # Load + bandpass once
    data = load_subject(dataset_id, subject)
    data_bp = bandpass_subject_data(data, band=bandpass)
    X_tr_bp, y_tr, X_te_bp, y_te = _get_train_test(data_bp, seed=seed)
    n_classes = int(len(np.unique(y_tr)))
    graph = build_graph(data_bp.ch_names, k=laplacian_k)

    # Pre-compute all test-set reference versions (cheap)
    X_te_by_ref = {
        mode: apply_reference(X_te_bp, mode, graph=graph)
        for mode in reference_modes
    }

    rows = []
    for train_ref in reference_modes:
        if verbose:
            print(f"[{dataset_id} sub{subject} seed{seed}] train_ref={train_ref}")
        X_tr_ref = apply_reference(X_tr_bp, train_ref, graph=graph)

        if model_type == "atcnet":
            # Standardize train; for each test_ref we'll standardize separately
            if standardization == "mechanistic":
                X_tr_std = standardize_mechanistic(X_tr_ref)
                fit_stats = None
            else:  # deployment
                mu, sd = fit_standardizer(X_tr_ref)
                X_tr_std = apply_standardizer(X_tr_ref, mu, sd)
                fit_stats = (mu, sd)

            model = fit_atcnet(
                X_tr_std, y_tr, n_classes=n_classes, sfreq=data_bp.sfreq,
                n_epochs=n_epochs, batch_size=batch_size, lr=lr,
                weight_decay=weight_decay, seed=seed, device=device,
                verbose=False,
            )

            for test_ref in reference_modes:
                X_te_ref = X_te_by_ref[test_ref]
                if standardization == "mechanistic":
                    X_te_std = standardize_mechanistic(X_te_ref)
                else:
                    mu, sd = fit_stats
                    X_te_std = apply_standardizer(X_te_ref, mu, sd)
                m = evaluate_atcnet(model, X_te_std, y_te, device=device)
                rows.append(dict(
                    dataset=dataset_id, subject=subject,
                    train_ref=train_ref, test_ref=test_ref,
                    accuracy=m["accuracy"], kappa=m["kappa"],
                    n_test=m["n_test"], model=model_type,
                    standardization=standardization, seed=seed,
                ))

        elif model_type == "csp_lda":
            # CSP+LDA doesn't use standardization (covariance trace normalization
            # inside CSP handles scale). Fit on referenced data directly.
            pipeline = fit_csp_lda(X_tr_ref, y_tr)
            for test_ref in reference_modes:
                m = evaluate_csp_lda(pipeline, X_te_by_ref[test_ref], y_te)
                rows.append(dict(
                    dataset=dataset_id, subject=subject,
                    train_ref=train_ref, test_ref=test_ref,
                    accuracy=m["accuracy"], kappa=m["kappa"],
                    n_test=m["n_test"], model=model_type,
                    standardization="none", seed=seed,
                ))

    return pd.DataFrame(rows)


def run_dataset_benchmark(
    dataset_id: str,
    *,
    subjects: Optional[List[int]] = None,
    seeds: List[int] = (0,),
    verbose: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """Run the 6x6 mismatch matrix for every (subject, seed).

    Args:
        dataset_id:    which dataset
        subjects:      defaults to ALL_SUBJECTS[dataset_id]
        seeds:         list of seeds (for mean +/- std over initializations)
        verbose:       print per-subject progress
        **kwargs:      passed through to run_mismatch_matrix

    Returns:
        Concatenated DataFrame across subjects and seeds.
    """
    if subjects is None:
        subjects = ALL_SUBJECTS[dataset_id]
    dfs = []
    for seed in seeds:
        for sub in subjects:
            if verbose:
                print(f"--- {dataset_id} subject {sub}, seed {seed} ---")
            df = run_mismatch_matrix(
                dataset_id, sub, seed=seed, verbose=verbose, **kwargs
            )
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


# ============================================================================
# Jitter runner
# ============================================================================

def run_jitter(
    dataset_id: str,
    subject: int,
    *,
    training_refs: List[str],
    test_refs: Optional[List[str]] = None,
    bandpass: Tuple[float, float] = DEFAULT_BANDPASS_HZ,
    laplacian_k: int = 4,
    n_epochs: int = 200,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    seed: int = 0,
    device: str = "auto",
    verbose: bool = False,
) -> pd.DataFrame:
    """Train ATCNet with per-trial reference jitter, then evaluate.

    Each batch during training samples a reference per trial uniformly at
    random from `training_refs`. Test is then evaluated against each
    reference in `test_refs` separately. Mechanistic standardization is
    used throughout (deployment standardization doesn't compose cleanly
    with per-trial reference changes).

    Args:
        training_refs: references to sample from during training (e.g.,
                       ['native', 'car', 'laplacian'])
        test_refs:     references to evaluate against (default: all 6)

    Returns:
        DataFrame with columns: dataset, subject, training_refs, test_ref,
        accuracy, kappa, n_test, model='atcnet_jitter', standardization,
        seed.
    """
    if test_refs is None:
        test_refs = list(REFERENCE_MODES)
    if not training_refs:
        raise ValueError("training_refs must be non-empty")

    data = load_subject(dataset_id, subject)
    data_bp = bandpass_subject_data(data, band=bandpass)
    X_tr_bp, y_tr, X_te_bp, y_te = _get_train_test(data_bp, seed=seed)
    n_classes = int(len(np.unique(y_tr)))
    graph = build_graph(data_bp.ch_names, k=laplacian_k)

    # Pre-compute standardized train versions for each training_ref
    X_tr_by_ref = {
        ref: standardize_mechanistic(apply_reference(X_tr_bp, ref, graph=graph))
        for ref in training_refs
    }
    # Pre-compute standardized test versions for each test_ref
    X_te_by_ref = {
        ref: standardize_mechanistic(apply_reference(X_te_bp, ref, graph=graph))
        for ref in test_refs
    }

    # Jitter batch function
    get_batch = make_jitter_batch_fn(X_tr_by_ref, training_refs, seed=seed)

    # Use any one of the pre-computed arrays as the shape template for the trainer
    X_shape_ref = next(iter(X_tr_by_ref.values()))

    if verbose:
        print(f"[jitter {dataset_id} sub{subject} seed{seed}] "
              f"training_refs={training_refs}, n_classes={n_classes}")

    model = fit_atcnet(
        X_shape_ref, y_tr, n_classes=n_classes, sfreq=data_bp.sfreq,
        n_epochs=n_epochs, batch_size=batch_size, lr=lr,
        weight_decay=weight_decay, seed=seed, device=device,
        get_train_batch=get_batch, verbose=False,
    )

    rows = []
    training_refs_str = "|".join(training_refs)
    for test_ref in test_refs:
        m = evaluate_atcnet(model, X_te_by_ref[test_ref], y_te, device=device)
        rows.append(dict(
            dataset=dataset_id, subject=subject,
            training_refs=training_refs_str, test_ref=test_ref,
            accuracy=m["accuracy"], kappa=m["kappa"],
            n_test=m["n_test"], model="atcnet_jitter",
            standardization="mechanistic", seed=seed,
        ))
    return pd.DataFrame(rows)


# ============================================================================
# Output aggregation helpers
# ============================================================================

def mismatch_matrix_mean(df: pd.DataFrame, metric: str = "accuracy") -> pd.DataFrame:
    """Pivot a long-form result DataFrame into a 6x6 mean-accuracy table.

    Aggregates across (subject, seed) within each (train_ref, test_ref) cell.
    """
    return (df
            .groupby(["train_ref", "test_ref"])[metric]
            .mean()
            .unstack("test_ref"))


def mismatch_matrix_std(df: pd.DataFrame, metric: str = "accuracy") -> pd.DataFrame:
    """Standard deviation across (subject, seed) per cell."""
    return (df
            .groupby(["train_ref", "test_ref"])[metric]
            .std()
            .unstack("test_ref"))
