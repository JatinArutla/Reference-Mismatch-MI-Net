"""refshift -- reference-mismatch experiments for EEG motor imagery (IV-2a).

A small, focused package: load IV-2a, apply one of seven reference operators,
and measure how well a model trained on one reference transfers to another.

Public API
----------
Setup (Kaggle):
    setup_kaggle_env       set MNE_DATA/threads and symlink attached datasets
    setup_moabb_symlinks   lower-level symlink helper

Experiments:
    run_mismatch            N x N train-ref by test-ref matrix (CSP+LDA or deep)
    run_mismatch_jitter     train one deep net with per-sample reference jitter
    run_loro_matrix         leave-one-reference-out sweep
    run_lofo_matrix         leave-one-family-out sweep
    calibrate_csp_lda       MOABB baseline sanity check

References and data:
    REFERENCE_MODES, FAMILIES, reference_modes_for_dataset, canonical_mode_tuple
    apply_reference, build_graph, euclidean_alignment

Analysis:
    mismatch_matrix, mismatch_std_matrix
    report_matrix, report_families, report_jitter_full, report_loro, report_lofo
"""

from refshift.analysis import (
    bootstrap_ci,
    mismatch_matrix,
    mismatch_std_matrix,
    per_subject_values,
    report_families,
    report_jitter_full,
    report_lofo,
    report_loro,
    report_matrix,
    report_transfer_gap,
    transfer_gap_ci,
)
from refshift.experiments import (
    calibrate_csp_lda,
    run_lofo_matrix,
    run_loro_matrix,
    run_mismatch,
    run_mismatch_jitter,
)
from refshift.inversion import contrast_recovery_report, operator_matrix
from refshift.kaggle import setup_kaggle_env, setup_moabb_symlinks
from refshift.references import (
    FAMILIES,
    REFERENCE_MODES,
    apply_reference,
    build_graph,
    canonical_mode_tuple,
    euclidean_alignment,
    reference_modes_for_dataset,
)

__all__ = [
    # experiments
    "run_mismatch",
    "run_mismatch_jitter",
    "run_loro_matrix",
    "run_lofo_matrix",
    "calibrate_csp_lda",
    # environment setup
    "setup_kaggle_env",
    "setup_moabb_symlinks",
    # references and data
    "REFERENCE_MODES",
    "FAMILIES",
    "reference_modes_for_dataset",
    "canonical_mode_tuple",
    "apply_reference",
    "build_graph",
    "euclidean_alignment",
    "contrast_recovery_report",
    "operator_matrix",
    # analysis
    "mismatch_matrix",
    "mismatch_std_matrix",
    "report_matrix",
    "report_families",
    "report_jitter_full",
    "report_loro",
    "report_lofo",
    # subject-level statistics
    "per_subject_values",
    "bootstrap_ci",
    "transfer_gap_ci",
    "report_transfer_gap",
]
