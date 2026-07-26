"""refshift -- reference-mismatch experiments for motor-imagery EEG.

Load a dataset, apply one of seven reference operators, and measure how well a
model trained on one reference transfers to another.

Setup
    setup_kaggle_env        set MNE_DATA/threads and symlink attached datasets

Experiments (each takes dataset_id first; pass results_dir to cache)
    run_mismatch            N x N train-ref by test-ref matrix
    run_mismatch_jitter     train one net with per-sample reference jitter
    run_loro_matrix         leave one reference out of the jitter mix
    run_lofo_matrix         leave one family out
    calibrate_csp_lda       check CSP+LDA against MOABB's published baseline

References and data
    REFERENCE_MODES, FAMILIES, reference_modes_for_dataset, canonical_mode_tuple
    apply_reference, build_graph, euclidean_alignment
    contrast_recovery_report

Reporting
    mismatch_matrix, mismatch_std_matrix, transfer_gap_ci
    report_matrix, report_families, report_jitter_full, report_loro, report_lofo
"""

from refshift.analysis import (
    mismatch_matrix,
    mismatch_std_matrix,
    report_families,
    report_jitter_full,
    report_lofo,
    report_loro,
    report_matrix,
    transfer_gap_ci,
)
from refshift.experiments import (
    calibrate_csp_lda,
    run_lofo_matrix,
    run_loro_matrix,
    run_mismatch,
    run_mismatch_jitter,
)
from refshift.inversion import contrast_recovery_report
from refshift.kaggle import setup_kaggle_env
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
    "setup_kaggle_env",
    "run_mismatch",
    "run_mismatch_jitter",
    "run_loro_matrix",
    "run_lofo_matrix",
    "calibrate_csp_lda",
    "REFERENCE_MODES",
    "FAMILIES",
    "reference_modes_for_dataset",
    "canonical_mode_tuple",
    "apply_reference",
    "build_graph",
    "euclidean_alignment",
    "contrast_recovery_report",
    "mismatch_matrix",
    "mismatch_std_matrix",
    "transfer_gap_ci",
    "report_matrix",
    "report_families",
    "report_jitter_full",
    "report_loro",
    "report_lofo",
]
