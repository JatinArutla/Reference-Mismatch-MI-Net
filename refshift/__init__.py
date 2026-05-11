"""refshift — reference-shift experiments for motor-imagery EEG decoding.

A classifier trained under one EEG reference operator and tested under another
suffers a structured, predictable accuracy collapse. This package measures
that across five MOABB MI datasets, two DL architectures + classical CSP+LDA,
six reference and spatial operators, and three interventions: per-sample
reference jitter, leave-one-reference-out training, and an EMS-control
ablation.

Notebook API:
    setup_kaggle_env()         environment + MOABB dataset symlinks
    calibrate_csp_lda(...)     MOABB CSP+LDA calibration
    run_mismatch(...)          6x6 mismatch matrix (CSP+LDA or DL)
    run_mismatch_jitter(...)   DL with per-sample jitter (full or LOFO)
    run_lofo_matrix(...)       sweep LOFO over every reference
    run_pre_ems_diagonal(...)  EMS-control ablation
    run_bandpass_mismatch(...) bandpass-mismatch control
    mismatch_matrix(df, ...)   long-form -> 6x6 pivot

Primitives:
    REFERENCE_MODES, ReferenceTransformer, build_graph, apply_reference
    make_csp_lda_pipeline, make_dl_model
"""

from refshift.analysis import (
    baseline_col_off_diag_view,
    baseline_diagonal_view,
    cluster_references,
    mismatch_std_matrix,
    operator_distance_correlation,
    paired_wilcoxon_per_test_ref,
    plot_dendrogram,
    plot_operator_distance_scatter,
)
from refshift.experiments import (
    calibrate_csp_lda,
    mismatch_matrix,
    run_bandpass_mismatch,
    run_lofo_matrix,
    run_mismatch,
    run_mismatch_jitter,
    run_pre_ems_diagonal,
)
from refshift.kaggle import setup_kaggle_env, setup_moabb_symlinks
from refshift.model import SUPPORTED_DL_MODELS, make_csp_lda_pipeline, make_dl_model
from refshift.plotting import plot_mismatch_matrix
from refshift.reference import (
    REFERENCE_MODES,
    ReferenceTransformer,
    apply_reference,
    build_graph,
    reference_modes_for_dataset,
    validate_reference_modes,
)
from refshift.report import report_experiment


__all__ = [
    # setup
    "setup_kaggle_env",
    "setup_moabb_symlinks",
    # runners
    "calibrate_csp_lda",
    "run_mismatch",
    "run_mismatch_jitter",
    "run_lofo_matrix",
    "run_pre_ems_diagonal",
    "run_bandpass_mismatch",
    # pivots and plots
    "mismatch_matrix",
    "plot_mismatch_matrix",
    "report_experiment",
    # analyses
    "mismatch_std_matrix",
    "cluster_references",
    "plot_dendrogram",
    "operator_distance_correlation",
    "plot_operator_distance_scatter",
    "paired_wilcoxon_per_test_ref",
    "baseline_diagonal_view",
    "baseline_col_off_diag_view",
    # primitives
    "REFERENCE_MODES",
    "ReferenceTransformer",
    "build_graph",
    "apply_reference",
    "reference_modes_for_dataset",
    "validate_reference_modes",
    "make_csp_lda_pipeline",
    "make_dl_model",
    "SUPPORTED_DL_MODELS",
]

__version__ = "0.14.2"
