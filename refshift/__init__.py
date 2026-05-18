"""refshift — reference-shift experiments for motor-imagery EEG decoding.

A classifier trained under one EEG reference operator and tested under another
suffers a structured, predictable accuracy collapse. This package measures
that across five MOABB MI datasets, two DL architectures + classical CSP+LDA,
eight reference and spatial operators, and four interventions: per-sample
reference jitter, leave-one-reference-out training, an EMS-control ablation
(diagonal and full matrix), and a bandpass-mismatch control.

Reference operators (v0.15+):
    Global symmetric:     native, car, median, rest (Yao 2001 spherical-model)
    Global asymmetric:    cz_ref           (X_i - X_Cz)
    Local spatial-deriv:  lap_small        (Hjorth k=4 NN Laplacian)
                          lap_large        (deterministic next-ring large-Laplacian
                                            approximation; motivated by McFarland 1997
                                            but not equivalent on sparse montages)
                          csd              (Perrin spherical-spline surface Laplacian)

Notebook API:
    setup_kaggle_env()         environment + MOABB dataset symlinks
    calibrate_csp_lda(...)     MOABB CSP+LDA calibration
    run_mismatch(...)          NxN mismatch matrix (CSP+LDA or DL)
    run_mismatch_jitter(...)   DL with per-sample jitter (full or LOFO)
    run_lofo_matrix(...)       sweep LOFO over every reference
    run_pre_ems_diagonal(...)  EMS-control diagonal
    run_pre_ems_mismatch(...)  EMS-control full NxN matrix (v0.16+); apply reference
                               BEFORE EMS rather than after, to disentangle operator
                               topology from EMS-interaction effects
    run_bandpass_mismatch(...) bandpass-mismatch control
    mismatch_matrix(df, ...)   long-form -> NxN pivot

Restrict the operator set per run by passing any iterable as reference_modes:

    REFERENCES = {"native", "car", "csd"}
    df = run_mismatch("iv2a", model="csp_lda", reference_modes=REFERENCES)

Output column order is always canonical (REFERENCE_MODES) regardless of
input iteration order. The legacy name 'laplacian' is accepted as an alias
for 'lap_small' (operator unchanged; renamed in v0.15).

v0.16 caveat for CSD operator-distance analyses:
    Use distance_metric='frobenius_normed' (the v0.16 default) when CSD is
    in the operator set, because raw Frobenius distance is dominated by
    CSD's amplitude scale rather than spatial topology.

Primitives:
    REFERENCE_MODES, ReferenceTransformer, build_graph, apply_reference,
    canonical_mode_tuple
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
    run_pre_ems_mismatch,
)
from refshift.kaggle import setup_kaggle_env, setup_moabb_symlinks
from refshift.model import SUPPORTED_DL_MODELS, make_csp_lda_pipeline, make_dl_model
from refshift.plotting import plot_mismatch_matrix
from refshift.reference import (
    REFERENCE_MODES,
    ReferenceTransformer,
    apply_reference,
    build_graph,
    canonical_mode_tuple,
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
    "run_pre_ems_mismatch",
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
    "canonical_mode_tuple",
    "reference_modes_for_dataset",
    "validate_reference_modes",
    "make_csp_lda_pipeline",
    "make_dl_model",
    "SUPPORTED_DL_MODELS",
]

__version__ = "0.16.0"
