"""refshift — reference-shift experiments for motor-imagery EEG decoding.

Notebook API:

    setup_kaggle_env()          environment + MOABB dataset symlinks
    calibrate_csp_lda(...)      MOABB calibration, returns (results, summary, passed)
    run_mismatch(...)           6x6 mismatch matrix by dataset_id
    mismatch_matrix(df, ...)    pivot long-form results into a 6x6 table

Primitives:

    ReferenceTransformer        sklearn transformer applying a reference op
    build_graph                 Laplacian/bipolar neighbor indices
    REFERENCE_MODES             tuple of the six supported modes
    make_csp_lda_pipeline       CSP+LDA pipeline matching MOABB's canonical CSP.yml

Phase 2 (not yet implemented): braindecode DL pipelines, jitter, SSL.
"""

from refshift.reference import (
    REFERENCE_MODES,
    DatasetGraph,
    ReferenceTransformer,
    build_graph,
)
from refshift.pipelines import make_csp_lda_pipeline
from refshift.experiments import (
    calibrate_csp_lda,
    mismatch_matrix,
    run_mismatch,
)
from refshift.env import setup_kaggle_env, setup_moabb_symlinks
from refshift.plotting import plot_mismatch_matrix
from refshift.analysis import (
    cluster_references,
    mismatch_std_matrix,
    operator_distance_correlation,
    plot_dendrogram,
    plot_operator_distance_scatter,
)


__all__ = [
    "setup_kaggle_env",
    "setup_moabb_symlinks",
    "calibrate_csp_lda",
    "run_mismatch",
    "mismatch_matrix",
    "plot_mismatch_matrix",
    "mismatch_std_matrix",
    "cluster_references",
    "plot_dendrogram",
    "operator_distance_correlation",
    "plot_operator_distance_scatter",
    "REFERENCE_MODES",
    "DatasetGraph",
    "ReferenceTransformer",
    "build_graph",
    "make_csp_lda_pipeline",
]

__version__ = "0.3.0"
