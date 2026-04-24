"""refshift — reference-shift experiments for motor-imagery EEG decoding.

Notebook API:

    setup_kaggle_env()          environment + MOABB dataset symlinks
    calibrate_csp_lda(...)      MOABB calibration, returns (results, summary, passed)
    run_mismatch(...)           7x7 mismatch matrix by dataset_id (CSP+LDA or DL)
    mismatch_matrix(df, ...)    pivot long-form results into a 7x7 table

Primitives:

    ReferenceTransformer        sklearn transformer applying a reference op
    build_graph                 Laplacian/bipolar/REST neighbor indices
    REFERENCE_MODES             tuple of the seven supported modes
    make_csp_lda_pipeline       CSP+LDA pipeline matching MOABB's canonical CSP.yml

Phase 2 (DL):
    refshift.dl.load_dl_data    braindecode MOABBDataset + canonical preprocess
    refshift.dl.make_dl_model   EEGNetv4 / ShallowFBCSPNet factory (skorch-wrapped)
    SUPPORTED_DL_MODELS         ('eegnet', 'shallow')
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

__version__ = "0.4.0"
