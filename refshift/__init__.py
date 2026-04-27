"""refshift — reference-shift experiments for motor-imagery EEG decoding.

Notebook API:

    setup_kaggle_env()          environment + MOABB dataset symlinks
    calibrate_csp_lda(...)      MOABB calibration, returns (results, summary, passed)
    run_mismatch(...)           7x7 mismatch matrix by dataset_id (CSP+LDA or DL)
    run_mismatch_jitter(...)    DL training with per-sample reference jitter
                                (full or LOFO-bipolar), evaluated on all 7 refs
    mismatch_matrix(df, ...)    pivot long-form results into a 7x7 table

Primitives:

    ReferenceTransformer        sklearn transformer applying a reference op
    build_graph                 Laplacian/bipolar/REST neighbor indices
    REFERENCE_MODES             tuple of the seven supported modes
    make_csp_lda_pipeline       CSP+LDA pipeline matching MOABB's canonical CSP.yml

Phase 2 (DL):
    refshift.dl.load_dl_data    braindecode MOABBDataset + canonical preprocess.
                                Supports a ``cache_dir=`` argument that
                                caches the preprocessed (X, y, metadata,
                                sfreq, ch_names) tuple to disk keyed on a
                                hash of all preprocessing parameters; reused
                                automatically by both ``run_mismatch`` and
                                ``run_mismatch_jitter`` via ``dl_cache_dir=``.
    refshift.dl.make_dl_model   EEGNetv4 / ShallowFBCSPNet factory (skorch-wrapped)
    refshift.jitter             RandomReferenceTransform for per-sample jitter
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
    run_mismatch_jitter,
)
from refshift.env import setup_kaggle_env, setup_moabb_symlinks
from refshift.plotting import plot_mismatch_matrix
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


__all__ = [
    "setup_kaggle_env",
    "setup_moabb_symlinks",
    "calibrate_csp_lda",
    "run_mismatch",
    "run_mismatch_jitter",
    "mismatch_matrix",
    "plot_mismatch_matrix",
    "mismatch_std_matrix",
    "cluster_references",
    "plot_dendrogram",
    "operator_distance_correlation",
    "plot_operator_distance_scatter",
    "paired_wilcoxon_per_test_ref",
    "baseline_diagonal_view",
    "baseline_col_off_diag_view",
    "REFERENCE_MODES",
    "DatasetGraph",
    "ReferenceTransformer",
    "build_graph",
    "make_csp_lda_pipeline",
]

__version__ = "0.7.2"
