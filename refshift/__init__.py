"""refshift — reference-shift experiments for motor-imagery EEG decoding.

Notebook API:

    setup_kaggle_env()          environment + MOABB dataset symlinks
    calibrate_csp_lda(...)      MOABB calibration, returns (results, summary, passed)
    run_mismatch(...)           6x6 mismatch matrix by dataset_id (CSP+LDA or DL)
    run_mismatch_jitter(...)    DL training with per-sample reference jitter
                                (full or LOFO with one held-out reference),
                                evaluated on all 6 refs.
    run_lofo_matrix(...)        Convenience wrapper that runs LOFO once per
                                reference in REFERENCE_MODES and concatenates
                                the long-form output, producing a complete
                                LOFO-by-test-ref table for all 6 hold-outs.
    run_pre_ems_diagonal(...)   EMS-control ablation: train + test on the same
                                reference, applied *before* exponential moving
                                standardization. Returns a 6-number diagonal
                                comparable to the standard pipeline's diagonal.
    run_bandpass_mismatch(...)  Preprocessing-mismatch control: train on one
                                bandpass, test on a shifted bandpass. Used to
                                show the reference effect is not generic
                                preprocessing brittleness.
    mismatch_matrix(df, ...)    pivot long-form results into a 6x6 table

Primitives:

    ReferenceTransformer        sklearn transformer applying a reference op
    build_graph                 kNN-Laplacian / REST / cz_ref state for one
                                channel set (neighbour indices, REST matrix,
                                Cz channel index)
    REFERENCE_MODES             tuple of the six supported modes
    make_csp_lda_pipeline       CSP+LDA pipeline matching MOABB's canonical CSP.yml

Phase 2 (DL):
    refshift.dl.load_dl_data    braindecode MOABBDataset + canonical preprocess.
                                Resamples to a common rate (default 250 Hz)
                                so the time-domain receptive field of every
                                model is identical across datasets. Supports a
                                ``cache_dir=`` argument that caches the
                                preprocessed (X, y, metadata, sfreq, ch_names)
                                tuple to disk keyed on a hash of all
                                preprocessing parameters; reused automatically
                                by both ``run_mismatch`` and
                                ``run_mismatch_jitter`` via ``dl_cache_dir=``.
    refshift.dl.make_dl_model   EEGNetv4 / ShallowFBCSPNet factory (skorch-wrapped)
    refshift.jitter             RandomReferenceTransform for per-sample jitter
    SUPPORTED_DL_MODELS         ('eegnet', 'shallow')
"""

from refshift.reference import (
    REFERENCE_MODES,
    ReferenceTransformer,
    build_graph,
)
from refshift.pipelines import make_csp_lda_pipeline
from refshift.experiments import (
    calibrate_csp_lda,
    mismatch_matrix,
    run_bandpass_mismatch,
    run_lofo_matrix,
    run_mismatch,
    run_mismatch_jitter,
    run_pre_ems_diagonal,
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
    "run_lofo_matrix",
    "run_pre_ems_diagonal",
    "run_bandpass_mismatch",
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
    "ReferenceTransformer",
    "build_graph",
    "make_csp_lda_pipeline",
]

__version__ = "0.13.0"
