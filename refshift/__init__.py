"""refshift — reference-shift experiments for motor-imagery EEG decoding.

Public API (Phase 1):
    ReferenceTransformer   — sklearn transformer applying a reference op to [N,C,T] arrays
    build_graph            — build Laplacian/bipolar neighbor indices from channel names
    REFERENCE_MODES        — tuple of the six supported modes
    make_csp_lda_pipeline  — CSP+LDA pipeline matching MOABB's canonical CSP.yml
    run_mismatch_matrix    — train-once/evaluate-six-times runner for CSP+LDA
"""

from refshift.reference import (
    REFERENCE_MODES,
    DatasetGraph,
    ReferenceTransformer,
    build_graph,
)
from refshift.pipelines import make_csp_lda_pipeline
from refshift.mismatch import run_mismatch_matrix

__all__ = [
    "REFERENCE_MODES",
    "DatasetGraph",
    "ReferenceTransformer",
    "build_graph",
    "make_csp_lda_pipeline",
    "run_mismatch_matrix",
]

__version__ = "0.1.0"
