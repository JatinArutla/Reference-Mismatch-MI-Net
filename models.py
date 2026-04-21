"""
refshift.models — ATCNet (via braindecode) and CSP+LDA (via MOABB's
canonical pyriemann pipeline).

The CSP+LDA pipeline matches the recipe in MOABB's pipelines/CSP.yml exactly:
    Covariances(estimator='oas')  ->  CSP(nfilter=6)  ->  LDA(solver='svd')

Inputs for both models are [N, C, T] float32 arrays with y as int64 labels.
"""

from __future__ import annotations

from typing import Optional


# ============================================================================
# ATCNet (braindecode)
# ============================================================================

def build_atcnet(
    n_chans: int,
    n_outputs: int,
    input_window_seconds: float,
    sfreq: float,
):
    """Instantiate a fresh braindecode ATCNet with the paper's defaults.

    Args:
        n_chans:              number of EEG channels (22 for iv2a, etc.)
        n_outputs:            number of classes (4 for iv2a, 2 for others)
        input_window_seconds: trial duration in seconds (4.0 for iv2a, etc.)
        sfreq:                sampling rate (250.0 throughout this project)

    Returns:
        A torch.nn.Module. Braindecode's ATCNet uses its own canonical
        defaults (conv_block_pool_size_1=8, pool_size_2=7, tcn_depth=2,
        att_num_heads=2) that match the Altaheri reference implementation.

    Notes:
        Braindecode is imported lazily here so the module can be imported
        without torch/braindecode installed (useful for tests that don't
        need the model).
    """
    from braindecode.models import ATCNet
    return ATCNet(
        n_chans=n_chans,
        n_outputs=n_outputs,
        input_window_seconds=float(input_window_seconds),
        sfreq=float(sfreq),
    )


# ============================================================================
# CSP + LDA (MOABB canonical)
# ============================================================================

def build_csp_lda(n_filters: int = 6):
    """Build a scikit-learn Pipeline: Covariances(oas) -> CSP -> LDA(svd).

    Matches MOABB's pipelines/CSP.yml exactly. Works on [N, C, T] inputs
    via pyriemann.estimation.Covariances.

    Args:
        n_filters: number of CSP spatial filters (MOABB default 6).

    Returns:
        A sklearn Pipeline with .fit(X, y) and .predict(X) methods.
    """
    from pyriemann.estimation import Covariances
    from pyriemann.spatialfilters import CSP
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.pipeline import make_pipeline

    return make_pipeline(
        Covariances(estimator="oas"),
        CSP(nfilter=n_filters),
        LinearDiscriminantAnalysis(solver="svd"),
    )
