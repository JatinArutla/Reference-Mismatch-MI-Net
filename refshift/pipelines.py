"""Pipeline factories.

Phase 1 contains the CSP+LDA pipeline only, built to match MOABB's
canonical ``pipelines/CSP.yml``:

    Covariances(estimator='oas')  ->  CSP(nfilter=6)  ->  LinearDiscriminantAnalysis(solver='svd')

We wrap this with an optional ReferenceTransformer step at the front so the
reference choice is swappable per experiment. A transformer with
``mode='native'`` is an identity operation (up to a fresh copy) and MUST
produce calibration numbers identical to the bare MOABB pipeline within
floating-point noise; this equivalence is the cheapest end-to-end
correctness test we have.

DL pipelines (braindecode EEGNet / ShallowFBCSPNet / ATCNet + EMS) are
deferred to Phase 2.
"""

from __future__ import annotations

from typing import Optional

from sklearn.pipeline import Pipeline

from refshift.reference import DatasetGraph, ReferenceTransformer


def make_csp_lda_pipeline(
    reference_mode: Optional[str] = None,
    *,
    graph: Optional[DatasetGraph] = None,
    n_filters: int = 6,
) -> Pipeline:
    """Build CSP+LDA matching MOABB's canonical pipeline.

    Parameters
    ----------
    reference_mode : one of REFERENCE_MODES or None
        If None, no reference step is added — equivalent to MOABB's bare
        ``CSP.yml`` pipeline. If a mode string, a ReferenceTransformer(mode)
        is inserted at the front of the pipeline.
    graph : DatasetGraph or None
        Required when reference_mode is 'laplacian', 'rest', or 'cz_ref'.
    n_filters : int
        Number of CSP spatial filters. MOABB default is 6.
    """
    from pyriemann.estimation import Covariances
    from pyriemann.spatialfilters import CSP
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    steps = []
    if reference_mode is not None:
        steps.append(("reference", ReferenceTransformer(reference_mode, graph=graph)))
    steps.extend([
        ("cov", Covariances(estimator="oas")),
        ("csp", CSP(nfilter=n_filters)),
        ("lda", LinearDiscriminantAnalysis(solver="svd")),
    ])
    return Pipeline(steps)
