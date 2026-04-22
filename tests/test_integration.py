"""End-to-end integration test.

Requires MOABB + pyriemann installed and network access on first run to
download the IV-2a .mat files. Auto-skips otherwise — most local dev loops
only need the pure-numpy tests in test_reference.py.

The core claim this test enforces: inserting
``ReferenceTransformer('native')`` at the front of MOABB's canonical
CSP+LDA pipeline must not change the score, because 'native' is the
identity operator (up to a fresh copy).

If this test fails, the ReferenceTransformer has silently broken
MOABB's pipeline and everything downstream (mismatch matrix, DL runs)
is suspect.
"""

from __future__ import annotations

import numpy as np
import pytest


moabb = pytest.importorskip("moabb")
pyriemann = pytest.importorskip("pyriemann")


@pytest.fixture(scope="module")
def iv2a_subject1():
    """Load IV-2a subject 1 via MOABB's MotorImagery paradigm once per module."""
    from moabb.datasets import BNCI2014_001
    from moabb.paradigms import MotorImagery
    paradigm = MotorImagery(n_classes=4)
    dataset = BNCI2014_001()
    X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[1])
    return X, y, metadata


def test_identity_transformer_does_not_change_csp_lda_score(iv2a_subject1):
    """Score with and without ReferenceTransformer('native') must match
    to within floating-point noise on the same 5-fold split.
    """
    from sklearn.model_selection import cross_val_score, StratifiedKFold

    from refshift.pipelines import make_csp_lda_pipeline
    from refshift.data import encode_labels

    X, y_raw, _ = iv2a_subject1
    y, _ = encode_labels(y_raw)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    bare = make_csp_lda_pipeline(reference_mode=None)
    with_native = make_csp_lda_pipeline(reference_mode="native")

    s_bare = cross_val_score(bare, X, y, cv=cv, scoring="accuracy")
    s_with_native = cross_val_score(with_native, X, y, cv=cv, scoring="accuracy")

    # Per-fold scores should match within numerical noise.
    np.testing.assert_allclose(s_bare, s_with_native, atol=1e-6)


def test_csp_lda_subject1_accuracy_in_expected_range(iv2a_subject1):
    """Subject 1 is a strong subject on IV-2a. Expect well above chance
    (0.25 for 4-class) with the canonical pipeline. This is a coarse
    smoke test — not a calibration check (that lives in the calibration
    script).
    """
    from sklearn.model_selection import cross_val_score, StratifiedKFold

    from refshift.pipelines import make_csp_lda_pipeline
    from refshift.data import encode_labels

    X, y_raw, _ = iv2a_subject1
    y, _ = encode_labels(y_raw)

    pipe = make_csp_lda_pipeline(reference_mode="native")
    scores = cross_val_score(
        pipe, X, y,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="accuracy",
    )
    assert scores.mean() > 0.50, (
        f"Subject 1 CSP+LDA accuracy {scores.mean():.3f} is surprisingly low; "
        f"expect >0.50 for this subject on IV-2a"
    )
