"""Unit tests for reference operators and graph construction.

Ported from the v2 test suite. These are pure-numpy sanity checks that
don't require MOABB or braindecode to be installed.

Run with:
    pytest tests/ -v
"""

from __future__ import annotations

import numpy as np
import pytest

from refshift.reference import (
    REFERENCE_MODES,
    ReferenceTransformer,
    apply_reference,
    build_graph,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def small_X(rng):
    """Small [N=4, C=8, T=64] array for fast math checks."""
    return rng.standard_normal((4, 8, 64)).astype(np.float32)


@pytest.fixture
def iv2a_ch_names():
    """BCI IV-2a EEG channel set (22 channels, 10-20/10-10 standard names)."""
    return [
        "Fz",
        "FC3", "FC1", "FCz", "FC2", "FC4",
        "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
        "CP3", "CP1", "CPz", "CP2", "CP4",
        "P1", "Pz", "P2",
        "POz",
    ]


# ---------------------------------------------------------------------------
# Global-mean family: algebraic properties
# ---------------------------------------------------------------------------

def test_native_is_copy_of_input(small_X):
    Y = apply_reference(small_X, "native")
    assert Y.shape == small_X.shape
    np.testing.assert_array_equal(Y, small_X.astype(np.float32))
    assert Y.base is None or Y.base is not small_X.base  # not a view


def test_car_residual_channel_mean_is_zero(small_X):
    Y = apply_reference(small_X, "car")
    # After CAR, the mean across channels is ~0 at every timepoint.
    resid = Y.mean(axis=1)
    assert np.max(np.abs(resid)) < 1e-5


def test_median_residual_channel_median_is_zero(small_X):
    Y = apply_reference(small_X, "median")
    resid = np.median(Y, axis=1)
    assert np.max(np.abs(resid)) < 1e-5


def test_gs_orthogonality_per_trial_per_channel(small_X):
    """After GS, Y[n,c,:] is orthogonal (time-domain inner product) to the
    leave-one-out mean r[n,c,:] computed on the *input* X. This is the
    algebraic guarantee of the GS projection: <Y, r_X> = 0, not <Y, r_Y>.
    """
    X = small_X
    Y = apply_reference(X, "gs")
    _, C, _ = X.shape
    s = X.sum(axis=1, keepdims=True)
    r = (s - X) / max(C - 1, 1)                      # LOO mean of X, not Y
    inner = np.sum(Y * r, axis=2)                    # [N, C]
    norm_r = np.sqrt(np.sum(r * r, axis=2))
    norm_Y = np.sqrt(np.sum(Y * Y, axis=2))
    cos = inner / np.maximum(norm_r * norm_Y, 1e-10)
    assert np.max(np.abs(cos)) < 1e-5


def test_2d_and_3d_shapes_equivalent(small_X):
    """Calling with [C, T] should give the same result as calling with
    [1, C, T] and squeezing."""
    single = small_X[0]                      # [C, T]
    for mode in ("native", "car", "median", "gs"):
        Y2 = apply_reference(single, mode)
        Y3 = apply_reference(small_X[:1], mode)[0]
        np.testing.assert_allclose(Y2, Y3, atol=1e-6)


# ---------------------------------------------------------------------------
# Spatial family: hand-computed 3-channel cases
# ---------------------------------------------------------------------------

def test_laplacian_hand_case():
    """3 channels, k=2 => every channel's Laplacian reference is the mean
    of the other two."""
    X = np.array([[1.0, 2.0],
                  [3.0, 4.0],
                  [5.0, 6.0]], dtype=np.float32)  # [C=3, T=2]
    # Each row's 2 nearest neighbors are the other two rows (index set).
    lap_idx = np.array([[1, 2],
                        [0, 2],
                        [0, 1]], dtype=np.int64)
    # Manually:
    #   ref[0] = mean([3,4], [5,6]) = [4,5]  -> Y[0] = [1-4, 2-5] = [-3,-3]
    #   ref[1] = mean([1,2], [5,6]) = [3,4]  -> Y[1] = [0, 0]
    #   ref[2] = mean([1,2], [3,4]) = [2,3]  -> Y[2] = [3, 3]
    from refshift.reference import _laplacian  # noqa: PLC0415
    Y = _laplacian(X, lap_idx)
    expected = np.array([[-3, -3], [0, 0], [3, 3]], dtype=np.float32)
    np.testing.assert_allclose(Y, expected, atol=1e-6)


def test_bipolar_hand_case():
    X = np.array([[1.0, 2.0],
                  [3.0, 4.0],
                  [5.0, 6.0]], dtype=np.float32)
    # nearest neighbor: 0->1, 1->0, 2->1
    bip_idx = np.array([1, 0, 1], dtype=np.int64)
    from refshift.reference import _bipolar  # noqa: PLC0415
    Y = _bipolar(X, bip_idx)
    # Row 0: X[0] - X[1] = [1-3, 2-4] = [-2, -2]
    # Row 1: X[1] - X[0] = [3-1, 4-2] = [+2, +2]
    # Row 2: X[2] - X[1] = [5-3, 6-4] = [+2, +2]
    expected = np.array([[-2, -2], [2, 2], [2, 2]], dtype=np.float32)
    np.testing.assert_allclose(Y, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Graph construction (uses MNE; skip if unavailable)
# ---------------------------------------------------------------------------

def test_build_graph_iv2a_c3_nearest_is_cp3(iv2a_ch_names):
    """Under standard_1005, C3's single nearest neighbor in the IV-2a
    channel set should be CP3. This is an anatomical sanity check."""
    pytest.importorskip("mne")
    g = build_graph(iv2a_ch_names, k=4, montage="standard_1005")

    c3 = iv2a_ch_names.index("C3")
    cp3 = iv2a_ch_names.index("CP3")
    assert g.bipolar_idx[c3] == cp3

    # k=4: CP3 should appear among C3's Laplacian neighbors.
    assert cp3 in g.laplacian_idx[c3].tolist()


def test_build_graph_no_self_loops(iv2a_ch_names):
    pytest.importorskip("mne")
    g = build_graph(iv2a_ch_names, k=4)
    C = len(iv2a_ch_names)
    for c in range(C):
        assert c not in g.laplacian_idx[c].tolist(), f"self-loop in Laplacian at {c}"
        assert g.bipolar_idx[c] != c, f"self-loop in bipolar at {c}"


# ---------------------------------------------------------------------------
# sklearn API contract
# ---------------------------------------------------------------------------

def test_transformer_is_sklearn_compatible(small_X):
    """ReferenceTransformer should have fit/transform/fit_transform and be
    clonable (required for MOABB's Evaluation)."""
    from sklearn.base import clone
    t = ReferenceTransformer(mode="car")
    t2 = clone(t)
    assert t2.mode == "car"
    assert t2.graph is None

    # fit returns self; transform is pure
    out = t.fit(small_X, y=None).transform(small_X)
    assert out.shape == small_X.shape
    np.testing.assert_allclose(
        out.mean(axis=1), 0.0, atol=1e-5
    )


def test_transformer_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unknown mode"):
        ReferenceTransformer(mode="reref_wisdom").fit(np.zeros((1, 4, 8)))


def test_transformer_spatial_requires_graph():
    with pytest.raises(ValueError, match="requires graph"):
        ReferenceTransformer(mode="laplacian").fit(np.zeros((1, 4, 8)))


def test_transformer_roundtrip_all_modes(small_X, iv2a_ch_names):
    """Smoke test: every mode produces a float32 array of the same shape."""
    pytest.importorskip("mne")
    # small_X has C=8, but iv2a fixture is C=22 — build graph for 8 random
    # channels from standard_1005 instead.
    g = build_graph(iv2a_ch_names[:8], k=4)
    for mode in REFERENCE_MODES:
        t = ReferenceTransformer(mode=mode, graph=g if mode in ("laplacian", "bipolar") else None)
        out = t.fit_transform(small_X)
        assert out.shape == small_X.shape
        assert out.dtype == np.float32
        assert np.isfinite(out).all()
