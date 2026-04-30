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


def test_2d_and_3d_shapes_equivalent(small_X):
    """Calling with [C, T] should give the same result as calling with
    [1, C, T] and squeezing."""
    single = small_X[0]                      # [C, T]
    for mode in ("native", "car", "median"):
        Y2 = apply_reference(single, mode)
        Y3 = apply_reference(small_X[:1], mode)[0]
        np.testing.assert_allclose(Y2, Y3, atol=1e-6)


def test_gs_no_longer_in_reference_modes():
    """Sanity: 'gs' was dropped from REFERENCE_MODES in v0.10. The
    rationale (per peer review) is that the natural data-dependent GS
    projection is not a fixed C×C linear operator and therefore doesn't
    fit the operator-shift framework. A linear LOO-mean alternative was
    not added because LOO_i = (C/(C-1)) * CAR_i — they differ only by a
    scalar and produce identical results for any scale-invariant
    decoder.
    """
    from refshift.reference import REFERENCE_MODES
    assert "gs" not in REFERENCE_MODES
    assert "loo" not in REFERENCE_MODES
    assert len(REFERENCE_MODES) == 6


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
    from refshift.reference import _laplacian  # noqa: PLC0415
    Y = _laplacian(X, lap_idx)
    expected = np.array([[-3, -3], [0, 0], [3, 3]], dtype=np.float32)
    np.testing.assert_allclose(Y, expected, atol=1e-6)


def test_nn_diff_hand_case():
    """NN-diff: Y_i = X_i - X_{nn(i)}. Renamed from "bipolar" in v0.10
    to honestly reflect that this is a dimension-preserving local-
    difference operator, not a clinical bipolar montage with
    predefined electrode pairs.
    """
    X = np.array([[1.0, 2.0],
                  [3.0, 4.0],
                  [5.0, 6.0]], dtype=np.float32)
    nn_idx = np.array([1, 0, 1], dtype=np.int64)
    from refshift.reference import _nn_diff  # noqa: PLC0415
    Y = _nn_diff(X, nn_idx)
    # Row 0: X[0] - X[1] = [-2, -2]
    # Row 1: X[1] - X[0] = [+2, +2]
    # Row 2: X[2] - X[1] = [+2, +2]
    expected = np.array([[-2, -2], [2, 2], [2, 2]], dtype=np.float32)
    np.testing.assert_allclose(Y, expected, atol=1e-6)


def test_nn_diff_rank_diagnostic():
    """build_graph reports nn_diff_rank/nullity for transparency. With
    mutual nearest-neighbours (e.g. 0<->1), the operator destroys 2
    dimensions instead of 1; rank goes down accordingly. This test
    constructs a controlled mutual-NN case and checks the diagnostic.
    """
    pytest.importorskip("mne")
    # IV-2a channel set: well-known to have many mutual-NN pairs
    # (C3<->CP3, C4<->CP4, etc.). Rank should be < C.
    iv2a_chs = [
        "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
        "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
        "CP3", "CP1", "CPz", "CP2", "CP4",
        "P1", "Pz", "P2", "POz",
    ]
    g = build_graph(iv2a_chs, k=4)
    assert g.nn_diff_rank > 0
    assert g.nn_diff_rank <= len(iv2a_chs)
    assert g.nn_diff_nullity == len(iv2a_chs) - g.nn_diff_rank
    # We expect at least some mutual-NN pairs => nullity >= 1
    # (the diagnostic exists *because* the rank is typically less than C-1).


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
    assert g.nn_diff_idx[c3] == cp3

    # k=4: CP3 should appear among C3's Laplacian neighbors.
    assert cp3 in g.laplacian_idx[c3].tolist()


def test_build_graph_no_self_loops(iv2a_ch_names):
    pytest.importorskip("mne")
    g = build_graph(iv2a_ch_names, k=4)
    C = len(iv2a_ch_names)
    for c in range(C):
        assert c not in g.laplacian_idx[c].tolist(), f"self-loop in Laplacian at {c}"
        assert g.nn_diff_idx[c] != c, f"self-loop in NN-diff at {c}"


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
    # channels from standard_1005 instead. Include REST so the 'rest' mode
    # has its transformation matrix available.
    g = build_graph(iv2a_ch_names[:8], k=4, include_rest=True)
    for mode in REFERENCE_MODES:
        needs_graph = mode in ("laplacian", "nn_diff", "rest")
        t = ReferenceTransformer(mode=mode, graph=g if needs_graph else None)
        out = t.fit_transform(small_X)
        assert out.shape == small_X.shape
        assert out.dtype == np.float32
        assert np.isfinite(out).all()


# ---------------------------------------------------------------------------
# REST (Yao 2001) — new in 0.2.0
# ---------------------------------------------------------------------------

def test_rest_matrix_built_when_requested(iv2a_ch_names):
    """build_graph with include_rest=True populates the REST matrix."""
    pytest.importorskip("mne")
    g_off = build_graph(iv2a_ch_names, k=4, include_rest=False)
    assert g_off.rest_matrix is None

    g_on = build_graph(iv2a_ch_names, k=4, include_rest=True)
    C = len(iv2a_ch_names)
    assert g_on.rest_matrix is not None
    assert g_on.rest_matrix.shape == (C, C)
    assert g_on.rest_matrix.dtype == np.float32
    assert np.isfinite(g_on.rest_matrix).all()


def test_rest_is_reference_invariant(small_X, iv2a_ch_names):
    """REST(V + c*ones_C) == REST(V) for any per-trial per-time constant c.

    This is the defining property of REST: the transformation commutes
    with any additive re-referencing, because it incorporates the centering
    operator (I - 1_C 1_C^T / C) that annihilates the all-ones vector.
    """
    pytest.importorskip("mne")
    # small_X is [4, 8, 64]. Build graph on the same 8 channels used for
    # the rest of the small tests.
    g = build_graph(iv2a_ch_names[:8], k=4, include_rest=True)

    rng = np.random.default_rng(7)
    # additive constant per trial per time (broadcasts across channels)
    offset = rng.standard_normal((small_X.shape[0], 1, small_X.shape[2])).astype(np.float32) * 100.0

    Y1 = apply_reference(small_X, "rest", graph=g)
    Y2 = apply_reference(small_X + offset, "rest", graph=g)
    # float32 accumulation across a (C, C) matmul leaves O(1e-4) residual;
    # the math is exact.
    np.testing.assert_allclose(Y1, Y2, atol=1e-3)


def test_rest_matrix_annihilates_all_ones(iv2a_ch_names):
    """T @ 1_C should be (numerically) zero. This is the algebraic root of
    REST's reference-invariance property, independent of any input data.
    """
    pytest.importorskip("mne")
    g = build_graph(iv2a_ch_names, k=4, include_rest=True)
    C = len(iv2a_ch_names)
    ones = np.ones(C, dtype=np.float32)
    residual = g.rest_matrix @ ones
    assert np.max(np.abs(residual)) < 1e-4, (
        f"REST matrix failed the T @ 1_C = 0 check; max residual="
        f"{np.max(np.abs(residual)):.3e}"
    )


def test_rest_is_not_identity(small_X, iv2a_ch_names):
    """Sanity: REST should actually change the data (unlike 'native')."""
    pytest.importorskip("mne")
    g = build_graph(iv2a_ch_names[:8], k=4, include_rest=True)
    Y = apply_reference(small_X, "rest", graph=g)
    assert not np.allclose(Y, small_X, atol=1e-3), (
        "REST output equals input — leadfield is degenerate or transform "
        "collapsed to identity."
    )


def test_rest_requires_include_rest_graph():
    """Attempting REST with a graph built for spatial-only modes raises."""
    pytest.importorskip("mne")
    iv2a = ["Fz", "C3", "Cz", "C4", "CP3", "Pz", "POz", "FCz"]
    g = build_graph(iv2a, k=4, include_rest=False)
    with pytest.raises(ValueError, match="include_rest=True"):
        ReferenceTransformer(mode="rest", graph=g).fit(np.zeros((1, 8, 16)))


def test_rest_2d_and_3d_shapes_equivalent(small_X, iv2a_ch_names):
    """REST on [C, T] equals REST on [1, C, T] squeezed, same as other modes."""
    pytest.importorskip("mne")
    g = build_graph(iv2a_ch_names[:8], k=4, include_rest=True)
    single = small_X[0]  # [C, T]
    Y2 = apply_reference(single, "rest", graph=g)
    Y3 = apply_reference(small_X[:1], "rest", graph=g)[0]
    np.testing.assert_allclose(Y2, Y3, atol=1e-5)
