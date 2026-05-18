"""Unit tests for reference operators and graph construction.

v0.15: tests cover the 8-operator set (native, car, median, rest, cz_ref,
lap_small, lap_large, csd) plus legacy 'laplacian' alias resolution.
"""

from __future__ import annotations

import numpy as np
import pytest

from refshift.reference import (
    REFERENCE_MODES,
    ReferenceTransformer,
    apply_reference,
    build_graph,
    canonical_mode_tuple,
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


def test_2d_input_rejected(small_X):
    """Ops only accept (N, C, T). 2D inputs raise (was supported in v0.13;
    dropped because nothing in the experimental pipeline uses it)."""
    with pytest.raises(ValueError, match="N, C, T"):
        apply_reference(small_X[0], "car")


def test_reference_modes_v015_contents():
    """v0.15: 8 modes across three families. 'gs', 'loo', 'nn_diff' remain
    excluded; 'laplacian' was renamed to 'lap_small' (alias kept for old CSVs);
    'lap_large' and 'csd' are new."""
    from refshift.reference import REFERENCE_MODES
    assert "gs" not in REFERENCE_MODES
    assert "loo" not in REFERENCE_MODES
    assert "nn_diff" not in REFERENCE_MODES
    assert "laplacian" not in REFERENCE_MODES  # renamed to lap_small
    assert len(REFERENCE_MODES) == 8
    for m in ("native", "car", "median", "rest", "cz_ref",
              "lap_small", "lap_large", "csd"):
        assert m in REFERENCE_MODES, f"missing {m!r}"


# ---------------------------------------------------------------------------
# Spatial family: hand-computed 3-channel cases
# ---------------------------------------------------------------------------

def test_lap_small_hand_case():
    """3 channels, k=2 => every channel's lap_small reference is the mean
    of the other two."""
    X = np.array([[[1.0, 2.0],
                   [3.0, 4.0],
                   [5.0, 6.0]]], dtype=np.float32)  # [N=1, C=3, T=2]
    lap_idx = np.array([[1, 2],
                        [0, 2],
                        [0, 1]], dtype=np.int64)
    from refshift.reference import _lap_small  # noqa: PLC0415
    Y = _lap_small(X, lap_idx)
    expected = np.array([[[-3, -3], [0, 0], [3, 3]]], dtype=np.float32)
    np.testing.assert_allclose(Y, expected, atol=1e-6)


def test_legacy_laplacian_function_alias_works():
    """The pre-v0.15 ``_laplacian`` function name still resolves to ``_lap_small``."""
    from refshift.reference import _lap_small, _laplacian
    assert _laplacian is _lap_small


# ---------------------------------------------------------------------------
# Alias resolution: 'laplacian' -> 'lap_small'
# ---------------------------------------------------------------------------

def test_apply_reference_accepts_laplacian_alias(small_X, iv2a_ch_names):
    """Old CSVs and notebooks may pass 'laplacian'; should resolve to 'lap_small'."""
    pytest.importorskip("mne")
    chs8 = ["Fz", "FC1", "FCz", "FC2", "C3", "Cz", "C4", "Pz"]
    g = build_graph(chs8, k_small=4)
    Y_old = apply_reference(small_X, "laplacian", graph=g)
    Y_new = apply_reference(small_X, "lap_small", graph=g)
    np.testing.assert_allclose(Y_old, Y_new, atol=1e-7)


def test_transformer_accepts_laplacian_alias(small_X, iv2a_ch_names):
    pytest.importorskip("mne")
    chs8 = ["Fz", "FC1", "FCz", "FC2", "C3", "Cz", "C4", "Pz"]
    g = build_graph(chs8, k_small=4)
    Y_old = ReferenceTransformer(mode="laplacian", graph=g).transform(small_X)
    Y_new = ReferenceTransformer(mode="lap_small", graph=g).transform(small_X)
    np.testing.assert_allclose(Y_old, Y_new, atol=1e-7)


def test_canonical_mode_tuple_orders_set_canonically():
    """Passing a set yields canonical REFERENCE_MODES ordering."""
    out = canonical_mode_tuple({"csd", "native", "car"})
    assert out == ("native", "car", "csd")


def test_canonical_mode_tuple_resolves_aliases():
    out = canonical_mode_tuple(["laplacian", "car"])
    assert out == ("car", "lap_small")


def test_canonical_mode_tuple_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown reference mode"):
        canonical_mode_tuple(["not_a_real_mode"])


# ---------------------------------------------------------------------------
# Graph construction (uses MNE; skip if unavailable)
# ---------------------------------------------------------------------------

def test_build_graph_iv2a_c3_nearest_is_cp3(iv2a_ch_names):
    """Under standard_1005, C3's single nearest neighbour in the IV-2a
    channel set should be CP3. Anatomical sanity check on the KD-tree
    distance computation that drives ``lap_small_idx``."""
    pytest.importorskip("mne")
    g = build_graph(iv2a_ch_names, k_small=4, montage="standard_1005")

    c3 = iv2a_ch_names.index("C3")
    cp3 = iv2a_ch_names.index("CP3")
    # lap_small_idx is sorted by ascending distance, so position 0 is
    # the single closest neighbour.
    assert g.lap_small_idx[c3][0] == cp3
    assert cp3 in g.lap_small_idx[c3].tolist()


def test_build_graph_legacy_field_aliases(iv2a_ch_names):
    """Backward-compat: graph.laplacian_idx == graph.lap_small_idx;
    graph.k == graph.k_small."""
    pytest.importorskip("mne")
    g = build_graph(iv2a_ch_names, k_small=4)
    np.testing.assert_array_equal(g.laplacian_idx, g.lap_small_idx)
    assert g.k == g.k_small == 4


def test_build_graph_legacy_k_kwarg(iv2a_ch_names):
    """Backward-compat: build_graph(..., k=4) is still accepted as an alias
    for k_small=4."""
    pytest.importorskip("mne")
    g = build_graph(iv2a_ch_names, k=4)
    assert g.k_small == 4


def test_build_graph_no_self_loops_lap_small(iv2a_ch_names):
    pytest.importorskip("mne")
    g = build_graph(iv2a_ch_names, k_small=4)
    C = len(iv2a_ch_names)
    for c in range(C):
        assert c not in g.lap_small_idx[c].tolist(), (
            f"self-loop in lap_small at {c}"
        )


def test_build_graph_no_self_loops_lap_large(iv2a_ch_names):
    pytest.importorskip("mne")
    g = build_graph(iv2a_ch_names, k_small=4, k_large_skip=4, k_large_use=4)
    C = len(iv2a_ch_names)
    for c in range(C):
        assert c not in g.lap_large_idx[c].tolist(), (
            f"self-loop in lap_large at {c}"
        )


# ---------------------------------------------------------------------------
# lap_large (McFarland next-ring Laplacian)
# ---------------------------------------------------------------------------

def test_lap_large_disjoint_from_lap_small_default(iv2a_ch_names):
    """Defaults (k_small=4, k_large_skip=4, k_large_use=4) give DISJOINT
    neighbour sets between lap_small and lap_large for every channel,
    by construction (small uses ranks 0..3; large uses ranks 4..7)."""
    pytest.importorskip("mne")
    g = build_graph(iv2a_ch_names, k_small=4, k_large_skip=4, k_large_use=4)
    C = len(iv2a_ch_names)
    for c in range(C):
        small = set(g.lap_small_idx[c].tolist())
        large = set(g.lap_large_idx[c].tolist())
        assert small.isdisjoint(large), (
            f"channel {c}: lap_small and lap_large neighbours overlap; "
            f"small={small}, large={large}"
        )


def test_lap_large_shape(iv2a_ch_names):
    pytest.importorskip("mne")
    g = build_graph(iv2a_ch_names, k_small=4, k_large_skip=4, k_large_use=4)
    assert g.lap_large_idx.shape == (len(iv2a_ch_names), 4)


def test_lap_large_changes_data(small_X, iv2a_ch_names):
    """lap_large output != native (sanity)."""
    pytest.importorskip("mne")
    chs8 = ["Fz", "FC1", "FCz", "FC2", "C3", "Cz", "C4", "Pz"]
    g = build_graph(chs8, k_small=4, k_large_skip=4, k_large_use=4)
    Y = apply_reference(small_X, "lap_large", graph=g)
    assert not np.allclose(Y, small_X, atol=1e-3)


def test_lap_large_zero_row_sum(small_X, iv2a_ch_names):
    """Each row of lap_large operator sums to zero (it's a discrete Laplacian).
    So lap_large(X + c*ones_C) == lap_large(X) (additive-constant invariant)."""
    pytest.importorskip("mne")
    chs8 = ["Fz", "FC1", "FCz", "FC2", "C3", "Cz", "C4", "Pz"]
    g = build_graph(chs8, k_small=4, k_large_skip=4, k_large_use=4)
    Y1 = apply_reference(small_X, "lap_large", graph=g)
    offset = np.full_like(small_X, 7.5)  # constant added to every channel
    Y2 = apply_reference(small_X + offset, "lap_large", graph=g)
    np.testing.assert_allclose(Y1, Y2, atol=1e-4)


# ---------------------------------------------------------------------------
# CSD (Perrin spherical-spline surface Laplacian) — new in v0.15
# ---------------------------------------------------------------------------

def test_csd_matrix_built_when_requested(iv2a_ch_names):
    """build_graph with include_csd=True populates the CSD matrix."""
    pytest.importorskip("mne")
    g_off = build_graph(iv2a_ch_names, k_small=4, include_csd=False)
    assert g_off.csd_matrix is None
    assert g_off.csd_cond is None

    g_on = build_graph(iv2a_ch_names, k_small=4, include_csd=True)
    C = len(iv2a_ch_names)
    assert g_on.csd_matrix is not None
    assert g_on.csd_matrix.shape == (C, C)
    assert g_on.csd_matrix.dtype == np.float32
    assert np.isfinite(g_on.csd_matrix).all()
    assert g_on.csd_cond is not None and g_on.csd_cond > 0


def test_csd_requires_include_csd_graph(iv2a_ch_names):
    """Attempting CSD with a graph built for spatial-only modes raises."""
    pytest.importorskip("mne")
    chs8 = ["Fz", "FC1", "FCz", "FC2", "C3", "Cz", "C4", "Pz"]
    g = build_graph(chs8, k_small=4, include_csd=False)
    with pytest.raises(ValueError, match="include_csd=True"):
        apply_reference(np.zeros((1, 8, 16), dtype=np.float32), "csd", graph=g)


def test_csd_is_not_identity(small_X, iv2a_ch_names):
    """Sanity: CSD output should differ from input (it's a surface Laplacian)."""
    pytest.importorskip("mne")
    chs8 = ["Fz", "FC1", "FCz", "FC2", "C3", "Cz", "C4", "Pz"]
    g = build_graph(chs8, k_small=4, include_csd=True)
    Y = apply_reference(small_X, "csd", graph=g)
    assert not np.allclose(Y, small_X, atol=1e-3)


def test_csd_matches_mne_per_epoch(iv2a_ch_names):
    """The matrix-cached CSD is mathematically identical to applying MNE's
    compute_current_source_density per epoch. Verify on a synthetic batch.
    """
    mne = pytest.importorskip("mne")
    chs = iv2a_ch_names
    C = len(chs)
    g = build_graph(chs, k_small=4, include_csd=True)

    rng = np.random.default_rng(123)
    # Use float64 internally to avoid float32 accumulation noise dominating.
    X = rng.standard_normal((3, C, 16)).astype(np.float64)
    # apply_reference returns float32; cast to float64 for comparison.
    Y_matrix = apply_reference(X.astype(np.float32), "csd", graph=g).astype(np.float64)

    # Now run the same data through MNE per-epoch.
    info = mne.create_info(ch_names=chs, sfreq=250.0, ch_types="eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1005"))
    epochs = mne.EpochsArray(X, info, verbose="ERROR")
    epochs_csd = mne.preprocessing.compute_current_source_density(
        epochs, sphere="auto", copy=True, verbose="ERROR",
    )
    Y_mne = epochs_csd.get_data().astype(np.float64)

    # Relative error: the matrix recovery is exact (machine precision in float64),
    # so float32 round-trip leaves ~1e-3 relative on amplitudes ~1e4.
    rel = np.max(np.abs(Y_matrix - Y_mne)) / max(np.max(np.abs(Y_mne)), 1e-9)
    assert rel < 1e-3, f"matrix CSD differs from MNE per-epoch: rel max diff={rel:.3e}"


def test_csd_row_sums_near_zero(iv2a_ch_names):
    """The CSD operator approximates the surface Laplacian, which has row
    sums ~ 0 (constants map to 0). Not exact like CAR, but close on a
    well-conditioned montage."""
    pytest.importorskip("mne")
    g = build_graph(iv2a_ch_names, k_small=4, include_csd=True)
    row_sums = g.csd_matrix.sum(axis=1)
    # CSD via spherical splines is approximately, not exactly, zero-sum;
    # tolerance reflects typical MNE output on standard_1005.
    assert np.max(np.abs(row_sums)) < 1.0, (
        f"CSD row sums far from zero: max abs row sum={np.max(np.abs(row_sums)):.3e}"
    )


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
    with pytest.raises(ValueError, match="Unknown reference mode"):
        ReferenceTransformer(mode="reref_wisdom").transform(np.zeros((1, 4, 8)))


def test_transformer_spatial_requires_graph():
    with pytest.raises(ValueError, match="requires a DatasetGraph"):
        ReferenceTransformer(mode="lap_small").transform(np.zeros((1, 4, 8)))


def test_transformer_roundtrip_all_modes(small_X, iv2a_ch_names):
    """Smoke test: every mode in REFERENCE_MODES produces a float32 array
    of the same shape."""
    pytest.importorskip("mne")
    from refshift.reference import _GRAPH_MODES
    # small_X has C=8. Pick an 8-channel subset that includes Cz so that
    # cz_ref is exercised by the roundtrip; the rest of the operators
    # don't care which 8 channels are chosen.
    chs8 = ["Fz", "FC1", "FCz", "FC2", "C3", "Cz", "C4", "Pz"]
    assert all(c in iv2a_ch_names for c in chs8)
    g = build_graph(chs8, k_small=4, include_rest=True, include_csd=True)
    for mode in REFERENCE_MODES:
        needs_graph = mode in _GRAPH_MODES
        t = ReferenceTransformer(mode=mode, graph=g if needs_graph else None)
        out = t.fit_transform(small_X)
        assert out.shape == small_X.shape
        assert out.dtype == np.float32
        assert np.isfinite(out).all()


# ---------------------------------------------------------------------------
# REST (Yao 2001)
# ---------------------------------------------------------------------------

def test_rest_matrix_built_when_requested(iv2a_ch_names):
    """build_graph with include_rest=True populates the REST matrix."""
    pytest.importorskip("mne")
    g_off = build_graph(iv2a_ch_names, k_small=4, include_rest=False)
    assert g_off.rest_matrix is None

    g_on = build_graph(iv2a_ch_names, k_small=4, include_rest=True)
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
    g = build_graph(iv2a_ch_names[:8], k_small=4, include_rest=True)

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
    g = build_graph(iv2a_ch_names, k_small=4, include_rest=True)
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
    g = build_graph(iv2a_ch_names[:8], k_small=4, include_rest=True)
    Y = apply_reference(small_X, "rest", graph=g)
    assert not np.allclose(Y, small_X, atol=1e-3), (
        "REST output equals input — leadfield is degenerate or transform "
        "collapsed to identity."
    )


def test_rest_requires_include_rest_graph():
    """Attempting REST with a graph built for spatial-only modes raises."""
    pytest.importorskip("mne")
    iv2a = ["Fz", "C3", "Cz", "C4", "CP3", "Pz", "POz", "FCz"]
    g = build_graph(iv2a, k_small=4, include_rest=False)
    with pytest.raises(ValueError, match="include_rest=True"):
        ReferenceTransformer(mode="rest", graph=g).transform(np.zeros((1, 8, 16)))


# ---------------------------------------------------------------------------
# cz_ref (single-electrode reference using Cz)
# ---------------------------------------------------------------------------

def test_cz_idx_populated_when_cz_present():
    """build_graph stores the index of 'Cz' in the channel order, or
    None when Cz is absent."""
    pytest.importorskip("mne")
    chs = ["Fz", "C3", "Cz", "C4", "CP3", "Pz", "POz", "FCz"]
    g = build_graph(chs, k_small=4, include_rest=False)
    assert g.cz_idx == 2

    # No Cz -> cz_idx = None (Schirrmeister-style case)
    chs_no_cz = ["Fz", "FC3", "FC1", "FCz", "C3", "C4", "CP3", "Pz"]
    assert "Cz" not in chs_no_cz
    g2 = build_graph(chs_no_cz, k_small=4, include_rest=False)
    assert g2.cz_idx is None


def test_cz_ref_zeros_the_cz_channel(small_X):
    """After cz_ref, the Cz row is identically zero."""
    pytest.importorskip("mne")
    chs = ["Fz", "C3", "Cz", "C4", "CP3", "Pz", "POz", "FCz"]
    g = build_graph(chs, k_small=4, include_rest=False)
    Y = apply_reference(small_X, "cz_ref", graph=g)
    assert Y.shape == small_X.shape
    cz = g.cz_idx
    np.testing.assert_array_equal(Y[:, cz, :], 0.0)


def test_cz_ref_linear_relationship():
    """Y_i = X_i - X_{Cz} exactly. Verify against direct computation."""
    pytest.importorskip("mne")
    chs = ["Fz", "C3", "Cz", "C4", "CP3", "Pz", "POz", "FCz"]
    g = build_graph(chs, k_small=4, include_rest=False)
    rng = np.random.default_rng(42)
    X = rng.standard_normal((3, len(chs), 64)).astype(np.float32)
    Y = apply_reference(X, "cz_ref", graph=g)
    cz = g.cz_idx
    expected = X - X[:, cz:cz + 1, :]
    np.testing.assert_allclose(Y, expected, atol=1e-6)


def test_cz_ref_rank_is_c_minus_one():
    """The cz_ref operator (I - 1_C e_{Cz}^T) has rank C-1."""
    pytest.importorskip("mne")
    chs = ["Fz", "C3", "Cz", "C4", "CP3", "Pz", "POz", "FCz"]
    g = build_graph(chs, k_small=4, include_rest=False)
    C = len(chs)
    # Recover the operator matrix via a Gaussian probe
    rng = np.random.default_rng(0)
    X = rng.standard_normal((1, C, 200)).astype(np.float32)
    Y = apply_reference(X, "cz_ref", graph=g)
    A = Y[0] @ np.linalg.pinv(X[0])
    assert np.linalg.matrix_rank(A) == C - 1


def test_cz_ref_idempotent():
    """cz_ref(cz_ref(X)) == cz_ref(X). Once Cz is zero, subtracting Cz again
    is a no-op."""
    pytest.importorskip("mne")
    chs = ["Fz", "C3", "Cz", "C4", "CP3", "Pz", "POz", "FCz"]
    g = build_graph(chs, k_small=4, include_rest=False)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((2, len(chs), 32)).astype(np.float32)
    Y = apply_reference(X, "cz_ref", graph=g)
    Y2 = apply_reference(Y, "cz_ref", graph=g)
    np.testing.assert_allclose(Y, Y2, atol=1e-6)


def test_cz_ref_raises_when_cz_absent():
    """If Cz isn't in the channel set, apply_reference and the transformer
    both raise an informative ValueError that mentions Schirrmeister."""
    pytest.importorskip("mne")
    chs_no_cz = ["Fz", "FC3", "FC1", "FCz", "C3", "C4", "CP3", "Pz"]
    g = build_graph(chs_no_cz, k_small=4, include_rest=False)

    rng = np.random.default_rng(0)
    X = rng.standard_normal((1, len(chs_no_cz), 16)).astype(np.float32)

    with pytest.raises(ValueError, match="cz_idx=None"):
        apply_reference(X, "cz_ref", graph=g)
    with pytest.raises(ValueError, match="cz_idx=None"):
        ReferenceTransformer(mode="cz_ref", graph=g).transform(X)
