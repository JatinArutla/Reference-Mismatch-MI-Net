"""Unit tests for the reference operators and families (no MOABB/torch needed)."""

import numpy as np
import pytest

from refshift.references import (
    FAMILIES,
    GRAPH_MODES,
    REFERENCE_MODES,
    DatasetGraph,
    apply_reference,
    build_graph,
    canonical_mode_tuple,
    euclidean_alignment,
)


def _graph(n_channels=22, cz_idx=9):
    idx = np.tile(np.arange(4), (n_channels, 1)).astype(np.int64)
    return DatasetGraph(
        ch_names=[f"c{i}" for i in range(n_channels)],
        lap_small_idx=idx, lap_large_idx=idx,
        k_small=4, k_large_skip=4, k_large_use=4,
        montage="standard_1005", cz_idx=cz_idx,
    )


def test_reference_modes_and_families_consistent():
    # Every family member is a known mode; native is intentionally excluded.
    members = [m for ms in FAMILIES.values() for m in ms]
    assert set(members) == set(REFERENCE_MODES) - {"native"}
    # No mode appears in two families.
    assert len(members) == len(set(members))


def test_canonical_mode_tuple_orders_and_validates():
    assert canonical_mode_tuple({"car", "native", "rest"}) == ("native", "car", "rest")
    with pytest.raises(ValueError):
        canonical_mode_tuple(["not_a_mode"])


def test_car_is_zero_mean():
    X = np.random.randn(5, 8, 50).astype(np.float32)
    out = apply_reference(X, "car")
    assert np.allclose(out.mean(axis=1), 0.0, atol=1e-5)


def test_median_subtracts_channel_median():
    X = np.random.randn(3, 6, 20).astype(np.float32)
    out = apply_reference(X, "median")
    expected = X - np.median(X, axis=1, keepdims=True)
    assert np.allclose(out, expected, atol=1e-6)


def test_native_is_identity():
    X = np.random.randn(4, 7, 30).astype(np.float32)
    assert np.array_equal(apply_reference(X, "native"), X)


def test_cz_ref_zeroes_cz_channel():
    X = np.random.randn(4, 22, 40).astype(np.float32)
    out = apply_reference(X, "cz_ref", graph=_graph(cz_idx=9))
    assert np.allclose(out[:, 9, :], 0.0, atol=1e-6)


def test_graph_modes_require_graph():
    X = np.random.randn(2, 22, 10).astype(np.float32)
    for mode in GRAPH_MODES:
        with pytest.raises(ValueError):
            apply_reference(X, mode, graph=None)


def test_unknown_mode_raises():
    X = np.random.randn(2, 4, 10).astype(np.float32)
    with pytest.raises(ValueError):
        apply_reference(X, "bogus")


def test_ea_whitens_to_identity():
    # After EA, the mean trial covariance should be ~ identity.
    X = np.random.randn(20, 5, 100).astype(np.float32)
    out = euclidean_alignment(X)
    covs = np.stack([np.cov(out[i].astype(np.float64)) for i in range(out.shape[0])])
    assert np.allclose(covs.mean(axis=0), np.eye(5), atol=0.1)


def test_ea_handles_empty_block():
    X = np.empty((0, 5, 100), dtype=np.float32)
    out = euclidean_alignment(X)
    assert out.shape == (0, 5, 100)


def test_laplacian_neighbours_are_anatomical():
    # The small Laplacian must pick each channel's true nearest neighbours on the
    # IV-2a montage, not just subtract the mean of arbitrary indices.
    pytest.importorskip("mne")
    chans = ["Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz",
             "C2", "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz",
             "P2", "POz"]
    g = build_graph(chans)
    idx = {c: i for i, c in enumerate(chans)}
    neighbours = lambda c: {chans[j] for j in g.lap_small_idx[idx[c]]}
    assert neighbours("Cz") == {"C1", "C2", "CPz", "FCz"}
    assert neighbours("C3") == {"C1", "C5", "CP3", "FC3"}
