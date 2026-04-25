"""Unit tests for refshift.jitter (per-sample reference augmentation).

Skipped if Phase 2 deps are missing. All tests run on CPU with synthetic
data — no MOABB / network calls.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("braindecode")
pytest.importorskip("torch")
pytest.importorskip("skorch")

import torch  # noqa: E402

from refshift.reference import REFERENCE_MODES, build_graph  # noqa: E402


# ---- Channel layout used by all tests ---------------------------------------

# Use a small subset of standard 10-05 channels that the build_graph helper
# already supports for IV-2a, OpenBMI, etc.
_CH_NAMES = ["Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3",
             "C1", "Cz", "C2", "C4", "C6", "CP3", "CP1", "CPz",
             "CP2", "CP4", "P1", "Pz", "P2", "POz"]


def _make_graph(include_rest: bool = True):
    return build_graph(_CH_NAMES, k=4, montage="standard_1005",
                       include_rest=include_rest)


def _make_synthetic_batch(B: int = 32, T: int = 250, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((B, len(_CH_NAMES), T)).astype(np.float32)
    y = rng.integers(0, 4, size=(B,)).astype(np.int64)
    return X, y


# ---- Construction validation -----------------------------------------------

def test_make_random_reference_transform_unknown_mode_raises():
    from refshift.jitter import make_random_reference_transform
    with pytest.raises(ValueError):
        make_random_reference_transform(["car", "no_such_mode"])


def test_make_random_reference_transform_empty_modes_raises():
    from refshift.jitter import make_random_reference_transform
    with pytest.raises(ValueError):
        make_random_reference_transform([])


def test_make_random_reference_transform_requires_graph_for_spatial():
    from refshift.jitter import make_random_reference_transform
    with pytest.raises(ValueError):
        make_random_reference_transform(["car", "laplacian"], graph=None)


def test_make_random_reference_transform_rest_requires_rest_matrix():
    from refshift.jitter import make_random_reference_transform
    g = _make_graph(include_rest=False)
    with pytest.raises(ValueError):
        make_random_reference_transform(["rest"], graph=g)


def test_make_random_reference_transform_global_modes_no_graph_ok():
    """Global-mean modes (native, car, median, gs) don't need a graph."""
    from refshift.jitter import make_random_reference_transform
    t = make_random_reference_transform(
        ["native", "car", "median", "gs"], graph=None,
    )
    assert t is not None


# ---- Forward-pass behaviour -------------------------------------------------

def test_transform_preserves_shape_and_dtype():
    from refshift.jitter import make_random_reference_transform
    g = _make_graph()
    t = make_random_reference_transform(REFERENCE_MODES, graph=g, random_state=0)
    X, y = _make_synthetic_batch()
    X_t, y_t = t(torch.from_numpy(X), torch.from_numpy(y))
    assert X_t.shape == X.shape
    assert X_t.dtype == torch.float32
    assert y_t.shape == y.shape
    # y should pass through untouched
    np.testing.assert_array_equal(y_t.numpy(), y)


def test_transform_native_only_is_identity():
    """Sampling exclusively from {'native'} should return X unchanged."""
    from refshift.jitter import make_random_reference_transform
    t = make_random_reference_transform(["native"], graph=None, random_state=0)
    X, y = _make_synthetic_batch()
    X_t, _ = t(torch.from_numpy(X), torch.from_numpy(y))
    np.testing.assert_allclose(X_t.numpy(), X, atol=1e-6)


def test_transform_car_only_zero_means():
    """Sampling exclusively from {'car'} -> every output sample sums to ~0
    along the channel axis."""
    from refshift.jitter import make_random_reference_transform
    t = make_random_reference_transform(["car"], graph=None, random_state=0)
    X, y = _make_synthetic_batch()
    X_t, _ = t(torch.from_numpy(X), torch.from_numpy(y))
    means = X_t.numpy().mean(axis=1)  # (B, T)
    np.testing.assert_allclose(means, 0.0, atol=1e-5)


def test_transform_uses_multiple_modes_per_batch():
    """With 7 allowed modes and a 32-trial batch, we should observe at least
    a few distinct modes' fingerprints in the output. We test this by checking
    that not all output samples have zero channel-mean (which would mean every
    sample got CAR / median / Laplacian / bipolar / rest, but never native or
    GS — extremely unlikely under uniform sampling).
    """
    from refshift.jitter import make_random_reference_transform
    g = _make_graph()
    t = make_random_reference_transform(REFERENCE_MODES, graph=g, random_state=0)
    X, y = _make_synthetic_batch(B=64)
    X_t, _ = t(torch.from_numpy(X), torch.from_numpy(y))
    # Per-sample channel means
    means = np.abs(X_t.numpy().mean(axis=1)).max(axis=1)  # (B,)
    # Some samples should retain a non-zero channel mean (native or laplacian
    # or bipolar leave channel mean non-zero in general; CAR/median/gs/rest do not).
    assert (means > 1e-3).any(), "No sample retained non-zero channel mean"


def test_transform_per_sample_independence():
    """Two consecutive calls with the *same* transform instance should produce
    different outputs (the per-sample mode RNG advances each call)."""
    from refshift.jitter import make_random_reference_transform
    g = _make_graph()
    t = make_random_reference_transform(REFERENCE_MODES, graph=g, random_state=42)
    X, y = _make_synthetic_batch(B=16)
    Xt = torch.from_numpy(X)
    yt = torch.from_numpy(y)
    out1, _ = t(Xt, yt)
    out2, _ = t(Xt, yt)
    # Extremely unlikely to draw identical mode sequences twice in a row
    # for 16 samples uniform over 7 options.
    assert not np.allclose(out1.numpy(), out2.numpy())


def test_transform_seed_reproducibility():
    """Two transforms with the same random_state should produce identical
    outputs given identical inputs."""
    from refshift.jitter import make_random_reference_transform
    g = _make_graph()
    t1 = make_random_reference_transform(REFERENCE_MODES, graph=g, random_state=123)
    t2 = make_random_reference_transform(REFERENCE_MODES, graph=g, random_state=123)
    X, y = _make_synthetic_batch(B=16)
    Xt = torch.from_numpy(X)
    yt = torch.from_numpy(y)
    out1, _ = t1(Xt, yt)
    out2, _ = t2(Xt, yt)
    np.testing.assert_allclose(out1.numpy(), out2.numpy(), atol=1e-6)


# ---- End-to-end smoke (synthetic, CPU) --------------------------------------

@pytest.mark.parametrize("arch", ["shallow", "eegnet"])
def test_make_dl_model_with_transforms_runs(arch):
    """make_dl_model with transforms argument: AugmentedDataLoader is wired
    in and a 2-epoch fit + predict completes on synthetic data."""
    from refshift.dl import make_dl_model
    from refshift.jitter import make_random_reference_transform

    g = _make_graph()
    transform = make_random_reference_transform(
        REFERENCE_MODES, graph=g, random_state=0,
    )

    rng = np.random.default_rng(0)
    N, C, T = 16, len(_CH_NAMES), 500
    X = rng.standard_normal((N, C, T)).astype(np.float32)
    y = rng.integers(0, 4, size=(N,)).astype(np.int64)

    clf = make_dl_model(
        model=arch,
        n_channels=C, n_classes=4, n_times=T, sfreq=250.0,
        seed=0, max_epochs=2, batch_size=4, device="cpu",
        transforms=[transform],
    )
    clf.fit(X, y)
    y_pred = clf.predict(X)
    assert y_pred.shape == (N,)
    assert 0 <= y_pred.min() and y_pred.max() < 4
