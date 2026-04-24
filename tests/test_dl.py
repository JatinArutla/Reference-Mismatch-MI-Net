"""Unit tests for refshift.dl.

The whole file is skipped if braindecode / torch / skorch are not installed,
so Phase 1 CI continues to pass without the ``[dl]`` extras. When the extras
are present, these run in a few seconds on CPU with no MOABB / network
dependency — they only verify construction + a synthetic-data forward pass.
"""

from __future__ import annotations

import numpy as np
import pytest

# Skip whole module if Phase 2 extras aren't installed.
pytest.importorskip("braindecode")
pytest.importorskip("torch")
pytest.importorskip("skorch")


def test_moabb_code_mapping():
    from refshift.dl import _moabb_code
    assert _moabb_code("iv2a") == "BNCI2014_001"
    assert _moabb_code("openbmi") == "Lee2019_MI"
    assert _moabb_code("cho2017") == "Cho2017"
    assert _moabb_code("dreyer2023") == "Dreyer2023"
    assert _moabb_code("IV2A") == "BNCI2014_001"  # case insensitive


def test_moabb_code_unknown_raises():
    from refshift.dl import _moabb_code
    with pytest.raises(ValueError):
        _moabb_code("unknown_dataset")


def test_make_dl_model_unknown_raises():
    from refshift.dl import make_dl_model
    with pytest.raises(ValueError):
        make_dl_model(
            model="not_a_real_model",
            n_channels=22, n_classes=4, n_times=1000, sfreq=250.0,
        )


def test_make_dl_model_shallow_constructs():
    from refshift.dl import make_dl_model
    clf = make_dl_model(
        model="shallow",
        n_channels=22, n_classes=4, n_times=1000, sfreq=250.0,
        seed=0, max_epochs=2, batch_size=4, device="cpu",
    )
    assert clf is not None


def test_make_dl_model_eegnet_constructs():
    from refshift.dl import make_dl_model
    clf = make_dl_model(
        model="eegnet",
        n_channels=22, n_classes=4, n_times=1000, sfreq=250.0,
        seed=0, max_epochs=2, batch_size=4, device="cpu",
    )
    assert clf is not None


@pytest.mark.parametrize("arch", ["shallow", "eegnet"])
def test_fit_predict_on_synthetic(arch):
    """End-to-end smoke: fit 2 epochs on 16 synthetic trials, predict shape ok."""
    from refshift.dl import make_dl_model

    rng = np.random.default_rng(0)
    N, C, T = 16, 22, 1000
    X = rng.standard_normal((N, C, T)).astype(np.float32)
    y = rng.integers(0, 4, size=(N,)).astype(np.int64)

    clf = make_dl_model(
        model=arch,
        n_channels=C, n_classes=4, n_times=T, sfreq=250.0,
        seed=0, max_epochs=2, batch_size=4, device="cpu",
    )
    clf.fit(X, y)

    y_pred = clf.predict(X)
    assert y_pred.shape == (N,)
    assert y_pred.dtype.kind in ("i", "u")
    assert 0 <= y_pred.min() and y_pred.max() < 4

    acc = clf.score(X, y)
    assert 0.0 <= acc <= 1.0


def test_seed_reproducibility_shallow():
    """Two models with same seed should produce identical predictions."""
    from refshift.dl import make_dl_model

    rng = np.random.default_rng(0)
    N, C, T = 16, 22, 500
    X = rng.standard_normal((N, C, T)).astype(np.float32)
    y = rng.integers(0, 4, size=(N,)).astype(np.int64)

    def _fit_predict():
        clf = make_dl_model(
            model="shallow",
            n_channels=C, n_classes=4, n_times=T, sfreq=250.0,
            seed=123, max_epochs=2, batch_size=4, device="cpu",
        )
        clf.fit(X, y)
        # Use predict_proba for a stricter equivalence check than argmax.
        return clf.predict_proba(X)

    p1 = _fit_predict()
    p2 = _fit_predict()
    # CPU-only with fixed seed should be reproducible to high precision.
    np.testing.assert_allclose(p1, p2, atol=1e-5)
