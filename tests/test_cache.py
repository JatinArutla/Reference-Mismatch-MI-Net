"""Unit tests for the preprocessed-tensor disk cache in refshift.dl.

Synthetic-only — no MOABB / network. Skipped without Phase 2 extras.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("braindecode")
pytest.importorskip("torch")
pytest.importorskip("skorch")


def _params(**overrides):
    base = dict(
        dataset_id="iv2a",
        subject=3,
        l_freq=8.0,
        h_freq=32.0,
        ems_factor_new=1e-3,
        ems_init_block_size=1000,
        trial_start_offset_s=0.0,
        trial_stop_offset_s=0.0,
    )
    base.update(overrides)
    return base


def _synthetic(seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((12, 22, 1000)).astype(np.float32)
    y = rng.integers(0, 4, size=12).astype(np.int64)
    metadata = pd.DataFrame({
        "session": ["0train"] * 6 + ["1test"] * 6,
        "run": ["0"] * 12,
        "subject": [3] * 12,
    })
    return X, y, metadata, 250.0, [f"Ch{i}" for i in range(22)]


# ---- _cache_path: stable hashing across param changes -----------------------

def test_cache_path_deterministic():
    from refshift.dl import _cache_path
    with tempfile.TemporaryDirectory() as tmp:
        a = _cache_path(tmp, _params())
        b = _cache_path(tmp, _params())
        assert a == b


def test_cache_path_differs_by_dataset_subject_filter_offsets():
    """Each parameter that affects the preprocessed output must produce a
    distinct path."""
    from refshift.dl import _cache_path
    with tempfile.TemporaryDirectory() as tmp:
        ref = _cache_path(tmp, _params())
        for override in (
            {"dataset_id": "openbmi"},
            {"subject": 7},
            {"l_freq": 4.0},
            {"h_freq": 38.0},
            {"trial_start_offset_s": -0.5},
        ):
            assert _cache_path(tmp, _params(**override)) != ref


def test_cache_path_directory_layout():
    """Files live under cache_dir/<dataset_id>/sub-<NNN>/<hash>.npz."""
    from refshift.dl import _cache_path
    with tempfile.TemporaryDirectory() as tmp:
        path = _cache_path(tmp, _params(dataset_id="openbmi", subject=53))
        assert path.startswith(os.path.join(tmp, "openbmi", "sub-053"))
        assert path.endswith(".npz")
        # The subdirectory should exist after the call.
        assert os.path.isdir(os.path.dirname(path))


# ---- load_dl_data round-trip via the cache ---------------------------------

def test_cache_round_trip_via_load_dl_data(monkeypatch):
    """End-to-end: a populated cache file is read back without invoking
    braindecode.

    We populate the cache by writing the exact .npz layout load_dl_data
    expects, then call load_dl_data with cache_dir set and a sentinel
    monkeypatched into braindecode.datasets.MOABBDataset that fails the test
    if invoked.
    """
    from refshift import dl as dl_mod

    with tempfile.TemporaryDirectory() as tmp:
        params = _params()
        path = dl_mod._cache_path(tmp, params)
        X, y, metadata, sfreq, ch_names = _synthetic()

        # Mirror the exact .npz layout load_dl_data writes on cache miss.
        np.savez(
            path[:-len(".npz")],
            X=X, y=y, sfreq=np.float64(sfreq),
            metadata_session=metadata["session"].to_numpy(),
            metadata_run=metadata["run"].to_numpy(),
            metadata_subject=metadata["subject"].to_numpy(),
            ch_names=np.asarray(ch_names, dtype=object),
        )

        def _fail(*a, **kw):
            raise AssertionError("Cache hit should bypass braindecode entirely")

        import braindecode.datasets as bd_ds
        monkeypatch.setattr(bd_ds, "MOABBDataset", _fail)

        out = dl_mod.load_dl_data(
            params["dataset_id"], params["subject"],
            l_freq=params["l_freq"], h_freq=params["h_freq"],
            cache_dir=tmp,
        )
        X2, y2, metadata2, sfreq2, ch2 = out
        np.testing.assert_array_equal(X, X2)
        np.testing.assert_array_equal(y, y2)
        assert sfreq == sfreq2
        assert ch_names == ch2


def test_cache_disabled_calls_braindecode(monkeypatch):
    """When cache_dir=None, braindecode is invoked even when a cache
    happens to be sitting in the default location."""
    from refshift import dl as dl_mod

    called = {"flag": False}

    def _record_call(*a, **kw):
        called["flag"] = True
        raise RuntimeError("intentional early-exit from monkeypatched MOABBDataset")

    import braindecode.datasets as bd_ds
    monkeypatch.setattr(bd_ds, "MOABBDataset", _record_call)

    with pytest.raises(RuntimeError, match="intentional"):
        dl_mod.load_dl_data("iv2a", subject=3, cache_dir=None)
    assert called["flag"]


def test_corrupt_cache_falls_through_to_preprocess(monkeypatch):
    """If a cache file is corrupt (not a valid .npz), load_dl_data should
    fall through to the preprocess path rather than crash."""
    from refshift import dl as dl_mod

    with tempfile.TemporaryDirectory() as tmp:
        params = _params()
        path = dl_mod._cache_path(tmp, params)
        with open(path, "wb") as f:
            f.write(b"not a valid npz")

        called = {"flag": False}

        def _record_call(*a, **kw):
            called["flag"] = True
            raise RuntimeError("fall-through reached")

        import braindecode.datasets as bd_ds
        monkeypatch.setattr(bd_ds, "MOABBDataset", _record_call)

        with pytest.raises(RuntimeError, match="fall-through"):
            dl_mod.load_dl_data(
                params["dataset_id"], params["subject"],
                l_freq=params["l_freq"], h_freq=params["h_freq"],
                cache_dir=tmp,
            )
        assert called["flag"]


# ---- run_mismatch / run_mismatch_jitter signature plumbing -----------------

def test_run_mismatch_signature_includes_dl_cache_dir():
    from refshift.experiments import run_mismatch
    import inspect
    sig = inspect.signature(run_mismatch)
    assert "dl_cache_dir" in sig.parameters
    assert sig.parameters["dl_cache_dir"].default is None


def test_run_mismatch_jitter_signature_includes_dl_cache_dir():
    from refshift.experiments import run_mismatch_jitter
    import inspect
    sig = inspect.signature(run_mismatch_jitter)
    assert "dl_cache_dir" in sig.parameters
    assert sig.parameters["dl_cache_dir"].default is None


# ---- Cache file shape: round trip preserves all fields ---------------------
