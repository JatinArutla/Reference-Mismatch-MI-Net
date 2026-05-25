"""Tests for the DL-only `normalization` knob added in v0.17.

These deliberately avoid braindecode / torch / mne so they run in a light
environment: they exercise the constant, the cache key, the pure-numpy
standardiser, and the runner signatures via inspect.
"""

from __future__ import annotations

import inspect

import numpy as np


# ---- constant + validation -------------------------------------------------

def test_normalizations_constant():
    from refshift.data import NORMALIZATIONS
    assert NORMALIZATIONS == ("zscore", "ems", "none")


def test_cache_key_includes_normalization():
    """Without normalization in the cache key, a zscore run would silently
    read an ems-normalised (or unnormalised) cache entry."""
    from refshift.data import _CACHE_KEY_PARAMS
    assert "normalization" in _CACHE_KEY_PARAMS


def test_load_dl_data_default_is_zscore():
    from refshift.data import load_dl_data
    sig = inspect.signature(load_dl_data)
    assert sig.parameters["normalization"].default == "zscore"


# ---- _zscore_standardize numerics ------------------------------------------

def test_zscore_standardize_per_channel_moments():
    from refshift.data import _zscore_standardize
    rng = np.random.default_rng(0)
    data = rng.normal(5.0, 3.0, size=(4, 1000))  # (C, T), non-zero mean/var
    out = _zscore_standardize(data)
    assert np.allclose(out.mean(axis=1), 0.0, atol=1e-9)
    assert np.allclose(out.std(axis=1), 1.0, atol=1e-6)


def test_zscore_standardize_flat_channel_is_safe():
    """A constant channel has std 0; the eps floor must map it to all-zeros
    rather than NaN/inf."""
    from refshift.data import _zscore_standardize
    flat = np.ones((2, 50))
    out = _zscore_standardize(flat)
    assert np.all(np.isfinite(out))
    assert np.allclose(out, 0.0)


# ---- DL runner signatures --------------------------------------------------

def test_main_dl_runners_default_zscore():
    """The four general DL runners default to zscore."""
    from refshift.experiments import (
        run_mismatch,
        run_mismatch_jitter,
        run_lofo_matrix,
        run_bandpass_mismatch,
    )
    for fn in (run_mismatch, run_mismatch_jitter, run_lofo_matrix, run_bandpass_mismatch):
        params = inspect.signature(fn).parameters
        assert "normalization" in params, fn.__name__
        assert params["normalization"].default == "zscore", fn.__name__


def test_pre_ems_runners_default_ems():
    """The two pre-EMS ordering controls default to ems, not zscore: they
    exist to probe EMS's (adaptive) non-commutativity with reference operators,
    so defaulting them to the normalizer they test keeps the control coherent
    and reproduces the v0.16 pre-EMS results."""
    from refshift.experiments import run_pre_ems_diagonal, run_pre_ems_mismatch
    for fn in (run_pre_ems_diagonal, run_pre_ems_mismatch):
        params = inspect.signature(fn).parameters
        assert "normalization" in params, fn.__name__
        assert params["normalization"].default == "ems", fn.__name__


def test_calibrate_csp_lda_has_no_normalization():
    """normalization is DL-only; the CSP calibration utility must not gain it."""
    from refshift.experiments import calibrate_csp_lda
    assert "normalization" not in inspect.signature(calibrate_csp_lda).parameters


def test_make_csp_lda_pipeline_has_no_normalization():
    """CSP+LDA stays byte-for-byte the MOABB-calibrated pipeline: no
    normalization argument, no z-score step."""
    from refshift.model import make_csp_lda_pipeline
    assert "normalization" not in inspect.signature(make_csp_lda_pipeline).parameters


# ---- regression: dl_lr default unified to None -----------------------------

def test_pre_ems_diagonal_dl_lr_default_is_none():
    """run_pre_ems_diagonal previously hardcoded dl_lr=6.25e-4 (shallow's LR),
    silently using the wrong LR for eegnet/atcnet. It must default to None so
    make_dl_model picks the per-model LR, matching the other runners."""
    from refshift.experiments import run_pre_ems_diagonal
    sig = inspect.signature(run_pre_ems_diagonal)
    assert sig.parameters["dl_lr"].default is None
