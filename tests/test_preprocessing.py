"""
Tests for refshift.preprocessing.

Covers:
    - shape + dtype preservation
    - in-band sinusoid amplitude is roughly preserved
    - out-of-band sinusoid is heavily attenuated
    - zero-phase property (symmetric filter lag)
    - no NaN/Inf introduction
    - invalid inputs raise clearly
    - SubjectData wrapper preserves all fields and applies bandpass in place
    - integration with real loader output

Runtime on Kaggle: ~30 seconds.
"""

from __future__ import annotations

import sys
import traceback
from typing import Tuple

import numpy as np

from loader import load_subject
from preprocessing import (
    DEFAULT_BANDPASS_HZ,
    DEFAULT_FILTER_ORDER,
    bandpass_filter,
    bandpass_subject_data,
)


# ============================================================================
# Helpers
# ============================================================================

def _sinusoid(freq_hz: float, fs: float, duration_s: float, amplitude: float = 1.0):
    t = np.arange(int(round(duration_s * fs))) / fs
    return (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def _rms(x):
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


# ============================================================================
# Tests
# ============================================================================

def test_shape_and_dtype():
    """bandpass_filter preserves shape and float32 dtype."""
    X = np.random.randn(10, 22, 1000).astype(np.float32)
    Y = bandpass_filter(X, fs=250.0, band=(8.0, 32.0))
    assert Y.shape == X.shape, f"shape changed: {X.shape} -> {Y.shape}"
    assert Y.dtype == np.float32, f"dtype changed: {Y.dtype}"
    assert Y.flags["C_CONTIGUOUS"], "output not C-contiguous"
    assert np.isfinite(Y).all(), "NaN or Inf introduced"
    print("  shape/dtype preservation: OK")


def test_rank_of_inputs():
    """Works on 1D [T], 2D [C, T], and 3D [N, C, T]."""
    fs = 250.0
    for shape in [(1000,), (22, 1000), (10, 22, 1000)]:
        X = np.random.randn(*shape).astype(np.float32)
        Y = bandpass_filter(X, fs=fs, band=(8.0, 32.0))
        assert Y.shape == X.shape, f"shape changed for input {shape}"
    print("  1D/2D/3D input support: OK")


def test_inband_passes():
    """A 15 Hz sinusoid survives an 8-32 Hz bandpass with most amplitude intact."""
    fs = 250.0
    x = _sinusoid(15.0, fs, duration_s=4.0)  # 1000 samples
    y = bandpass_filter(x, fs=fs, band=(8.0, 32.0))
    # Ignore first and last 50 samples to avoid edge-effect smearing
    rms_in = _rms(x[50:-50])
    rms_out = _rms(y[50:-50])
    ratio = rms_out / rms_in
    assert 0.85 < ratio < 1.15, (
        f"in-band 15Hz amplitude ratio = {ratio:.3f}, expected ~1.0"
    )
    print(f"  in-band (15 Hz) passes: ratio={ratio:.3f}")


def test_outband_attenuated():
    """A 2 Hz sinusoid is heavily attenuated by an 8-32 Hz bandpass."""
    fs = 250.0
    x = _sinusoid(2.0, fs, duration_s=4.0)
    y = bandpass_filter(x, fs=fs, band=(8.0, 32.0))
    rms_in = _rms(x[50:-50])
    rms_out = _rms(y[50:-50])
    ratio = rms_out / rms_in
    assert ratio < 0.05, (
        f"out-of-band 2Hz amplitude ratio = {ratio:.4f}, expected < 0.05"
    )

    # Same for a 60 Hz sinusoid (line noise territory)
    x2 = _sinusoid(60.0, fs, duration_s=4.0)
    y2 = bandpass_filter(x2, fs=fs, band=(8.0, 32.0))
    ratio2 = _rms(y2[50:-50]) / _rms(x2[50:-50])
    assert ratio2 < 0.05, (
        f"out-of-band 60Hz amplitude ratio = {ratio2:.4f}, expected < 0.05"
    )
    print(f"  out-of-band attenuation: 2Hz={ratio:.4f}, 60Hz={ratio2:.4f}")


def test_zero_phase():
    """sosfiltfilt is zero-phase: cross-correlation peak is at lag 0."""
    fs = 250.0
    # Pulse in the middle of the signal; bandpassed output should also
    # have its peak energy near the middle (no phase shift).
    x = np.zeros(1000, dtype=np.float32)
    x[500] = 1.0
    y = bandpass_filter(x, fs=fs, band=(8.0, 32.0))
    peak_idx = int(np.argmax(np.abs(y)))
    # Allow a few samples tolerance due to the impulse response shape
    assert abs(peak_idx - 500) < 5, (
        f"zero-phase check: peak at idx {peak_idx}, expected near 500"
    )
    print(f"  zero-phase property: peak at idx {peak_idx}")


def test_invalid_inputs_raise():
    """Invalid bands and order values raise ValueError clearly."""
    X = np.random.randn(10, 22, 1000).astype(np.float32)

    invalid_configs = [
        {"band": (0.0, 32.0)},       # zero lo
        {"band": (8.0, 130.0)},      # hi >= Nyquist for fs=250
        {"band": (32.0, 8.0)},       # lo > hi
        {"band": (-1.0, 32.0)},      # negative
        {"order": 0},                # zero order
    ]
    for cfg in invalid_configs:
        kwargs = {"fs": 250.0, "band": (8.0, 32.0), "order": 4}
        kwargs.update(cfg)
        try:
            bandpass_filter(X, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"Invalid config did not raise: {cfg}")

    # Too-short signal
    short = np.random.randn(10).astype(np.float32)
    try:
        bandpass_filter(short, fs=250.0, band=(8.0, 32.0), order=4)
    except ValueError:
        pass
    else:
        raise AssertionError("Too-short signal did not raise")

    print("  invalid inputs raise: OK")


def test_subject_data_wrapper():
    """bandpass_subject_data applies filter, preserves non-X fields and None fields."""
    # Session-split dataset
    data = load_subject("iv2a", 1)
    data_bp = bandpass_subject_data(data, band=(8.0, 32.0))

    assert data_bp.dataset_id == data.dataset_id
    assert data_bp.subject == data.subject
    assert data_bp.ch_names == data.ch_names
    assert data_bp.sfreq == data.sfreq
    assert data_bp.X_all is None  # session-split; X_all stays None
    assert data_bp.y_all is None
    assert data_bp.X_train.shape == data.X_train.shape
    assert data_bp.X_test.shape == data.X_test.shape
    assert np.array_equal(data_bp.y_train, data.y_train)
    assert np.array_equal(data_bp.y_test, data.y_test)

    # Check the filter actually did something
    diff = np.abs(data_bp.X_train - data.X_train).mean()
    assert diff > 0, "bandpass had no effect on X_train"

    # DC drift should be gone (mean near zero per trial per channel)
    # after an 8-32 Hz bandpass
    mean_per_trial_ch = data_bp.X_train.mean(axis=-1)  # [N, C]
    max_abs_mean = np.abs(mean_per_trial_ch).max()
    assert max_abs_mean < 1e-6, (
        f"post-bandpass per-trial-per-channel max mean = {max_abs_mean:.3g}, "
        f"expected ~0 (DC removed)"
    )
    print(f"  SubjectData wrapper (iv2a): diff mean={diff:.3g}, "
          f"post-bp DC max={max_abs_mean:.3g}")


def test_subject_data_wrapper_single_session():
    """Single-session dataset (Cho2017) bandpass preserves X_all, sets X_train/X_test None."""
    data = load_subject("cho2017", 1)
    data_bp = bandpass_subject_data(data, band=(8.0, 32.0))
    assert data_bp.X_train is None
    assert data_bp.X_test is None
    assert data_bp.X_all is not None
    assert data_bp.X_all.shape == data.X_all.shape
    assert np.isfinite(data_bp.X_all).all()
    diff = np.abs(data_bp.X_all - data.X_all).mean()
    assert diff > 0
    print(f"  SubjectData wrapper (cho2017, single session): diff mean={diff:.3g}")


def test_defaults_match_paper_convention():
    """Sanity: default band and order match what we commit to in the paper."""
    assert DEFAULT_BANDPASS_HZ == (8.0, 32.0)
    assert DEFAULT_FILTER_ORDER == 4
    print(f"  defaults: band={DEFAULT_BANDPASS_HZ}, order={DEFAULT_FILTER_ORDER}")


# ============================================================================
# Main
# ============================================================================

TESTS = [
    ("shape/dtype preservation",      test_shape_and_dtype),
    ("1D/2D/3D input support",        test_rank_of_inputs),
    ("in-band passes",                test_inband_passes),
    ("out-of-band attenuated",        test_outband_attenuated),
    ("zero-phase",                    test_zero_phase),
    ("invalid inputs raise",          test_invalid_inputs_raise),
    ("SubjectData wrapper (session)", test_subject_data_wrapper),
    ("SubjectData wrapper (single)",  test_subject_data_wrapper_single_session),
    ("defaults",                      test_defaults_match_paper_convention),
]


def run_all():
    passed, failed = [], []
    for name, fn in TESTS:
        print(f"\n===== {name} =====")
        try:
            fn()
            passed.append(name)
        except Exception as e:
            traceback.print_exc()
            failed.append((name, repr(e)))
    print("\n" + "=" * 72)
    print(f"PASSED ({len(passed)}):")
    for n in passed:
        print(f"  PASS  {n}")
    if failed:
        print(f"\nFAILED ({len(failed)}):")
        for n, err in failed:
            print(f"  FAIL  {n}: {err}")
    else:
        print("\nAll preprocessing tests PASSED.")
    return failed


if __name__ == "__main__":
    failed = run_all()
    sys.exit(1 if failed else 0)
