"""
refshift.preprocessing — bandpass filtering for epoched EEG trials.

Applies zero-phase Butterworth bandpass via scipy.signal.sosfiltfilt along
the time axis. Standardization is separate (chunk 3) because it depends on
the reference operator.

Conventions:
    Default band:  8.0 - 32.0 Hz
    Filter order:  4 (applied twice via sosfiltfilt → effective 8th-order,
                      zero-phase)
    Input shape:   [..., T] along last axis (works for [N, C, T], [C, T],
                   and [T])

Used as:
    from refshift.preprocessing import bandpass_filter
    X_bp = bandpass_filter(X, fs=250.0, band=(8.0, 32.0))

    # or via SubjectData convenience wrapper:
    from refshift.preprocessing import bandpass_subject_data
    data_bp = bandpass_subject_data(data, band=(8.0, 32.0))
"""

from __future__ import annotations

from dataclasses import replace
from typing import Tuple

import numpy as np
from scipy.signal import butter, sosfiltfilt


# ============================================================================
# Defaults
# ============================================================================

DEFAULT_BANDPASS_HZ = (8.0, 32.0)
DEFAULT_FILTER_ORDER = 4


# ============================================================================
# Core filter
# ============================================================================

def bandpass_filter(
    X: np.ndarray,
    fs: float,
    band: Tuple[float, float] = DEFAULT_BANDPASS_HZ,
    order: int = DEFAULT_FILTER_ORDER,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass along the last (time) axis.

    Args:
        X:     input array, any shape, time axis is last.
        fs:    sampling rate in Hz.
        band:  (low, high) passband in Hz.
        order: Butterworth order. sosfiltfilt applies forward + backward so
               the effective magnitude response is |H(f)|^2 with
               -3 dB at the passband edges shifted slightly inward.

    Returns:
        Filtered array, same shape and dtype as X, C-contiguous.

    Raises:
        ValueError if the band is invalid for the given fs, or if the signal
        is too short for the filter padding (sosfiltfilt needs at least
        3 * (2 * order + 1) samples on the time axis).
    """
    lo, hi = band
    if lo <= 0 or hi <= 0:
        raise ValueError(f"Band must be strictly positive: {band}")
    if hi >= fs / 2.0:
        raise ValueError(f"High cutoff {hi} Hz >= Nyquist {fs/2} Hz")
    if lo >= hi:
        raise ValueError(f"Low cutoff {lo} >= high cutoff {hi}")
    if order < 1:
        raise ValueError(f"Order must be >= 1, got {order}")
    if X.shape[-1] < 3 * (2 * order + 1):
        raise ValueError(
            f"Signal too short for order-{order} filter: {X.shape[-1]} "
            f"samples, need at least {3 * (2 * order + 1)}"
        )

    sos = butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
    out = sosfiltfilt(sos, X, axis=-1)
    return np.ascontiguousarray(out, dtype=X.dtype)


# ============================================================================
# SubjectData convenience wrapper
# ============================================================================

def bandpass_subject_data(
    data,
    band: Tuple[float, float] = DEFAULT_BANDPASS_HZ,
    order: int = DEFAULT_FILTER_ORDER,
):
    """Apply bandpass to every X_* array in a SubjectData. Returns a new
    SubjectData; input is unchanged.

    Skips None entries (e.g. X_test is None for single-session datasets).
    """
    kwargs = {}
    for name in ("X_train", "X_test", "X_all"):
        val = getattr(data, name)
        if val is not None:
            kwargs[name] = bandpass_filter(val, data.sfreq, band, order)
    return replace(data, **kwargs)
