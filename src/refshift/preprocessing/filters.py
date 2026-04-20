
from __future__ import annotations
from fractions import Fraction
import numpy as np
from scipy.signal import butter, sosfiltfilt, resample_poly


def bandpass_filter_trials(X: np.ndarray, fs: float, band: tuple[float,float], order: int = 4) -> np.ndarray:
    lo, hi = band
    if lo <= 0 and hi >= fs/2:
        return X.astype(np.float32, copy=False)
    sos = butter(order, [lo, hi], btype='bandpass', fs=fs, output='sos')
    Xf = sosfiltfilt(sos, X, axis=-1)
    return Xf.astype(np.float32)


def resample_trials(X: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    if abs(fs_in - fs_out) < 1e-9:
        return X.astype(np.float32, copy=False)
    frac = Fraction(fs_out / fs_in).limit_denominator(1000)
    up, down = frac.numerator, frac.denominator
    Xr = resample_poly(X, up, down, axis=-1)
    return Xr.astype(np.float32)
