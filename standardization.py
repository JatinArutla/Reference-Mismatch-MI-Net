"""
refshift.standardization — two standardization protocols.

    mechanistic — per-trial, per-channel z-score over time.
                  Each trial normalized independently. The same transform is
                  applied to train and test trials. Used for mechanism studies
                  where we want to isolate the structural effect of the
                  reference operator from dataset/subject scale variation.

    deployment  — per-channel z-score using statistics computed ONLY on the
                  training set (averaged over trials and time), then applied
                  to both train and test. This is the realistic deployment
                  pipeline and the one used for headline accuracy numbers.

Both operate on [N, C, T] arrays along the time axis.
Mu and sd are per-channel in the deployment protocol.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


DEFAULT_EPS = 1e-6


# ============================================================================
# Mechanistic (per-trial z-score)
# ============================================================================

def standardize_mechanistic(X: np.ndarray, eps: float = DEFAULT_EPS) -> np.ndarray:
    """Per-trial, per-channel z-score over the time axis.

    For each trial n, each channel c:
        Y[n, c, t] = (X[n, c, t] - mean_t(X[n, c, :])) / max(std_t(X[n, c, :]), eps)

    Args:
        X:   [N, C, T] or [C, T] float array
        eps: small floor on std to prevent divide-by-zero

    Returns:
        Output array, same shape, float32, C-contiguous.
    """
    if X.ndim not in (2, 3):
        raise ValueError(f"Expected [C,T] or [N,C,T], got {X.shape}")
    mu = X.mean(axis=-1, keepdims=True)
    sd = X.std(axis=-1, keepdims=True)
    out = (X - mu) / np.maximum(sd, eps)
    return np.ascontiguousarray(out, dtype=np.float32)


# ============================================================================
# Deployment (fit on train, apply to both)
# ============================================================================

def fit_standardizer(
    X_train: np.ndarray,
    eps: float = DEFAULT_EPS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-channel (mu, sd) from training trials.

    Statistics are pooled across trials and time — one number per channel.

    Args:
        X_train: [N, C, T]
        eps:     small floor on std

    Returns:
        (mu, sd) each of shape [1, C, 1], float32.
    """
    if X_train.ndim != 3:
        raise ValueError(f"fit_standardizer expects [N,C,T], got {X_train.shape}")
    mu = X_train.mean(axis=(0, 2), keepdims=True)
    sd = X_train.std(axis=(0, 2), keepdims=True)
    sd = np.maximum(sd, eps)
    return mu.astype(np.float32), sd.astype(np.float32)


def apply_standardizer(
    X: np.ndarray,
    mu: np.ndarray,
    sd: np.ndarray,
) -> np.ndarray:
    """Apply a previously fitted per-channel (mu, sd) to an array.

    Args:
        X:  [N, C, T] or [C, T]
        mu: [1, C, 1]
        sd: [1, C, 1]

    Returns:
        Standardized output, same shape as X, float32.
    """
    if X.ndim == 2:
        # Expand mu, sd to broadcast against [C, T]
        mu_b = mu.reshape(-1, 1)
        sd_b = sd.reshape(-1, 1)
        out = (X - mu_b) / sd_b
    elif X.ndim == 3:
        out = (X - mu) / sd
    else:
        raise ValueError(f"Expected [C,T] or [N,C,T], got {X.shape}")
    return np.ascontiguousarray(out, dtype=np.float32)
