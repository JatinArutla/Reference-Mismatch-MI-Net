"""
refshift.reference_ops — six reference operators + graph construction.

The paper studies how re-referencing creates a systematic distribution shift.
This module implements the six reference operators that define the 6x6
mismatch matrix:

    native     — identity (use the dataset's native reference)
    car        — common average reference: subtract mean over channels
    median     — robust CAR: subtract median over channels
    gs         — Gram-Schmidt: subtract projection onto leave-one-out mean
                 (per-trial, per-channel, in the time domain)
    laplacian  — subtract mean of k=4 nearest neighbors (spatial, per-channel)
    bipolar    — subtract single nearest neighbor (spatial, per-channel)

The spatial operators (laplacian, bipolar) need a graph built from channel
positions. We use MNE's standard_1005 montage as the positions source,
which covers every channel in the four datasets we use. Graphs are built
once per dataset and reused.

Conventions:
    X shape: [N, C, T] or [C, T]. Output has the same shape and is float32,
             C-contiguous.
    Graphs: integer index arrays. laplacian_idx is [C, k]; bipolar_idx is [C].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


# ============================================================================
# Channel positions and graph construction
# ============================================================================

def get_channel_positions(
    ch_names: List[str],
    montage: str = "standard_1005",
) -> np.ndarray:
    """Return [C, 3] xyz positions for the given channel names.

    Uses an MNE standard montage as the position source. Raises if any
    channel is unknown to the montage.
    """
    import mne
    mont = mne.channels.make_standard_montage(montage)
    pos = mont.get_positions()["ch_pos"]
    missing = [ch for ch in ch_names if ch not in pos]
    if missing:
        raise ValueError(
            f"Channels not in montage {montage!r}: {missing}"
        )
    return np.array([pos[ch] for ch in ch_names], dtype=np.float64)


def _pairwise_distances(xyz: np.ndarray) -> np.ndarray:
    """Euclidean distance matrix [C, C]. Diagonal is np.inf so it won't be
    picked as a nearest neighbor of itself."""
    d = np.sqrt(((xyz[:, None, :] - xyz[None, :, :]) ** 2).sum(axis=-1))
    np.fill_diagonal(d, np.inf)
    return d


@dataclass
class DatasetGraph:
    """Pre-computed neighbor indices for a dataset's channel set."""
    ch_names: List[str]
    laplacian_idx: np.ndarray  # [C, k] int
    bipolar_idx:   np.ndarray  # [C]    int
    k: int
    montage: str


def build_graph(
    ch_names: List[str],
    k: int = 4,
    montage: str = "standard_1005",
) -> DatasetGraph:
    """Build laplacian k-NN and bipolar nearest-neighbor graphs for a
    channel layout.

    Laplacian: laplacian_idx[c] = indices of k channels nearest to c (in
               Euclidean distance over the montage xyz) excluding c itself.
    Bipolar:   bipolar_idx[c]   = index of the single nearest channel to c.
    """
    xyz = get_channel_positions(ch_names, montage=montage)
    d = _pairwise_distances(xyz)
    lap = np.argsort(d, axis=1)[:, :k].astype(np.int64)
    bip = np.argmin(d, axis=1).astype(np.int64)
    return DatasetGraph(
        ch_names=list(ch_names),
        laplacian_idx=lap,
        bipolar_idx=bip,
        k=k,
        montage=montage,
    )


# ============================================================================
# Reference operators (each accepts [N, C, T] or [C, T])
# ============================================================================

def _ensure(X: np.ndarray) -> np.ndarray:
    if X.ndim not in (2, 3):
        raise ValueError(f"Expected [C,T] or [N,C,T], got {X.shape}")
    return X.astype(np.float32, copy=False)


def native_ref(X: np.ndarray) -> np.ndarray:
    """Identity: returns a fresh copy of X."""
    return np.ascontiguousarray(_ensure(X).copy())


def car_ref(X: np.ndarray) -> np.ndarray:
    """Common average reference: subtract mean across channels per timepoint."""
    X = _ensure(X)
    axis = 1 if X.ndim == 3 else 0
    out = X - X.mean(axis=axis, keepdims=True)
    return np.ascontiguousarray(out, dtype=np.float32)


def median_ref(X: np.ndarray) -> np.ndarray:
    """Median reference: subtract median across channels per timepoint."""
    X = _ensure(X)
    axis = 1 if X.ndim == 3 else 0
    out = X - np.median(X, axis=axis, keepdims=True)
    return np.ascontiguousarray(out, dtype=np.float32)


def gs_ref(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Gram-Schmidt reference: per-trial per-channel, orthogonalize each
    channel's time series against the mean of the OTHER channels.

    For each (trial n, channel c), computes the scalar projection coefficient
    of X[n,c,:] onto r[n,c,:] = mean_{c'!=c} X[n,c',:] in the time domain,
    then subtracts that component:

        Y[n,c,:] = X[n,c,:] - <X[n,c,:], r[n,c,:]> / <r[n,c,:], r[n,c,:]> * r[n,c,:]

    This makes Y[n,c,:] orthogonal (in the time-axis inner product) to the
    leave-one-out channel mean for that trial.
    """
    X = _ensure(X)
    squeeze = (X.ndim == 2)
    if squeeze:
        X = X[None, ...]
    N, C, T = X.shape
    s = X.sum(axis=1, keepdims=True)              # [N, 1, T]
    r = (s - X) / max(C - 1, 1)                    # [N, C, T] leave-one-out mean
    num = np.sum(X * r, axis=2, keepdims=True)     # [N, C, 1]
    den = np.sum(r * r, axis=2, keepdims=True) + eps
    Y = X - (num / den) * r
    Y = Y[0] if squeeze else Y
    return np.ascontiguousarray(Y, dtype=np.float32)


def laplacian_ref(X: np.ndarray, laplacian_idx: np.ndarray) -> np.ndarray:
    """Subtract mean of k nearest spatial neighbors per channel.

    Args:
        X: [N, C, T] or [C, T]
        laplacian_idx: [C, k] int, laplacian_idx[c] = indices of c's k neighbors
    """
    X = _ensure(X)
    if X.ndim == 2:
        # X[laplacian_idx] -> [C, k, T]
        ref = X[laplacian_idx].mean(axis=1)
        out = X - ref
    else:
        # X[:, laplacian_idx, :] -> [N, C, k, T]
        ref = X[:, laplacian_idx].mean(axis=2)
        out = X - ref
    return np.ascontiguousarray(out, dtype=np.float32)


def bipolar_ref(X: np.ndarray, bipolar_idx: np.ndarray) -> np.ndarray:
    """Subtract single nearest neighbor per channel.

    Args:
        X: [N, C, T] or [C, T]
        bipolar_idx: [C] int, bipolar_idx[c] = index of c's nearest neighbor
    """
    X = _ensure(X)
    if X.ndim == 2:
        out = X - X[bipolar_idx]
    else:
        out = X - X[:, bipolar_idx]
    return np.ascontiguousarray(out, dtype=np.float32)


# ============================================================================
# Unified entry point
# ============================================================================

REFERENCE_MODES = ("native", "car", "median", "gs", "laplacian", "bipolar")


def apply_reference(
    X: np.ndarray,
    mode: str,
    graph: Optional[DatasetGraph] = None,
) -> np.ndarray:
    """Apply one of the six reference operators.

    For 'laplacian' and 'bipolar', a DatasetGraph must be supplied.
    """
    mode = mode.lower()
    if mode == "native":
        return native_ref(X)
    if mode == "car":
        return car_ref(X)
    if mode == "median":
        return median_ref(X)
    if mode == "gs":
        return gs_ref(X)
    if mode == "laplacian":
        if graph is None:
            raise ValueError("Mode 'laplacian' requires a DatasetGraph")
        return laplacian_ref(X, graph.laplacian_idx)
    if mode == "bipolar":
        if graph is None:
            raise ValueError("Mode 'bipolar' requires a DatasetGraph")
        return bipolar_ref(X, graph.bipolar_idx)
    raise ValueError(
        f"Unknown reference mode: {mode!r}. Known: {REFERENCE_MODES}"
    )
