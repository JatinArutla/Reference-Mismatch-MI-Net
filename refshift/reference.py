"""Reference operators, neighbor graph, and a sklearn-compatible transformer.

Six operators, grouped into two families by computational structure:

    Global-mean family      (reduce across all channels)
        native              identity — use the dataset's native hardware reference
        car                 common average: X - mean_c(X)
        median              robust CAR: X - median_c(X)
        gs                  Gram-Schmidt projection against the leave-one-out mean

    Spatial-differential family  (reduce across local spatial neighbors)
        laplacian           X - mean of k=4 nearest-neighbor channels
        bipolar             X - single nearest-neighbor channel

Nearest-neighbor graphs are built from Euclidean distances in the MNE
``standard_1005`` montage. The same montage covers every EEG channel in
IV-2a, OpenBMI, Cho2017, and Dreyer2023.

Operator implementations are numerically identical to the v2 implementation
(verified: CAR residual channel mean ~1e-6, GS orthogonality cosine ~1e-8,
hand-computed 3-channel Laplacian and bipolar cases match). See tests/.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


REFERENCE_MODES = ("native", "car", "median", "gs", "laplacian", "bipolar")
_SPATIAL_MODES = ("laplacian", "bipolar")


# ---------------------------------------------------------------------------
# Channel positions and neighbor graph
# ---------------------------------------------------------------------------

def _get_channel_positions(
    ch_names: List[str],
    montage: str = "standard_1005",
) -> np.ndarray:
    """Return [C, 3] xyz positions for ``ch_names`` under an MNE standard montage.

    Raises if any channel is not present in the montage.
    """
    import mne
    mont = mne.channels.make_standard_montage(montage)
    pos = mont.get_positions()["ch_pos"]
    missing = [ch for ch in ch_names if ch not in pos]
    if missing:
        raise ValueError(
            f"Channels not in montage {montage!r}: {missing}. "
            f"Consider case/alias mismatches (e.g. 'FCz' vs 'FCZ')."
        )
    return np.array([pos[ch] for ch in ch_names], dtype=np.float64)


def _pairwise_distances(xyz: np.ndarray) -> np.ndarray:
    """Euclidean distance matrix [C, C] with np.inf on the diagonal so a
    channel cannot be picked as a nearest neighbor of itself."""
    d = np.sqrt(((xyz[:, None, :] - xyz[None, :, :]) ** 2).sum(axis=-1))
    np.fill_diagonal(d, np.inf)
    return d


@dataclass(frozen=True)
class DatasetGraph:
    """Pre-computed neighbor indices for a dataset's EEG channel set.

    Built once per dataset from channel names + montage positions, then
    passed into ReferenceTransformer for spatial modes.

    Attributes
    ----------
    ch_names : list of str
        Channel names in the order used to build the graph. Must match the
        channel order of the arrays fed to the transformer.
    laplacian_idx : np.ndarray of shape (C, k), int64
        laplacian_idx[c] = indices of c's k nearest spatial neighbors.
    bipolar_idx : np.ndarray of shape (C,), int64
        bipolar_idx[c] = index of c's single nearest spatial neighbor.
    k : int
        Neighbor count used for the Laplacian operator.
    montage : str
        Name of the MNE standard montage used as the position source.
    """
    ch_names: List[str]
    laplacian_idx: np.ndarray
    bipolar_idx: np.ndarray
    k: int
    montage: str


def build_graph(
    ch_names: List[str],
    k: int = 4,
    montage: str = "standard_1005",
) -> DatasetGraph:
    """Build nearest-neighbor indices for Laplacian and bipolar references."""
    xyz = _get_channel_positions(ch_names, montage=montage)
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


# ---------------------------------------------------------------------------
# Reference operators (accept [N, C, T] or [C, T])
# ---------------------------------------------------------------------------

def _ensure_f32(X: np.ndarray) -> np.ndarray:
    if X.ndim not in (2, 3):
        raise ValueError(f"Expected [C,T] or [N,C,T], got shape {X.shape}")
    return X.astype(np.float32, copy=False)


def _native(X: np.ndarray) -> np.ndarray:
    """Identity: returns a fresh copy so downstream mutations don't leak."""
    return np.ascontiguousarray(_ensure_f32(X).copy())


def _car(X: np.ndarray) -> np.ndarray:
    X = _ensure_f32(X)
    axis = 1 if X.ndim == 3 else 0
    return np.ascontiguousarray(X - X.mean(axis=axis, keepdims=True), dtype=np.float32)


def _median(X: np.ndarray) -> np.ndarray:
    X = _ensure_f32(X)
    axis = 1 if X.ndim == 3 else 0
    return np.ascontiguousarray(X - np.median(X, axis=axis, keepdims=True), dtype=np.float32)


def _gs(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Gram-Schmidt: for each (trial, channel) orthogonalize the channel's
    time series against the mean of the *other* channels in that trial.

    Y[n, c, :] = X[n, c, :] - <X[n,c,:], r[n,c,:]> / <r[n,c,:], r[n,c,:]> * r[n,c,:]
    where r[n, c, :] = mean_{c' != c} X[n, c', :].
    """
    X = _ensure_f32(X)
    squeeze = (X.ndim == 2)
    if squeeze:
        X = X[None, ...]
    _, C, _ = X.shape
    s = X.sum(axis=1, keepdims=True)                   # [N, 1, T]
    r = (s - X) / max(C - 1, 1)                         # [N, C, T] leave-one-out mean
    num = np.sum(X * r, axis=2, keepdims=True)          # [N, C, 1]
    den = np.sum(r * r, axis=2, keepdims=True) + eps
    Y = X - (num / den) * r
    Y = Y[0] if squeeze else Y
    return np.ascontiguousarray(Y, dtype=np.float32)


def _laplacian(X: np.ndarray, laplacian_idx: np.ndarray) -> np.ndarray:
    X = _ensure_f32(X)
    if X.ndim == 2:
        ref = X[laplacian_idx].mean(axis=1)              # [C, k, T] -> [C, T]
    else:
        ref = X[:, laplacian_idx].mean(axis=2)           # [N, C, k, T] -> [N, C, T]
    return np.ascontiguousarray(X - ref, dtype=np.float32)


def _bipolar(X: np.ndarray, bipolar_idx: np.ndarray) -> np.ndarray:
    X = _ensure_f32(X)
    if X.ndim == 2:
        return np.ascontiguousarray(X - X[bipolar_idx], dtype=np.float32)
    return np.ascontiguousarray(X - X[:, bipolar_idx], dtype=np.float32)


def apply_reference(
    X: np.ndarray,
    mode: str,
    graph: Optional[DatasetGraph] = None,
) -> np.ndarray:
    """Dispatch X through the named reference operator.

    ``graph`` is required for 'laplacian' and 'bipolar'; ignored otherwise.
    """
    mode = mode.lower()
    if mode == "native":
        return _native(X)
    if mode == "car":
        return _car(X)
    if mode == "median":
        return _median(X)
    if mode == "gs":
        return _gs(X)
    if mode in _SPATIAL_MODES:
        if graph is None:
            raise ValueError(f"Mode {mode!r} requires a DatasetGraph")
        if mode == "laplacian":
            return _laplacian(X, graph.laplacian_idx)
        return _bipolar(X, graph.bipolar_idx)
    raise ValueError(f"Unknown reference mode: {mode!r}. Known: {REFERENCE_MODES}")


# ---------------------------------------------------------------------------
# sklearn-compatible transformer
# ---------------------------------------------------------------------------

class ReferenceTransformer(BaseEstimator, TransformerMixin):
    """Apply a reference operator to [N, C, T] epoched EEG arrays.

    This is a stateless transformer: ``fit`` is a no-op. It is designed to
    sit as the first step of a scikit-learn Pipeline, before CSP/Covariances
    or a DL classifier, so that the reference choice is swappable per
    experiment without touching the rest of the pipeline.

    Parameters
    ----------
    mode : {'native','car','median','gs','laplacian','bipolar'}
        Which reference operator to apply.
    graph : DatasetGraph or None, default None
        Required for 'laplacian' and 'bipolar'. Must be built from the same
        channel ordering as the input arrays. Ignored for global-mean modes.

    Notes
    -----
    Input is expected to be a ``numpy.ndarray`` of shape ``(n_epochs,
    n_channels, n_times)`` and dtype float (will be cast to float32). This
    matches MOABB's ``paradigm.get_data(..., return_epochs=False)`` output
    and braindecode's windowed-epoch tensors after ``.numpy()``.
    """

    def __init__(self, mode: str, graph: Optional[DatasetGraph] = None):
        self.mode = mode
        self.graph = graph

    def _check(self) -> None:
        if self.mode not in REFERENCE_MODES:
            raise ValueError(
                f"Unknown mode: {self.mode!r}. Known: {REFERENCE_MODES}"
            )
        if self.mode in _SPATIAL_MODES and self.graph is None:
            raise ValueError(f"Mode {self.mode!r} requires graph=DatasetGraph(...)")

    def fit(self, X, y=None):
        self._check()
        return self

    def transform(self, X):
        self._check()
        return apply_reference(X, self.mode, graph=self.graph)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
