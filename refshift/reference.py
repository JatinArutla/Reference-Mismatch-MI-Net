"""Reference operators, neighbor graph, and a sklearn-compatible transformer.

Seven operators, grouped by computational structure:

    Global-mean family      (reduce across all channels)
        native              identity — use the dataset's native hardware reference
        car                 common average: X - mean_c(X)
        median              robust CAR: X - median_c(X)
        gs                  Gram-Schmidt projection against the leave-one-out mean

    Spatial-differential family  (local neighbor subtraction)
        laplacian           X - mean of k=4 nearest-neighbor channels
        bipolar             X - single nearest-neighbor channel

    Source-model family
        rest                Reference Electrode Standardization Technique
                            (Yao 2001): linear transform that approximates
                            the potential referenced to infinity, via a
                            three-layer spherical head model fit to the
                            electrode montage.

Nearest-neighbor graphs are built from Euclidean distances in the MNE
``standard_1005`` montage. The REST transformation matrix is built from the
same montage via ``mne.make_sphere_model`` + ``mne.make_forward_solution``.
The same montage covers every EEG channel in IV-2a, OpenBMI, Cho2017, and
Dreyer2023.

Operator implementations are numerically identical to the v2 implementation
for the original six; REST is new and validated by the sum-to-zero spatial
property of its output on any input reference (see tests/test_reference.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


REFERENCE_MODES = (
    "native", "car", "median", "gs", "laplacian", "bipolar", "rest",
)

# Modes that require a DatasetGraph (i.e. some dataset-specific precomputed
# state: neighbor indices for spatial modes, leadfield transform for REST).
_GRAPH_MODES = ("laplacian", "bipolar", "rest")


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


def _build_rest_matrix(
    ch_names: List[str],
    montage: str = "standard_1005",
    *,
    sfreq: float = 250.0,
    source_spacing_mm: float = 10.0,
    source_mindist_mm: float = 5.0,
) -> np.ndarray:
    """Compute the REST linear-transform matrix T for a given channel set.

    REST (Yao 2001) estimates the potential referenced to infinity.
    Mathematically, given a leadfield G (C x D) relating D equivalent
    source dipoles to C scalp electrodes, and letting
    Ga = G - mean_c(G), the transformation that maps from any linear
    re-reference of the scalp potentials to REST is

        T = G @ pinv(Ga) @ (I - (1/C) * 1 1^T)

    Applied to an epoch tensor X[..., C, T] as Y = T @ X, the output has the
    reference-invariance property: REST(V) = REST(V + a*1) for any scalar a,
    because the centering step `(I - J/C)` subtracts the per-sample mean.

    The leadfield G is computed from a standard 3-layer spherical head
    model fit to the electrode coordinates, using MNE's standard API.

    Parameters
    ----------
    ch_names : list of str
        Channel names in the order used by the downstream data.
    montage : str
        MNE standard montage name (same as build_graph).
    sfreq : float
        Placeholder sampling frequency for mne.create_info; has no effect
        on the computed transform (REST is a pure spatial operator).
    source_spacing_mm : float
        Volume source space spacing in mm. 10 mm -> ~1500 sources for a
        typical head volume. Smaller spacing increases compute and storage
        but has diminishing returns on the transform once well-conditioned.
    source_mindist_mm : float
        Minimum distance (mm) between sources and the inner skull surface,
        passed to mne.setup_volume_source_space to avoid ill-posed dipoles
        near the boundary.

    Returns
    -------
    T : np.ndarray, shape (C, C), float32
        The REST transformation matrix for this channel set.
    """
    import mne  # lazy
    _ll = mne.get_config("MNE_LOGGING_LEVEL")
    mne.set_log_level("ERROR")
    try:
        info = mne.create_info(
            ch_names=list(ch_names), sfreq=float(sfreq), ch_types="eeg",
        )
        info.set_montage(montage)

        sphere = mne.make_sphere_model(r0="auto", head_radius="auto", info=info)
        src = mne.setup_volume_source_space(
            subject=None,
            pos=float(source_spacing_mm),
            sphere=sphere,
            exclude=0.0,
            mindist=float(source_mindist_mm),
        )
        fwd = mne.make_forward_solution(
            info, trans=None, src=src, bem=sphere,
            eeg=True, meg=False, verbose="ERROR",
        )
        G = fwd["sol"]["data"]  # (C, D) float64
    finally:
        if _ll is not None:
            mne.set_log_level(_ll)

    C = G.shape[0]
    Ga = G - G.mean(axis=0, keepdims=True)
    pinvGa = np.linalg.pinv(Ga)
    center = np.eye(C) - np.ones((C, C)) / C
    T = G @ pinvGa @ center
    return np.ascontiguousarray(T, dtype=np.float32)


@dataclass(frozen=True)
class DatasetGraph:
    """Pre-computed per-dataset state for reference operators that need it.

    Built once per dataset from channel names + montage; passed to
    ReferenceTransformer for any mode in ``_GRAPH_MODES``.

    Attributes
    ----------
    ch_names : list of str
        Channel names in the order used to build the graph. Must match the
        channel order of the arrays fed to the transformer.
    laplacian_idx : np.ndarray of shape (C, k), int64
        Indices of each channel's k nearest spatial neighbors (for Laplacian).
    bipolar_idx : np.ndarray of shape (C,), int64
        Index of each channel's single nearest spatial neighbor (for bipolar).
    k : int
        Neighbor count used for the Laplacian operator.
    montage : str
        Name of the MNE standard montage used as the position source.
    rest_matrix : np.ndarray or None, shape (C, C), float32
        REST transformation matrix. ``None`` if REST wasn't requested when
        the graph was built (avoids the ~10-30 s cost of computing a
        forward solution when the caller doesn't need REST).
    """
    ch_names: List[str]
    laplacian_idx: np.ndarray
    bipolar_idx: np.ndarray
    k: int
    montage: str
    rest_matrix: Optional[np.ndarray] = field(default=None)


def build_graph(
    ch_names: List[str],
    k: int = 4,
    montage: str = "standard_1005",
    *,
    include_rest: bool = False,
) -> DatasetGraph:
    """Build nearest-neighbor indices (and optionally the REST transform).

    Parameters
    ----------
    ch_names : list of str
        Channel names defining the graph.
    k : int
        Number of Laplacian neighbors. Default 4.
    montage : str
        MNE standard montage name. Default ``"standard_1005"``.
    include_rest : bool
        If True, also compute and store the REST transformation matrix.
        Default False; set True only when the 'rest' reference mode is
        used, since forward-solution computation takes several seconds.
    """
    xyz = _get_channel_positions(ch_names, montage=montage)
    d = _pairwise_distances(xyz)
    lap = np.argsort(d, axis=1)[:, :k].astype(np.int64)
    bip = np.argmin(d, axis=1).astype(np.int64)

    rest_matrix = None
    if include_rest:
        rest_matrix = _build_rest_matrix(ch_names, montage=montage)

    return DatasetGraph(
        ch_names=list(ch_names),
        laplacian_idx=lap,
        bipolar_idx=bip,
        k=k,
        montage=montage,
        rest_matrix=rest_matrix,
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
    """
    X = _ensure_f32(X)
    squeeze = (X.ndim == 2)
    if squeeze:
        X = X[None, ...]
    _, C, _ = X.shape
    s = X.sum(axis=1, keepdims=True)                   # [N, 1, T]
    r = (s - X) / max(C - 1, 1)                         # [N, C, T] LOO mean
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


def _rest(X: np.ndarray, rest_matrix: np.ndarray) -> np.ndarray:
    """Apply the REST transformation: Y[..., c, t] = sum_j T[c, j] X[..., j, t].

    Works for [C, T] and [N, C, T] inputs. The matrix T is expected to
    already incorporate the centering step (I - 1_C 1_C^T / C), so the
    operator is insensitive to the input's native reference.
    """
    X = _ensure_f32(X)
    T = rest_matrix.astype(np.float32, copy=False)
    if X.ndim == 2:
        return np.ascontiguousarray(T @ X, dtype=np.float32)
    # Batched matmul over the channel axis: einsum avoids materialising a
    # broadcasted (N, C, C, T) tensor.
    return np.ascontiguousarray(
        np.einsum("ij,njt->nit", T, X), dtype=np.float32,
    )


def apply_reference(
    X: np.ndarray,
    mode: str,
    graph: Optional[DatasetGraph] = None,
) -> np.ndarray:
    """Dispatch X through the named reference operator.

    ``graph`` is required for any mode in ``_GRAPH_MODES``
    ('laplacian', 'bipolar', 'rest'). For REST, the graph must have been
    built with ``include_rest=True``.
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

    if mode in _GRAPH_MODES:
        if graph is None:
            raise ValueError(f"Mode {mode!r} requires a DatasetGraph")
        if mode == "laplacian":
            return _laplacian(X, graph.laplacian_idx)
        if mode == "bipolar":
            return _bipolar(X, graph.bipolar_idx)
        if mode == "rest":
            if graph.rest_matrix is None:
                raise ValueError(
                    "Mode 'rest' requires a DatasetGraph built with "
                    "include_rest=True."
                )
            return _rest(X, graph.rest_matrix)

    raise ValueError(f"Unknown reference mode: {mode!r}. Known: {REFERENCE_MODES}")


# ---------------------------------------------------------------------------
# sklearn-compatible transformer
# ---------------------------------------------------------------------------

class ReferenceTransformer(BaseEstimator, TransformerMixin):
    """Apply a reference operator to [N, C, T] epoched EEG arrays.

    Stateless transformer: ``fit`` is a no-op. Designed to sit as the first
    step of a scikit-learn Pipeline before CSP/Covariances or a DL classifier
    so that the reference choice is swappable per experiment.

    Parameters
    ----------
    mode : one of REFERENCE_MODES
    graph : DatasetGraph or None, default None
        Required for 'laplacian', 'bipolar', and 'rest'. Must match the
        channel ordering of the input arrays. For 'rest', the graph must
        have been built with ``include_rest=True``.

    Notes
    -----
    Input is ``numpy.ndarray`` of shape ``(n_epochs, n_channels, n_times)``
    or ``(n_channels, n_times)``, any float dtype (cast to float32).
    """

    def __init__(self, mode: str, graph: Optional[DatasetGraph] = None):
        self.mode = mode
        self.graph = graph

    def _check(self) -> None:
        if self.mode not in REFERENCE_MODES:
            raise ValueError(
                f"Unknown mode: {self.mode!r}. Known: {REFERENCE_MODES}"
            )
        if self.mode in _GRAPH_MODES and self.graph is None:
            raise ValueError(f"Mode {self.mode!r} requires graph=DatasetGraph(...)")
        if self.mode == "rest" and (
            self.graph is not None and self.graph.rest_matrix is None
        ):
            raise ValueError(
                "Mode 'rest' requires DatasetGraph built with include_rest=True."
            )

    def fit(self, X, y=None):
        self._check()
        return self

    def transform(self, X):
        self._check()
        return apply_reference(X, self.mode, graph=self.graph)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
