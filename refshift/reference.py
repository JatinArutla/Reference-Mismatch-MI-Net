"""Reference and spatial-representation operators for EEG decoding.

Six operators, grouped by computational structure. The naming throughout
deliberately distinguishes between *reference* operators (which redefine
the zero-potential point) and *spatial-derivative* operators (which form
local differences, a different category from rereferencing in the strict
sense). The paper studies both as channel-space transformations.

    Global / as-recorded family
        native              identity — use the dataset's native hardware
                            reference (e.g. left-mastoid for IV-2a)
        car                 common average reference: X - mean_c(X)
        median              robust common-reference control: X - median_c(X)
                            included as a robustness control, not a
                            mainstream MI reference
        rest                REST-like spherical-model re-reference
                            (Yao 2001 spherical-head approximation):
                            linear transform that approximates the
                            potential referenced to infinity, computed
                            from a three-layer spherical head model and
                            a regularized leadfield pseudo-inverse
                            (rcond=1e-4). Not validated against a known
                            REST implementation; we describe it as
                            "REST-like" rather than "REST".

    Local spatial-derivative family
        laplacian           kNN local Laplacian: X - mean of the k=4
                            nearest-neighbour channels. *Not* formal
                            CSD/spherical-spline Laplacian; this is the
                            discrete neighbour-mean form used as a
                            spatial-filter approximation.
        nn_diff             nearest-neighbour local difference:
                            X_i - X_{nn(i)}. Dimension-preserving local
                            derivative. *Not* a clinical bipolar montage
                            (which uses predefined electrode pairs and
                            typically reduces channel count). The naming
                            is deliberately "NN-diff" and not "bipolar".
                            See ``DatasetGraph.nn_diff_rank`` for the
                            per-dataset rank diagnostic.

Note on omitted operators. We do not include a leave-one-out (LOO) mean
reference because LOO_i = (C/(C-1)) * CAR_i — they differ only by a
scalar factor and behave identically for any scale-invariant decoder
(CSP+LDA, batch-normalized neural networks). We do not include a
projection-based "GS" operator in the main set because the natural
implementation is data-dependent and does not form a fixed C×C linear
operator, putting it outside the operator-shift framework.

Nearest-neighbour graphs are built from Euclidean distances in the MNE
``standard_1005`` montage. REST uses the same montage via
``mne.make_sphere_model`` + ``mne.make_forward_solution``. The same
montage covers every EEG channel in IV-2a, OpenBMI, Cho2017, Dreyer2023,
and Schirrmeister2017.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


REFERENCE_MODES = (
    "native", "car", "median", "laplacian", "nn_diff", "rest",
)

# Modes that require a DatasetGraph (i.e. some dataset-specific precomputed
# state: neighbor indices for spatial modes, leadfield transform for REST).
_GRAPH_MODES = ("laplacian", "nn_diff", "rest")


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
    # Regularized pseudo-inverse: rcond=1e-4 is the standard choice in
    # realistic-head-model REST work. The default numpy rcond depends on
    # the largest singular value and can be too aggressive for
    # well-conditioned but small Ga, leading to numerical noise in the
    # inverse. 1e-4 is conservative and matches the published REST
    # toolbox recommendations.
    pinvGa = np.linalg.pinv(Ga, rcond=1e-4)
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
        Indices of each channel's k nearest spatial neighbors (for the
        kNN local-Laplacian operator).
    nn_diff_idx : np.ndarray of shape (C,), int64
        Index of each channel's single nearest spatial neighbor (for the
        nearest-neighbour local-difference operator). This is *not* a
        clinical bipolar montage with predefined electrode pairs; it is
        a dimension-preserving local-difference operator. See
        ``nn_diff_rank`` and ``nn_diff_nullity`` for the per-dataset
        rank diagnostic.
    k : int
        Neighbor count used for the Laplacian operator.
    montage : str
        Name of the MNE standard montage used as the position source.
    rest_matrix : np.ndarray or None, shape (C, C), float32
        REST transformation matrix. ``None`` if REST wasn't requested when
        the graph was built (avoids the ~10-30 s cost of computing a
        forward solution when the caller doesn't need REST).
    nn_diff_rank : int
        Rank of the (I - P) matrix where P is the channel permutation
        induced by ``nn_diff_idx``. For a rank-(C-1) chain, rank = C-1.
        Smaller ranks indicate that the nearest-neighbour graph contains
        cycles (e.g. mutual nearest-neighbours), which inflate the operator's
        nullity beyond the minimum. Logged alongside results so reviewers
        can verify that the NN-diff effect is not driven by pathological
        rank collapse.
    nn_diff_nullity : int
        ``C - nn_diff_rank``. Number of dimensions destroyed by the operator
        beyond the single global-DC dimension that any local-difference
        operator must remove.
    rest_cond : float or None
        Condition number of the REST transformation matrix (``np.linalg.cond``).
        ``None`` if REST wasn't requested. Reported for transparency about
        REST conditioning, which can be sensitive to electrode coverage.
    """
    ch_names: List[str]
    laplacian_idx: np.ndarray
    nn_diff_idx: np.ndarray
    k: int
    montage: str
    rest_matrix: Optional[np.ndarray] = field(default=None)
    nn_diff_rank: int = field(default=0)
    nn_diff_nullity: int = field(default=0)
    rest_cond: Optional[float] = field(default=None)


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

    Notes
    -----
    The NN-diff operator is implemented as ``Y_i = X_i - X_{nn(i)}``,
    which is equivalent to multiplication by ``(I - P)`` where ``P`` is
    the channel-permutation matrix indexed by ``nn_diff_idx``. When the
    nearest-neighbour graph has cycles (e.g. C3-C5 mutually nearest),
    the operator destroys more than one dimension. We compute and store
    the rank of ``(I - P)`` alongside the indices so the rank can be
    inspected per dataset.
    """
    xyz = _get_channel_positions(ch_names, montage=montage)
    d = _pairwise_distances(xyz)
    lap = np.argsort(d, axis=1)[:, :k].astype(np.int64)
    nn = np.argmin(d, axis=1).astype(np.int64)

    # Rank diagnostic for NN-diff
    C = len(ch_names)
    M = np.eye(C) - np.eye(C)[nn]  # M[i] = e_i - e_{nn(i)}
    nn_rank = int(np.linalg.matrix_rank(M))
    nn_nullity = C - nn_rank

    rest_matrix = None
    rest_cond = None
    if include_rest:
        rest_matrix = _build_rest_matrix(ch_names, montage=montage)
        rest_cond = float(np.linalg.cond(rest_matrix))

    return DatasetGraph(
        ch_names=list(ch_names),
        laplacian_idx=lap,
        nn_diff_idx=nn,
        k=k,
        montage=montage,
        rest_matrix=rest_matrix,
        nn_diff_rank=nn_rank,
        nn_diff_nullity=nn_nullity,
        rest_cond=rest_cond,
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


def _laplacian(X: np.ndarray, laplacian_idx: np.ndarray) -> np.ndarray:
    """kNN local Laplacian: subtract the mean of the k nearest spatial
    neighbours from each channel. Not formal CSD/spherical-spline
    Laplacian; this is the discrete neighbour-mean form used as an EEG
    spatial-filter approximation.
    """
    X = _ensure_f32(X)
    if X.ndim == 2:
        ref = X[laplacian_idx].mean(axis=1)              # [C, k, T] -> [C, T]
    else:
        ref = X[:, laplacian_idx].mean(axis=2)           # [N, C, k, T] -> [N, C, T]
    return np.ascontiguousarray(X - ref, dtype=np.float32)


def _nn_diff(X: np.ndarray, nn_diff_idx: np.ndarray) -> np.ndarray:
    """Nearest-neighbour local difference: ``Y_i = X_i - X_{nn(i)}``.

    Dimension-preserving local-derivative operator. *Not* a clinical
    bipolar montage, which would use predefined electrode pairs and
    typically reduces channel count. The naming "NN-diff" rather than
    "bipolar" is deliberate. See ``DatasetGraph.nn_diff_rank`` for the
    rank diagnostic per dataset.
    """
    X = _ensure_f32(X)
    if X.ndim == 2:
        return np.ascontiguousarray(X - X[nn_diff_idx], dtype=np.float32)
    return np.ascontiguousarray(X - X[:, nn_diff_idx], dtype=np.float32)


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
    ('laplacian', 'nn_diff', 'rest'). For REST, the graph must have been
    built with ``include_rest=True``.
    """
    mode = mode.lower()
    if mode == "native":
        return _native(X)
    if mode == "car":
        return _car(X)
    if mode == "median":
        return _median(X)

    if mode in _GRAPH_MODES:
        if graph is None:
            raise ValueError(f"Mode {mode!r} requires a DatasetGraph")
        if mode == "laplacian":
            return _laplacian(X, graph.laplacian_idx)
        if mode == "nn_diff":
            return _nn_diff(X, graph.nn_diff_idx)
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
        Required for 'laplacian', 'nn_diff', and 'rest'. Must match the
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
