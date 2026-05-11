"""Reference and spatial-derivative operators for EEG decoding.

Six modes in three families:
    Global symmetric:     native, car, median, rest
    Global asymmetric:    cz_ref           (X_i - X_Cz)
    Local spatial filter: laplacian        (X - mean of k nearest neighbours)

LOO-mean is omitted because LOO_i = (C/(C-1)) * CAR_i (scalar multiple of CAR;
identical for any scale-invariant decoder). GS is omitted because the natural
implementation is data-dependent and not a fixed C×C operator. NN-diff was
removed in v0.13: not a literature-recognised reference, and rank-deficient
on dense montages.

All operators take (N, C, T) float arrays. Channel order must match the
graph's ch_names. Graphs are computed once per dataset via build_graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


REFERENCE_MODES = ("native", "car", "median", "laplacian", "rest", "cz_ref")

# Modes that require precomputed dataset state (neighbour indices, REST
# matrix, or Cz channel index). Single source of truth for "build a graph?".
_GRAPH_MODES = ("laplacian", "rest", "cz_ref")


def reference_modes_for_dataset(
    dataset_id: str,
    modes: tuple = REFERENCE_MODES,
) -> tuple:
    """Return the subset of `modes` that is well-defined for `dataset_id`.

    Currently the only dataset-specific exclusion is cz_ref on Schirrmeister2017,
    which used Cz as the recording reference and therefore has no Cz channel
    in its data. Without this filter, the default reference_modes pipeline
    crashes on Schirrmeister with a ValueError from apply_reference. Use this
    helper before calling run_mismatch / run_mismatch_jitter / run_pre_ems_diagonal
    so the call works on every dataset without per-dataset overrides.
    """
    dataset_id = dataset_id.lower()
    modes_t = tuple(modes)
    if dataset_id == "schirrmeister2017":
        return tuple(m for m in modes_t if m != "cz_ref")
    return modes_t


def validate_reference_modes(
    modes: tuple,
    graph: Optional["DatasetGraph"],
    dataset_id: Optional[str] = None,
) -> None:
    """Fail loudly before training starts if `modes` contains anything that
    apply_reference would crash on at runtime, given `graph`.

    Catches the most common footgun: cz_ref requested for a channel set
    without Cz (e.g. Schirrmeister2017). The error message points at
    reference_modes_for_dataset so the user knows the fix.
    """
    modes_t = tuple(m.lower() for m in modes)
    unknown = [m for m in modes_t if m not in REFERENCE_MODES]
    if unknown:
        raise ValueError(
            f"Unknown reference modes: {unknown}. Known: {REFERENCE_MODES}"
        )
    if "cz_ref" in modes_t and graph is not None and graph.cz_idx is None:
        ds_msg = f" for {dataset_id!r}" if dataset_id else ""
        raise ValueError(
            f"reference_modes contains 'cz_ref'{ds_msg}, but the channel "
            f"set has no Cz channel (graph.cz_idx is None). This usually "
            f"means the dataset uses Cz as recording reference (e.g. "
            f"Schirrmeister2017). Use "
            f"refshift.reference.reference_modes_for_dataset(dataset_id) "
            f"to get a dataset-safe mode list, or drop 'cz_ref' from "
            f"reference_modes manually."
        )
    if "rest" in modes_t and graph is not None and graph.rest_matrix is None:
        raise ValueError(
            "reference_modes contains 'rest' but graph was built with "
            "include_rest=False. Build with build_graph(..., include_rest=True)."
        )
    needs_graph = [m for m in modes_t if m in _GRAPH_MODES]
    if needs_graph and graph is None:
        raise ValueError(
            f"reference_modes={modes_t} includes graph-requiring modes "
            f"{needs_graph} but graph=None."
        )


# ---------------------------------------------------------------------------
# Channel positions, neighbour graph, REST matrix
# ---------------------------------------------------------------------------

def _get_channel_positions(ch_names: List[str], montage: str) -> np.ndarray:
    """Return [C, 3] xyz positions under the given MNE standard montage."""
    import mne
    mont = mne.channels.make_standard_montage(montage)
    pos = mont.get_positions()["ch_pos"]
    missing = [ch for ch in ch_names if ch not in pos]
    if missing:
        raise ValueError(f"Channels not in {montage!r}: {missing}")
    return np.array([pos[ch] for ch in ch_names], dtype=np.float64)


def _pairwise_distances(xyz: np.ndarray) -> np.ndarray:
    """Euclidean [C, C] with inf on the diagonal so a channel isn't its own NN."""
    d = np.sqrt(((xyz[:, None, :] - xyz[None, :, :]) ** 2).sum(axis=-1))
    np.fill_diagonal(d, np.inf)
    return d


def _build_rest_matrix(
    ch_names: List[str],
    montage: str,
    *,
    sfreq: float = 250.0,
    source_spacing_mm: float = 10.0,
    source_mindist_mm: float = 5.0,
) -> np.ndarray:
    """REST (Yao 2001) transform matrix for the given channel set.

    Builds a 3-layer spherical head model, computes leadfield G, and returns
    T = G @ pinv(G - mean_c(G)) @ (I - 1_C 1_C^T / C). The trailing centering
    operator gives reference invariance: REST(V + a*1_C) = REST(V).

    rcond=1e-4 in the pseudo-inverse: published REST toolbox convention; the
    numpy default (rcond derived from largest singular value) was too aggressive
    on small well-conditioned leadfields and produced numerical noise.
    """
    import mne
    prev_log_level = mne.get_config("MNE_LOGGING_LEVEL")
    mne.set_log_level("ERROR")
    try:
        info = mne.create_info(
            ch_names=list(ch_names), sfreq=float(sfreq), ch_types="eeg",
        )
        info.set_montage(montage)
        sphere = mne.make_sphere_model(r0="auto", head_radius="auto", info=info)
        src = mne.setup_volume_source_space(
            subject=None, pos=float(source_spacing_mm), sphere=sphere,
            exclude=0.0, mindist=float(source_mindist_mm),
        )
        fwd = mne.make_forward_solution(
            info, trans=None, src=src, bem=sphere,
            eeg=True, meg=False, verbose="ERROR",
        )
        G = fwd["sol"]["data"]
    finally:
        if prev_log_level is not None:
            mne.set_log_level(prev_log_level)

    C = G.shape[0]
    Ga = G - G.mean(axis=0, keepdims=True)
    pinvGa = np.linalg.pinv(Ga, rcond=1e-4)
    center = np.eye(C) - np.ones((C, C)) / C
    T = G @ pinvGa @ center
    return np.ascontiguousarray(T, dtype=np.float32)


@dataclass(frozen=True)
class DatasetGraph:
    """Pre-computed per-dataset state for graph-aware operators.

    cz_idx is None when Cz isn't in ch_names (e.g. Schirrmeister2017, which
    used Cz as the recording reference). rest_matrix is None unless built
    with include_rest=True (the spherical model is the slow part).
    """
    ch_names: List[str]
    laplacian_idx: np.ndarray  # (C, k) int64
    k: int
    montage: str
    rest_matrix: Optional[np.ndarray] = field(default=None)  # (C, C) float32
    rest_cond: Optional[float] = field(default=None)
    cz_idx: Optional[int] = field(default=None)


def build_graph(
    ch_names: List[str],
    k: int = 4,
    montage: str = "standard_1005",
    *,
    include_rest: bool = False,
) -> DatasetGraph:
    """Build neighbour indices for the given channel set; optionally REST."""
    xyz = _get_channel_positions(ch_names, montage=montage)
    d = _pairwise_distances(xyz)
    lap = np.argsort(d, axis=1)[:, :k].astype(np.int64)

    rest_matrix = None
    rest_cond = None
    if include_rest:
        rest_matrix = _build_rest_matrix(ch_names, montage=montage)
        rest_cond = float(np.linalg.cond(rest_matrix))

    cz_idx = ch_names.index("Cz") if "Cz" in ch_names else None

    return DatasetGraph(
        ch_names=list(ch_names),
        laplacian_idx=lap, k=k, montage=montage,
        rest_matrix=rest_matrix, rest_cond=rest_cond, cz_idx=cz_idx,
    )


# ---------------------------------------------------------------------------
# Operators. All take (N, C, T) and return (N, C, T) float32.
# ---------------------------------------------------------------------------

def _check_3d(X: np.ndarray) -> np.ndarray:
    if X.ndim != 3:
        raise ValueError(f"Expected (N, C, T), got shape {X.shape}")
    return X.astype(np.float32, copy=False)


def _car(X: np.ndarray) -> np.ndarray:
    X = _check_3d(X)
    return np.ascontiguousarray(X - X.mean(axis=1, keepdims=True), dtype=np.float32)


def _median(X: np.ndarray) -> np.ndarray:
    X = _check_3d(X)
    return np.ascontiguousarray(X - np.median(X, axis=1, keepdims=True), dtype=np.float32)


def _laplacian(X: np.ndarray, lap_idx: np.ndarray) -> np.ndarray:
    """X - mean of k nearest spatial neighbours (kNN local Laplacian)."""
    X = _check_3d(X)
    ref = X[:, lap_idx].mean(axis=2)  # (N, C, k, T) -> (N, C, T)
    return np.ascontiguousarray(X - ref, dtype=np.float32)


def _cz_ref(X: np.ndarray, cz_idx: int) -> np.ndarray:
    """Y_i = X_i - X_Cz; the Cz channel becomes identically zero."""
    X = _check_3d(X)
    return np.ascontiguousarray(X - X[:, cz_idx:cz_idx + 1, :], dtype=np.float32)


def _rest(X: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply REST: einsum avoids materialising (N, C, C, T)."""
    X = _check_3d(X)
    T = T.astype(np.float32, copy=False)
    return np.ascontiguousarray(np.einsum("ij,njt->nit", T, X), dtype=np.float32)


def apply_reference(
    X: np.ndarray,
    mode: str,
    graph: Optional[DatasetGraph] = None,
) -> np.ndarray:
    """Dispatch X through the named operator. graph is required for graph modes."""
    mode = mode.lower()
    if mode == "native":
        return _check_3d(X).copy()
    if mode == "car":
        return _car(X)
    if mode == "median":
        return _median(X)
    if mode not in _GRAPH_MODES:
        raise ValueError(f"Unknown reference mode: {mode!r}. Known: {REFERENCE_MODES}")
    if graph is None:
        raise ValueError(f"Mode {mode!r} requires a DatasetGraph")
    if mode == "laplacian":
        return _laplacian(X, graph.laplacian_idx)
    if mode == "rest":
        if graph.rest_matrix is None:
            raise ValueError("Mode 'rest' requires build_graph(..., include_rest=True)")
        return _rest(X, graph.rest_matrix)
    # cz_ref
    if graph.cz_idx is None:
        raise ValueError(
            "Mode 'cz_ref' requires Cz in the channel set. cz_idx=None usually "
            "means the dataset uses Cz as recording reference (e.g. Schirrmeister)."
        )
    return _cz_ref(X, graph.cz_idx)


# ---------------------------------------------------------------------------
# sklearn-compatible transformer
# ---------------------------------------------------------------------------

class ReferenceTransformer(BaseEstimator, TransformerMixin):
    """Stateless sklearn transformer wrapping apply_reference.

    Sits at the front of a Pipeline so the reference is swappable per
    experiment. fit is a no-op; transform calls apply_reference, which
    raises on invalid mode/graph combinations.
    """

    def __init__(self, mode: str, graph: Optional[DatasetGraph] = None):
        self.mode = mode
        self.graph = graph

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return apply_reference(X, self.mode, graph=self.graph)
