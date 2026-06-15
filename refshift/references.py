"""EEG reference operators, their families, and Euclidean Alignment.

WHAT A "REFERENCE" IS
---------------------
Every EEG voltage is measured against some baseline. Re-referencing recomputes
each channel relative to a different baseline. The whole point of this project
is that the choice of reference changes the signal, and a model trained on one
reference can fail on another. Each operator below is a fixed, linear (mostly)
recipe applied to a batch of trials shaped (n_trials, n_channels, n_times).

THE SEVEN OPERATORS
-------------------
  native     no change. The recording's original reference.
  car        Common Average Reference: subtract the average of all channels at
             each time point. "What is this channel doing relative to the mean
             of the head?"
  median     like CAR but subtract the median instead of the mean. More robust
             to a few noisy channels. NON-linear (median isn't a weighted sum).
  cz_ref     subtract the Cz channel from every channel. Cz itself becomes
             zero. A single-electrode reference.
  lap_small  small Laplacian (Hjorth): subtract the average of a channel's k
             nearest spatial neighbours. Emphasises local activity.
  lap_large  large Laplacian: subtract the average of a ring of slightly
             farther neighbours (skip the closest few, then average the next
             few). A coarser spatial filter than lap_small.
  rest       Reference Electrode Standardization Technique (Yao 2001): uses a
             head model to estimate activity relative to a point at infinity.
             A fixed channels x channels matrix; see _build_rest_matrix.

FAMILIES
--------
For the leave-one-family-out experiment we group operators by the *kind* of
spatial operation they perform:
  global   car, median, rest   (each mixes information across the whole head)
  single   cz_ref              (references against one electrode)
  spatial  lap_small, lap_large(local spatial derivatives)
"native" is left out of the families: it is the no-op baseline, not a
re-referencing strategy.

GRAPH-DEPENDENT OPERATORS
-------------------------
cz_ref, the two Laplacians, and rest need to know channel positions or a
precomputed matrix. That information is bundled once per dataset into a
``DatasetGraph`` (see build_graph). native/car/median need nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# Canonical operator order. Any user-supplied set is sorted into this order so
# matrices always have the same layout regardless of input ordering.
REFERENCE_MODES: tuple = (
    "native", "car", "median", "rest", "cz_ref", "lap_small", "lap_large",
)

# Operators that need a DatasetGraph (positions or a precomputed matrix).
GRAPH_MODES: tuple = ("rest", "cz_ref", "lap_small", "lap_large")

# Family grouping for the leave-one-family-out experiment. "native" is excluded
# on purpose: it is the no-op baseline, not a re-referencing family.
FAMILIES: dict = {
    "global": ["car", "median", "rest"],
    "single": ["cz_ref"],
    "spatial": ["lap_small", "lap_large"],
}


def canonical_mode_tuple(refs) -> tuple:
    """Sort any iterable of operator names into canonical REFERENCE_MODES order.

    Accepts a set, list, tuple, etc. Raises on an unknown name. Used by the
    runners so e.g. {"car","native"} always becomes ("native","car").
    """
    resolved = []
    for m in refs:
        m_lc = str(m).lower()
        if m_lc not in REFERENCE_MODES:
            raise ValueError(
                f"Unknown reference mode: {m!r}. Known: {REFERENCE_MODES}"
            )
        resolved.append(m_lc)
    resolved_set = set(resolved)
    return tuple(m for m in REFERENCE_MODES if m in resolved_set)


def reference_modes_for_dataset(dataset_id: str = "iv2a") -> tuple:
    """Operators well-defined for the dataset.

    All seven for most datasets. Schirrmeister2017 used Cz as its recording
    reference, so it has no Cz channel and cz_ref is dropped.
    """
    if dataset_id.lower() == "schirrmeister2017":
        return tuple(m for m in REFERENCE_MODES if m != "cz_ref")
    return REFERENCE_MODES


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

    Builds a 3-layer spherical head model, computes the leadfield G (how each
    modelled brain source projects to each electrode), and returns the fixed
    matrix T = G @ pinv(G - mean_c(G)) @ (I - 1 1^T / C). The trailing centering
    term gives reference invariance: adding a constant to every channel does not
    change REST's output.

    rcond=1e-4 in the pseudo-inverse follows the published REST toolbox; numpy's
    default rcond was too aggressive on these small leadfields and added noise.
    """
    import mne
    with mne.use_log_level("ERROR"):
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

    C = G.shape[0]
    Ga = G - G.mean(axis=0, keepdims=True)
    pinvGa = np.linalg.pinv(Ga, rcond=1e-4)
    center = np.eye(C) - np.ones((C, C)) / C
    T = G @ pinvGa @ center
    return np.ascontiguousarray(T, dtype=np.float32)


@dataclass(frozen=True)
class DatasetGraph:
    """Per-dataset state the graph-dependent operators need.

    Built once per dataset (channel order is fixed across subjects in IV-2a).

    Fields
    ------
    ch_names       channel names, matching the channel axis of X.
    lap_small_idx  (C, k_small) indices of each channel's k nearest neighbours.
    lap_large_idx  (C, k_large_use) indices of the next ring of neighbours
                   (ranks k_large_skip .. k_large_skip + k_large_use).
    k_small        neighbour count for lap_small.
    k_large_skip   how many nearest neighbours lap_large skips.
    k_large_use    how many neighbours lap_large then averages.
    montage        MNE standard montage name used for positions.
    rest_matrix    (C, C) REST operator, or None if not requested.
    rest_cond      condition number of rest_matrix (diagnostic), or None.
    cz_idx         index of 'Cz' in ch_names (IV-2a always has Cz).
    """
    ch_names: List[str]
    lap_small_idx: np.ndarray
    lap_large_idx: np.ndarray
    k_small: int
    k_large_skip: int
    k_large_use: int
    montage: str
    rest_matrix: Optional[np.ndarray] = field(default=None)
    rest_cond: Optional[float] = field(default=None)
    cz_idx: Optional[int] = field(default=None)


def build_graph(
    ch_names: List[str],
    *,
    k_small: int = 4,
    k_large_skip: int = 4,
    k_large_use: int = 4,
    montage: str = "standard_1005",
    include_rest: bool = False,
) -> DatasetGraph:
    """Compute neighbour indices for the channel set; optionally the REST matrix.

    Neighbours come from 3D Euclidean distance between montage positions.
    lap_small uses the closest ``k_small``; lap_large uses the ring of
    ``k_large_use`` neighbours starting after the closest ``k_large_skip`` (so
    with both at 4, the two Laplacians use disjoint neighbour sets).

    include_rest builds the REST matrix (slow: a spherical forward model). Only
    pass it when 'rest' is among the operators you will actually apply.
    """
    k_small = int(k_small)
    k_large_skip = int(k_large_skip)
    k_large_use = int(k_large_use)

    xyz = _get_channel_positions(ch_names, montage=montage)
    d = _pairwise_distances(xyz)
    nn_sorted = np.argsort(d, axis=1)  # (C, C); self is inf so never selected
    lap_small = nn_sorted[:, :k_small].astype(np.int64)
    lap_large = nn_sorted[
        :, k_large_skip : k_large_skip + k_large_use
    ].astype(np.int64)

    rest_matrix = None
    rest_cond = None
    if include_rest:
        rest_matrix = _build_rest_matrix(ch_names, montage=montage)
        rest_cond = float(np.linalg.cond(rest_matrix))

    cz_idx = ch_names.index("Cz") if "Cz" in ch_names else None

    return DatasetGraph(
        ch_names=list(ch_names),
        lap_small_idx=lap_small,
        lap_large_idx=lap_large,
        k_small=k_small,
        k_large_skip=k_large_skip,
        k_large_use=k_large_use,
        montage=montage,
        rest_matrix=rest_matrix,
        rest_cond=rest_cond,
        cz_idx=cz_idx,
    )


# ---------------------------------------------------------------------------
# The operators. Each takes (N, C, T) and returns (N, C, T) float32.
# ---------------------------------------------------------------------------

def _check_3d(X: np.ndarray) -> np.ndarray:
    if X.ndim != 3:
        raise ValueError(f"Expected (N, C, T), got shape {X.shape}")
    return X.astype(np.float32, copy=False)


def _car(X: np.ndarray) -> np.ndarray:
    """Subtract the mean across channels at each time point."""
    X = _check_3d(X)
    return np.ascontiguousarray(X - X.mean(axis=1, keepdims=True), dtype=np.float32)


def _median(X: np.ndarray) -> np.ndarray:
    """Subtract the median across channels at each time point (robust CAR)."""
    X = _check_3d(X)
    return np.ascontiguousarray(X - np.median(X, axis=1, keepdims=True), dtype=np.float32)


def _lap_small(X: np.ndarray, lap_small_idx: np.ndarray) -> np.ndarray:
    """Subtract the mean of each channel's k nearest neighbours (Hjorth)."""
    X = _check_3d(X)
    ref = X[:, lap_small_idx].mean(axis=2)  # (N, C, k, T) -> (N, C, T)
    return np.ascontiguousarray(X - ref, dtype=np.float32)


def _lap_large(X: np.ndarray, lap_large_idx: np.ndarray) -> np.ndarray:
    """Subtract the mean of the next ring of neighbours (coarser Laplacian)."""
    X = _check_3d(X)
    ref = X[:, lap_large_idx].mean(axis=2)
    return np.ascontiguousarray(X - ref, dtype=np.float32)


def _cz_ref(X: np.ndarray, cz_idx: int) -> np.ndarray:
    """Subtract the Cz channel from every channel; Cz becomes zero."""
    X = _check_3d(X)
    return np.ascontiguousarray(X - X[:, cz_idx:cz_idx + 1, :], dtype=np.float32)


def _rest(X: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply the REST matrix. einsum avoids building a giant intermediate."""
    X = _check_3d(X)
    T = T.astype(np.float32, copy=False)
    return np.ascontiguousarray(np.einsum("ij,njt->nit", T, X), dtype=np.float32)


def apply_reference(
    X: np.ndarray,
    mode: str,
    graph: Optional[DatasetGraph] = None,
) -> np.ndarray:
    """Apply the named operator to X. ``graph`` is required for graph modes."""
    mode = str(mode).lower()
    if mode == "native":
        return _check_3d(X).copy()
    if mode == "car":
        return _car(X)
    if mode == "median":
        return _median(X)
    if mode not in GRAPH_MODES:
        raise ValueError(f"Unknown reference mode: {mode!r}. Known: {REFERENCE_MODES}")
    if graph is None:
        raise ValueError(f"Mode {mode!r} requires a DatasetGraph")
    if mode == "lap_small":
        return _lap_small(X, graph.lap_small_idx)
    if mode == "lap_large":
        return _lap_large(X, graph.lap_large_idx)
    if mode == "rest":
        if graph.rest_matrix is None:
            raise ValueError("Mode 'rest' requires build_graph(..., include_rest=True)")
        return _rest(X, graph.rest_matrix)
    # cz_ref
    if graph.cz_idx is None:
        raise ValueError("Mode 'cz_ref' requires a Cz channel in the montage.")
    return _cz_ref(X, graph.cz_idx)


class ReferenceTransformer(BaseEstimator, TransformerMixin):
    """sklearn transformer wrapping apply_reference, for the CSP+LDA pipeline.

    Stateless: fit does nothing, transform applies the operator. Sits at the
    front of the pipeline so the reference is swappable per experiment.
    """

    def __init__(self, mode: str, graph: Optional[DatasetGraph] = None):
        self.mode = mode
        self.graph = graph

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return apply_reference(X, self.mode, graph=self.graph)


# ---------------------------------------------------------------------------
# Euclidean Alignment (He & Wu 2020)
# ---------------------------------------------------------------------------
# EA is a per-subject whitening, applied AFTER the reference operator. For one
# block of trials it computes the average trial covariance R_bar, then
# left-multiplies every trial by R_bar^{-1/2}. After this, the average trial
# covariance is the identity, so each subject's second-order statistics are
# standardised. It is fit and applied on the same block (train and test are
# whitened independently; no leakage between them). This matches MIRepNet's
# released implementation.


def _ea_fit(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Estimate the EA whitener R_bar^{-1/2} from a block of trials.

    R_bar is the mean of per-trial sample covariances. A tiny ridge (eps) is
    added before the inverse square root so rank-reducing references (cz_ref,
    the Laplacians) don't make the matrix power return complex/NaN values.
    """
    from scipy.linalg import fractional_matrix_power

    X = _check_3d(X)
    N, C, T = X.shape
    if N == 0:
        raise ValueError("cannot fit EA on an empty block")

    cov = np.empty((N, C, C), dtype=np.float64)
    for i in range(N):
        cov[i] = np.cov(X[i].astype(np.float64))
    R_bar = cov.mean(axis=0)

    if eps:
        R_bar = R_bar + eps * np.eye(C, dtype=np.float64)

    R_inv_sqrt = fractional_matrix_power(R_bar, -0.5)
    return np.real(R_inv_sqrt).astype(np.float64)


def _ea_apply(X: np.ndarray, whitener: np.ndarray) -> np.ndarray:
    """Left-multiply every (C, T) trial by a precomputed (C, C) whitener."""
    X = _check_3d(X)
    if X.shape[0] == 0:
        return X.copy()
    out = np.einsum("ij,njt->nit", whitener, X.astype(np.float64))
    return np.ascontiguousarray(out, dtype=np.float32)


def euclidean_alignment(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Euclidean-align a block of trials. Input/output (N, C, T) float32.

    Equivalent to _ea_apply(X, _ea_fit(X)): fit the whitener on this block and
    apply it to the same block.
    """
    X = _check_3d(X)
    if X.shape[0] == 0:
        return X.copy()
    return _ea_apply(X, _ea_fit(X, eps=eps))


def apply_reference_then_ea(
    X: np.ndarray,
    mode: str,
    graph: Optional[DatasetGraph] = None,
    *,
    apply_ea: bool = False,
    ea_eps: float = 1e-12,
) -> np.ndarray:
    """Apply the reference operator, then optionally Euclidean-align.

    With apply_ea=False this is just apply_reference. With apply_ea=True the
    whitening is computed on the already-referenced trials, which is the
    ordering that answers "does EA absorb the reference shift?"
    """
    Y = apply_reference(X, mode, graph=graph)
    if apply_ea:
        Y = euclidean_alignment(Y, eps=ea_eps)
    return Y
