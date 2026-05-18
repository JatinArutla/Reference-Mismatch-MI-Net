"""Reference and spatial-derivative operators for EEG decoding.

Eight modes in three families:
    Global symmetric:     native, car, median, rest
    Global asymmetric:    cz_ref           (X_i - X_Cz)
    Local spatial-deriv:  lap_small        (Hjorth k=4 NN local Laplacian)
                          lap_large        (McFarland next-ring skip-NN Laplacian)
                          csd              (Perrin spherical-spline surface Laplacian)

LOO-mean is omitted because LOO_i = (C/(C-1)) * CAR_i (scalar multiple of CAR;
identical for any scale-invariant decoder). GS is omitted because the natural
implementation is data-dependent and not a fixed C×C operator. NN-diff was
removed in v0.13: not a literature-recognised reference, and rank-deficient
on dense montages.

v0.15 changes:
    laplacian renamed to lap_small (still accepted via alias for old CSVs).
    lap_large added with k_skip=4, k_use=4 (McFarland 1997 next-ring variant);
        on dense h-suffix montages (Schirrmeister motor subset) this produces
        a natural fine-vs-coarse scale separation against lap_small.
    csd added via mne.preprocessing.compute_current_source_density;
        operator is recovered as a fixed C×C matrix via an identity-basis push
        (purely spatial, time-invariant by construction; verified empirically).
        Defaults match MNE: lambda2=1e-5, stiffness=4, n_legendre_terms=50,
        sphere='auto'.

All operators take (N, C, T) float arrays. Channel order must match the
graph's ch_names. Graphs are computed once per dataset via build_graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


REFERENCE_MODES = (
    "native", "car", "median", "rest", "cz_ref",
    "lap_small", "lap_large", "csd",
)

# Modes that require precomputed dataset state (neighbour indices, REST
# matrix, CSD matrix, or Cz channel index). Single source of truth for
# "build a graph?".
_GRAPH_MODES = ("lap_small", "lap_large", "rest", "csd", "cz_ref")

# Legacy mode name aliases. Old CSVs and notebooks may reference 'laplacian';
# we accept it and resolve to 'lap_small' (same operator, renamed in v0.15).
_MODE_ALIASES = {
    "laplacian": "lap_small",
}


def _resolve_alias(mode: str) -> str:
    """Map legacy mode names to canonical ones. Case-insensitive."""
    mode_lc = mode.lower()
    return _MODE_ALIASES.get(mode_lc, mode_lc)


def canonical_mode_tuple(refs) -> tuple:
    """Resolve aliases and reorder an iterable of references to canonical order.

    Accepts any iterable (set, list, tuple, frozenset). Output is a tuple in
    REFERENCE_MODES order. Raises on unknown modes after alias resolution.

    Used by experiment runners to normalize user-supplied reference_modes,
    so passing reference_modes={"native","csd","car"} produces a deterministic
    ("native","car","csd") matrix layout regardless of set iteration order.
    """
    resolved = []
    for m in refs:
        canonical = _resolve_alias(m)
        if canonical not in REFERENCE_MODES:
            raise ValueError(
                f"Unknown reference mode: {m!r} (resolved to {canonical!r}). "
                f"Known: {REFERENCE_MODES}"
            )
        resolved.append(canonical)
    resolved_set = set(resolved)
    return tuple(m for m in REFERENCE_MODES if m in resolved_set)


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
    modes_t = tuple(_resolve_alias(m) for m in modes)
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

    Catches the most common footguns: cz_ref requested for a channel set
    without Cz (e.g. Schirrmeister2017); 'rest' or 'csd' requested without
    the corresponding precomputed matrix on the graph.
    """
    modes_t = tuple(_resolve_alias(m) for m in modes)
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
    if "csd" in modes_t and graph is not None and graph.csd_matrix is None:
        raise ValueError(
            "reference_modes contains 'csd' but graph was built with "
            "include_csd=False. Build with build_graph(..., include_csd=True)."
        )
    needs_graph = [m for m in modes_t if m in _GRAPH_MODES]
    if needs_graph and graph is None:
        raise ValueError(
            f"reference_modes={modes_t} includes graph-requiring modes "
            f"{needs_graph} but graph=None."
        )


# ---------------------------------------------------------------------------
# Channel positions, neighbour graph, REST matrix, CSD matrix
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


def _build_csd_matrix(
    ch_names: List[str],
    montage: str,
    *,
    sfreq: float = 250.0,
    lambda2: float = 1e-5,
    stiffness: int = 4,
    n_legendre_terms: int = 50,
) -> np.ndarray:
    """CSD (Perrin et al. 1989 spherical-spline surface Laplacian) transform matrix.

    MNE's compute_current_source_density is purely spatial. We recover the
    fixed C×C operator A by feeding an identity basis (C single-sample
    "epochs" each with one channel set to 1, others to 0) through it and
    reading the output: column j of A is the output for basis vector e_j.

    This matrix-cached form is mathematically identical to applying
    compute_current_source_density per-epoch (verified empirically: relative
    max diff ~1e-16 on float64 random data, machine precision). It's much
    faster than re-invoking MNE for every reference op call.

    Defaults match MNE: lambda2=1e-5 (regularization), stiffness=4 (spline
    order m), n_legendre_terms=50, sphere='auto' (fit to digitization).
    """
    import mne
    prev_log_level = mne.get_config("MNE_LOGGING_LEVEL")
    mne.set_log_level("ERROR")
    try:
        C = len(ch_names)
        # Shape (C, C, 1): epoch i has X[i, j, 0] = delta_ij. Use float64
        # internally so the recovered matrix is machine-precision accurate.
        X_basis = np.eye(C, dtype=np.float64)[:, :, None]
        info = mne.create_info(
            ch_names=list(ch_names), sfreq=float(sfreq), ch_types="eeg",
        )
        info.set_montage(
            mne.channels.make_standard_montage(montage), on_missing="raise",
        )
        epochs = mne.EpochsArray(X_basis, info, verbose="ERROR")
        epochs_csd = mne.preprocessing.compute_current_source_density(
            epochs, sphere="auto",
            lambda2=float(lambda2),
            stiffness=int(stiffness),
            n_legendre_terms=int(n_legendre_terms),
            copy=True, verbose="ERROR",
        )
        # MNE output shape: (C basis epochs, C output channels, 1 time).
        # epochs_csd[i, :, 0] is A @ e_i = A[:, i], so stack as columns
        # gives A = epochs_csd[..., 0].T
        A = epochs_csd.get_data()[:, :, 0].T
    finally:
        if prev_log_level is not None:
            mne.set_log_level(prev_log_level)
    return np.ascontiguousarray(A, dtype=np.float32)


@dataclass(frozen=True)
class DatasetGraph:
    """Pre-computed per-dataset state for graph-aware operators.

    Fields:
        ch_names         channel name list (must match channel axis of X)
        lap_small_idx    (C, k_small) int64: indices of k_small nearest neighbours
                         per channel (used by lap_small / Hjorth Laplacian).
        lap_large_idx    (C, k_large_use) int64: indices of the ring of
                         neighbours at ranks [k_large_skip .. k_large_skip+k_large_use)
                         (used by lap_large / McFarland next-ring Laplacian).
        k_small          number of nearest neighbours for lap_small.
        k_large_skip     number of nearest neighbours to skip before lap_large.
        k_large_use      number of neighbours used by lap_large.
        montage          MNE standard montage name used to compute positions.
        rest_matrix      (C, C) float32: precomputed REST operator
                         (None if include_rest=False at build_graph).
        rest_cond        condition number of rest_matrix.
        csd_matrix       (C, C) float32: precomputed CSD operator
                         (None if include_csd=False at build_graph).
        csd_cond         condition number of csd_matrix.
        cz_idx           index of 'Cz' in ch_names, or None if Cz is absent
                         (e.g. Schirrmeister2017 motor subset).

    Legacy aliases (read-only properties):
        laplacian_idx -> lap_small_idx
        k             -> k_small
    """
    ch_names: List[str]
    lap_small_idx: np.ndarray  # (C, k_small) int64
    lap_large_idx: np.ndarray  # (C, k_large_use) int64
    k_small: int
    k_large_skip: int
    k_large_use: int
    montage: str
    rest_matrix: Optional[np.ndarray] = field(default=None)
    rest_cond: Optional[float] = field(default=None)
    csd_matrix: Optional[np.ndarray] = field(default=None)
    csd_cond: Optional[float] = field(default=None)
    cz_idx: Optional[int] = field(default=None)

    @property
    def laplacian_idx(self) -> np.ndarray:
        """Backward-compat alias for lap_small_idx (the v0.14 field name)."""
        return self.lap_small_idx

    @property
    def k(self) -> int:
        """Backward-compat alias for k_small (the v0.14 field name)."""
        return self.k_small


def build_graph(
    ch_names: List[str],
    k_small: int = 4,
    k_large_skip: int = 4,
    k_large_use: int = 4,
    montage: str = "standard_1005",
    *,
    k: Optional[int] = None,
    include_rest: bool = False,
    include_csd: bool = False,
    csd_sfreq: float = 250.0,
    csd_lambda2: float = 1e-5,
    csd_stiffness: int = 4,
    csd_n_legendre_terms: int = 50,
) -> DatasetGraph:
    """Build neighbour indices for the given channel set; optionally REST and CSD.

    Parameters
    ----------
    ch_names :
        Channel names in the order matching the channel axis of X.
    k_small :
        Number of nearest neighbours for lap_small (Hjorth local Laplacian).
        Default 4. Pass via the legacy alias `k=` for backwards compatibility.
    k_large_skip, k_large_use :
        For lap_large (McFarland-style next-ring Laplacian): each channel's
        neighbour set is the ring of ranks [k_large_skip .. k_large_skip+k_large_use)
        in 3D Euclidean distance order. Defaults (4, 4) give disjoint
        neighbour sets between lap_small and lap_large when both use k=4.
    montage :
        MNE standard montage name.
    include_rest :
        If True, compute and store the REST (C, C) operator matrix.
        Adds 10-60s per dataset (spherical forward model is the slow part).
    include_csd :
        If True, compute and store the CSD (C, C) operator matrix via
        mne.preprocessing.compute_current_source_density on an identity basis.
        Adds ~1-3s per dataset.
    csd_sfreq, csd_lambda2, csd_stiffness, csd_n_legendre_terms :
        CSD parameters. Defaults match MNE 1.8+.
    """
    # Backward compat: accept legacy `k=` kwarg.
    if k is not None:
        k_small = int(k)
    k_small = int(k_small)
    k_large_skip = int(k_large_skip)
    k_large_use = int(k_large_use)

    xyz = _get_channel_positions(ch_names, montage=montage)
    d = _pairwise_distances(xyz)
    nn_sorted = np.argsort(d, axis=1)  # (C, C); diag is inf so self never selected
    lap_small = nn_sorted[:, :k_small].astype(np.int64)
    lap_large = nn_sorted[
        :, k_large_skip : k_large_skip + k_large_use
    ].astype(np.int64)

    rest_matrix = None
    rest_cond = None
    if include_rest:
        rest_matrix = _build_rest_matrix(ch_names, montage=montage)
        rest_cond = float(np.linalg.cond(rest_matrix))

    csd_matrix = None
    csd_cond = None
    if include_csd:
        csd_matrix = _build_csd_matrix(
            ch_names, montage=montage,
            sfreq=csd_sfreq, lambda2=csd_lambda2,
            stiffness=csd_stiffness, n_legendre_terms=csd_n_legendre_terms,
        )
        csd_cond = float(np.linalg.cond(csd_matrix))

    cz_idx = ch_names.index("Cz") if "Cz" in ch_names else None

    return DatasetGraph(
        ch_names=list(ch_names),
        lap_small_idx=lap_small,
        lap_large_idx=lap_large,
        k_small=k_small,
        k_large_skip=k_large_skip,
        k_large_use=k_large_use,
        montage=montage,
        rest_matrix=rest_matrix, rest_cond=rest_cond,
        csd_matrix=csd_matrix, csd_cond=csd_cond,
        cz_idx=cz_idx,
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


def _lap_small(X: np.ndarray, lap_small_idx: np.ndarray) -> np.ndarray:
    """X - mean of k_small nearest spatial neighbours (Hjorth local Laplacian)."""
    X = _check_3d(X)
    ref = X[:, lap_small_idx].mean(axis=2)  # (N, C, k, T) -> (N, C, T)
    return np.ascontiguousarray(X - ref, dtype=np.float32)


# Legacy function name kept for any external code that imported it directly.
_laplacian = _lap_small


def _lap_large(X: np.ndarray, lap_large_idx: np.ndarray) -> np.ndarray:
    """X - mean of the ring of neighbours skipping the closest k_large_skip
    (McFarland 1997 next-ring large Laplacian).
    """
    X = _check_3d(X)
    ref = X[:, lap_large_idx].mean(axis=2)
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


def _csd(X: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Apply CSD: same einsum form as REST (CSD is a fixed C×C linear op)."""
    X = _check_3d(X)
    A = A.astype(np.float32, copy=False)
    return np.ascontiguousarray(np.einsum("ij,njt->nit", A, X), dtype=np.float32)


def apply_reference(
    X: np.ndarray,
    mode: str,
    graph: Optional[DatasetGraph] = None,
) -> np.ndarray:
    """Dispatch X through the named operator. graph is required for graph modes.

    Accepts the legacy mode name 'laplacian' as an alias for 'lap_small'.
    """
    mode = _resolve_alias(mode)
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
    if mode == "lap_small":
        return _lap_small(X, graph.lap_small_idx)
    if mode == "lap_large":
        return _lap_large(X, graph.lap_large_idx)
    if mode == "rest":
        if graph.rest_matrix is None:
            raise ValueError("Mode 'rest' requires build_graph(..., include_rest=True)")
        return _rest(X, graph.rest_matrix)
    if mode == "csd":
        if graph.csd_matrix is None:
            raise ValueError("Mode 'csd' requires build_graph(..., include_csd=True)")
        return _csd(X, graph.csd_matrix)
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

    Accepts the legacy mode name 'laplacian' as an alias for 'lap_small'.
    """

    def __init__(self, mode: str, graph: Optional[DatasetGraph] = None):
        self.mode = mode
        self.graph = graph

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return apply_reference(X, self.mode, graph=self.graph)
