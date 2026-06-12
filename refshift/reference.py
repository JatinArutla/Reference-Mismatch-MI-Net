"""Reference and spatial-derivative operators for EEG decoding.

Eight modes in three families:
    Global symmetric:     native, car, median, rest
    Global asymmetric:    cz_ref           (X_i - X_Cz)
    Local spatial-deriv:  lap_small        (Hjorth k=4 NN local Laplacian)
                          lap_large        (deterministic next-ring large-Laplacian
                                            approximation: ranks k_large_skip..
                                            k_large_skip+k_large_use of Euclidean
                                            neighbour distance)
                          csd              (Perrin spherical-spline surface Laplacian)

LOO-mean is omitted because LOO_i = (C/(C-1)) * CAR_i (scalar multiple of CAR;
identical for any scale-invariant decoder). GS is omitted because the natural
implementation is data-dependent and not a fixed C×C operator. NN-diff was
removed in v0.13: not a literature-recognised reference, and rank-deficient
on dense montages.

v0.15 / v0.16 changes:
    laplacian renamed to lap_small (still accepted via alias for old CSVs).
    lap_large added: deterministic next-ring large-Laplacian approximation
        with k_skip=4, k_use=4 by default. Motivated by McFarland 1997's
        large Laplacian but NOT equivalent to it on sparse montages: on
        IV-2a's 22-channel layout, ranks 4..7 are NOT a clean anatomical
        ring (e.g. Fz's "next ring" includes FC4, Cz, C2, C1, which are
        a mixture of frontal, midline and central channels). The operator
        is still well-defined and useful, but framing it as "the McFarland
        large Laplacian" overclaims equivalence on sparse montages. See
        KNOWN_LIMITATIONS.md.
    csd added via mne.preprocessing.compute_current_source_density;
        operator is recovered as a fixed C×C matrix via an identity-basis push
        (purely spatial, time-invariant by construction; verified empirically).
        Defaults match MNE: lambda2=1e-5, stiffness=4, n_legendre_terms=50,
        sphere='auto'.

Important caveat on CSD amplitude scale (v0.16):
    CSD output magnitude is ~10^2-10^3 times larger than other operators
    post-EMS on standard MI EEG. This is a unit/scale property of the
    spherical-spline operator, NOT a spatial-topology property. The raw
    Frobenius operator distance ||A_csd - A_other||_F is therefore
    dominated by CSD's scale rather than by spatial-derivative topology.
    Analyses that depend on operator distance should use scale-normalized
    variants (see operator_distance_correlation's distance_metric arg).
    Cross-reference transfer involving CSD also confounds spatial topology
    with amplitude scale; the run_pre_ems_mismatch runner provides an
    alternative pipeline (operator before standardization) that controls
    for this.

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
    with mne.use_log_level("ERROR"):
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
                         (used by lap_large / deterministic next-ring approximation).
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
        For lap_large (deterministic next-ring large-Laplacian approximation):
        each channel's neighbour set is the ring of ranks
        [k_large_skip..k_large_skip+k_large_use) in 3D Euclidean distance order.
        Defaults (4, 4) give disjoint neighbour sets between lap_small and
        lap_large when both use k=4. Motivated by McFarland 1997; on sparse
        montages the "ring" is approximate (see module docstring).
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
    """X - mean of the ring of neighbours skipping the closest k_large_skip.

    A deterministic next-ring large-Laplacian approximation, motivated by but
    not equivalent to McFarland 1997's dense-montage large Laplacian. On
    sparse montages the "ring" interpretation breaks down (see module
    docstring).
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


# ---------------------------------------------------------------------------
# Euclidean Alignment (He & Wu 2020), MIRepNet-exact form.
# ---------------------------------------------------------------------------
# EA is a per-subject spatial whitening, NOT a per-channel standardisation,
# so it does not belong in the data.py `normalization` rail (zscore/ems/none),
# which is per-channel and DL-only. EA is applied to windowed (N, C, T) trials
# AFTER the reference operator, in the mismatch loop, on both the CSP+LDA and
# DL paths. This module-level function is the single source of truth.
#
# Implementation matches MIRepNet's released `utils.EA` exactly:
#   R_bar = mean_i cov(X_i)          (per-trial sample covariance, then mean)
#   X_i  <- R_bar^{-1/2} @ X_i
# After EA, mean_i (X_i X_i^T) = I, so per-subject second-order statistics are
# matched to the identity. The reference covariance is computed over whatever
# set of trials is passed in: callers apply it per split (fit on the trials in
# hand), which is MIRepNet's own behaviour (train and test get independently
# estimated R_bar; no train->test leakage, but also no shared whitening).


def _ea_fit(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Estimate the EA whitener R_bar^{-1/2} from a block of trials.

    R_bar is the mean of per-trial sample covariances (np.cov per trial, then
    mean), matching MIRepNet. Returns the (C, C) inverse-square-root whitening
    matrix. Separated from application so a whitener fit on a small calibration
    subset can be applied to other trials (the low-target-data EA sweep).
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
    # Guard the rank-reducing operators (cz_ref, Laplacians): non-PSD R_bar can
    # make fractional_matrix_power return a ~0-imaginary complex array.
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

    Computes one reference covariance R_bar as the mean of per-trial sample
    covariances, then left-multiplies every trial by R_bar^{-1/2}. This is the
    He & Wu (2020) Euclidean alignment, in the exact form MIRepNet uses. This
    is fit-and-apply on the SAME block; it equals _ea_apply(X, _ea_fit(X)).

    Parameters
    ----------
    X : (N, C, T) array. The block of trials to align together (e.g. one
        subject's train split, or one subject's test split).
    eps : ridge added to R_bar's diagonal before the inverse-sqrt, guarding
        against a singular / ill-conditioned reference covariance on short or
        low-rank blocks (e.g. cz_ref, which zeroes a channel and so produces a
        rank-deficient covariance). MIRepNet's bare implementation omits this;
        we add a tiny ridge because some reference operators in this codebase
        (cz_ref, the Laplacians) are deliberately rank-reducing and would
        otherwise make fractional_matrix_power emit complex/NaN values.

    Returns
    -------
    (N, C, T) float32, Euclidean-aligned.
    """
    X = _check_3d(X)
    if X.shape[0] == 0:
        return X.copy()
    return _ea_apply(X, _ea_fit(X, eps=eps))


def stratified_calibration_index(
    y: np.ndarray, k_per_class: int, *, seed: int = 0
) -> np.ndarray:
    """Indices of k trials per class, drawn reproducibly.

    Used to carve a small calibration subset out of the test block for the
    low-target-data EA sweep: these trials estimate R_bar and are then EXCLUDED
    from scoring. If a class has fewer than k_per_class trials, all of its
    trials are taken; the runner reports the realized calibration count.
    """
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    picks: List[int] = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        picks.extend(idx[:k_per_class].tolist())
    return np.asarray(sorted(picks), dtype=int)


def apply_reference_then_ea(
    X: np.ndarray,
    mode: str,
    graph: Optional[DatasetGraph] = None,
    *,
    apply_ea: bool = False,
    ea_eps: float = 1e-12,
) -> np.ndarray:
    """Apply the reference operator, then (optionally) Euclidean-align.

    Convenience wrapper used by the mismatch runner so the reference->EA
    ordering lives in one place. With apply_ea=False this is exactly
    apply_reference(X, mode, graph); with apply_ea=True the EA whitening is
    computed on the *referenced* trials, which is the ordering that answers
    "does EA absorb the reference shift" (reference first, then whiten the
    already-referenced covariance).
    """
    Y = apply_reference(X, mode, graph=graph)
    if apply_ea:
        Y = euclidean_alignment(Y, eps=ea_eps)
    return Y
