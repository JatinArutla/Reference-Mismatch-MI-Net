"""Operator invertibility: a data-free algebraic check on the reference operators.

This underpins the global-vs-spatial claim. Two questions per operator, using the
contrast projector H = I - 11^T/C (CAR):

  1. Is it canonicalizable by re-referencing?  i.e. does it preserve channel
     contrasts, H M = H?  Global re-references (native, CAR, single-electrode,
     REST) do; so re-referencing their output to CAR collapses them to one point
     (this is the car_after experiment). Laplacians do NOT (H M != H).

  2. If it does not preserve contrasts, are the native contrasts still *linearly
     recoverable* from the operator's output?  A Laplacian whose only null
     direction is the constant vector is invertible on the contrast subspace, so
     the contrasts are recoverable and the only cost is conditioning. This
     distinguishes "spatial operators change the coordinate system but lose no
     contrast information (recovery is just ill-conditioned)" from "spatial
     operators destroy contrast information", which are very different claims.

The report returns both facts plus the condition number, so the paper states the
true one instead of assuming it. 'median' is nonlinear (no fixed matrix) and is
excluded.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from refshift.references import (
    GRAPH_MODES,
    REFERENCE_MODES,
    apply_reference,
    build_graph,
)


def operator_matrix(mode: str, graph, n_channels: int) -> np.ndarray:
    """Recover the C x C matrix M with operator(x) = M @ x by feeding the identity.

    Feeding a batch whose single trial is the identity (channel c is the basis
    vector at time c) returns M directly, since each operator acts as M @ x at
    every time point. Linear operators only; 'median' has no fixed matrix.
    """
    eye = np.eye(n_channels, dtype=np.float32)[None]  # (1, C, C)
    return apply_reference(eye, mode, graph=graph)[0].astype(np.float64)


def contrast_recovery_report(
    ch_names: Sequence[str],
    *,
    modes: Optional[Sequence[str]] = None,
    montage: str = "standard_1005",
) -> pd.DataFrame:
    """Per-operator canonicalizability and contrast recoverability on a montage.

    Columns: contrast_preserving (H M = H, i.e. canonicalizable by re-referencing),
    contrasts_recoverable (native contrasts linearly recoverable from the output),
    cond_contrast (condition number of inverting the operator on the contrast
    subspace -- ~1 is trivial, large is ill-conditioned/noise-amplifying),
    recover_residual (||H - H M^+ M||, ~0 means exactly recoverable).

    Conditioning is measured on the (C-1)-dim contrast subspace, which removes the
    constant null direction analytically for every operator (so CAR reads as the
    trivial cond=1 it is, not a float artifact of its near-zero null singular value).
    """
    C = len(ch_names)
    H = np.eye(C) - np.ones((C, C)) / C
    # Orthonormal basis of the contrast subspace (range of the projector H).
    w, V = np.linalg.eigh(H)
    U = V[:, w > 0.5]  # C x (C-1)

    if modes is None:
        modes = REFERENCE_MODES
    modes = [m for m in modes if m != "median"]  # nonlinear, no fixed matrix
    graph = (
        build_graph(list(ch_names), montage=montage, include_rest=("rest" in modes))
        if any(m in GRAPH_MODES for m in modes) else None
    )

    rows: List[dict] = []
    for m in modes:
        M = operator_matrix(m, graph, C)
        Mc = U.T @ M @ U  # operator restricted to the contrast subspace
        sc = np.linalg.svd(Mc, compute_uv=False)
        tol = sc.max() * len(sc) * np.finfo(float).eps
        rank_c = int((sc > tol).sum())
        cond_c = float(sc[0] / sc[rank_c - 1]) if rank_c else float("inf")
        preserving = float(np.linalg.norm(H @ M - H))
        recover_residual = float(np.linalg.norm(H - H @ np.linalg.pinv(M) @ M))
        rows.append({
            "operator": m,
            "contrast_preserving": preserving < 1e-6,
            "contrasts_recoverable": recover_residual < 1e-6,
            "cond_contrast": round(cond_c, 2),
            "recover_residual": recover_residual,
        })
    return pd.DataFrame(rows)
