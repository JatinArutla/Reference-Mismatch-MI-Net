"""Per-sample reference jitter for Phase 2.

Each training sample independently gets a reference drawn uniformly from
allowed_modes. Used in two conditions:

  full-jitter: allowed_modes = REFERENCE_MODES. The model never sees a
    fixed reference; test-time accuracy on any reference is in-distribution.

  LOFO: allowed_modes = REFERENCE_MODES \\ {holdout}. Test on the holdout
    is the cleanest invariance probe.

Implementation: braindecode Transform plugged into AugmentedDataLoader. The
operation decodes (B, C, T) to numpy, calls apply_reference per sub-batch
grouped by mode (so the underlying primitives stay vectorised), and re-uploads.
The CPU round-trip is ~30 s per 200-epoch training on T4 — small relative to
training cost, and avoids a parallel GPU implementation that would need its
own validation against the numpy reference.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch

from refshift.reference import (
    REFERENCE_MODES,
    _GRAPH_MODES,
    DatasetGraph,
    apply_reference,
)


def _random_reference_op(
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    modes: Sequence[str],
    graph: Optional[DatasetGraph],
):
    """Apply mode[i] to sample i. Group by mode to amortise the dispatch cost."""
    if X.ndim != 3:
        raise ValueError(f"Expected (B, C, T), got shape {tuple(X.shape)}")
    if len(modes) != X.shape[0]:
        raise ValueError(f"Got {len(modes)} modes for batch of size {X.shape[0]}")

    device = X.device
    X_np = X.detach().cpu().numpy().astype(np.float32, copy=False)
    out_np = np.empty_like(X_np)

    by_mode: dict[str, list[int]] = {}
    for i, m in enumerate(modes):
        by_mode.setdefault(m, []).append(i)
    for mode, idxs in by_mode.items():
        out_np[idxs] = apply_reference(X_np[idxs], mode, graph=graph)

    return torch.from_numpy(out_np).to(device, non_blocking=True), y


def make_random_reference_transform(
    allowed_modes: Sequence[str],
    *,
    graph: Optional[DatasetGraph] = None,
    probability: float = 1.0,
    random_state: Optional[int] = None,
):
    """braindecode Transform that re-references each training sample.

    allowed_modes is a subset of REFERENCE_MODES (the legacy alias 'laplacian'
    is accepted and resolved to 'lap_small'). graph is required if any mode
    in allowed_modes needs one (lap_small, lap_large, rest, csd, cz_ref).
    For 'rest' the graph must be built with include_rest=True; for 'csd'
    with include_csd=True; for 'cz_ref' graph.cz_idx must be set.
    """
    from braindecode.augmentation import Transform

    from refshift.reference import _resolve_alias

    allowed = tuple(_resolve_alias(m) for m in allowed_modes)
    if not allowed:
        raise ValueError("allowed_modes must be non-empty")
    unknown = [m for m in allowed if m not in REFERENCE_MODES]
    if unknown:
        raise ValueError(f"Unknown reference modes: {unknown}")
    needs_graph = [m for m in allowed if m in _GRAPH_MODES]
    if needs_graph and graph is None:
        raise ValueError(f"graph=None but allowed_modes includes {needs_graph}")
    if "rest" in allowed and (graph is None or graph.rest_matrix is None):
        raise ValueError("'rest' requires graph built with include_rest=True")
    if "csd" in allowed and (graph is None or graph.csd_matrix is None):
        raise ValueError("'csd' requires graph built with include_csd=True")
    if "cz_ref" in allowed and (graph is None or graph.cz_idx is None):
        raise ValueError("'cz_ref' requires Cz in the channel set; got cz_idx=None")

    class RandomReferenceTransform(Transform):
        operation = staticmethod(_random_reference_op)

        def __init__(self):
            super().__init__(probability=probability, random_state=random_state)
            self._allowed_modes = allowed
            self._graph = graph
            self._mode_rng = np.random.default_rng(random_state)

        def get_augmentation_params(self, X, y):
            n = X.shape[0]
            idx = self._mode_rng.integers(0, len(self._allowed_modes), size=n)
            modes = [self._allowed_modes[i] for i in idx]
            return {"modes": modes, "graph": self._graph}

    return RandomReferenceTransform()
