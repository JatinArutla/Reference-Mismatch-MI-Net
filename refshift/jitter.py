"""Per-sample reference jitter: the data-augmentation intervention.

During training, instead of fixing one reference, we draw a *different*
reference for each sample at each batch, uniformly from an allowed set. The
model therefore never sees a single fixed reference and is pushed to be
invariant to the choice. This module builds the braindecode Transform that
does the per-sample re-referencing; the experiment runner plugs it into the
training data loader.

The transform decodes each batch to numpy, groups samples by their drawn mode
(so each reference operator is still applied to a sub-batch in one vectorised
call), re-references, and re-uploads. The CPU round-trip is small next to the
GPU training cost.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch

from refshift.references import (
    GRAPH_MODES,
    REFERENCE_MODES,
    DatasetGraph,
    _zscore_trials,
    apply_reference,
)


def _random_reference_op(
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    modes: Sequence[str],
    graph: Optional[DatasetGraph],
):
    """Apply modes[i] to sample i. Group by mode to amortise the dispatch cost."""
    if X.ndim != 3:
        raise ValueError(f"Expected (B, C, T), got shape {tuple(X.shape)}")
    if len(modes) != X.shape[0]:
        raise ValueError(f"Got {len(modes)} modes for batch of size {X.shape[0]}")

    device = X.device
    X_np = X.detach().cpu().numpy().astype(np.float32, copy=False)
    out_np = np.empty_like(X_np)

    by_mode: dict = {}
    for i, m in enumerate(modes):
        by_mode.setdefault(m, []).append(i)
    for mode, idxs in by_mode.items():
        out_np[idxs] = apply_reference(X_np[idxs], mode, graph=graph)

    # Per-trial z-score after referencing, matching the fixed-reference runner.
    out_np = _zscore_trials(out_np)

    return torch.from_numpy(out_np).to(device, non_blocking=True), y


def make_random_reference_transform(
    allowed_modes: Sequence[str],
    *,
    graph: Optional[DatasetGraph] = None,
    random_state: Optional[int] = None,
):
    """Build the braindecode Transform that re-references each training sample.

    ``allowed_modes`` is the set of references to draw from. ``graph`` is
    required if any of them needs one (rest, cz_ref, the Laplacians); for
    'rest' the graph must have been built with include_rest=True.
    """
    from braindecode.augmentation import Transform

    allowed = tuple(str(m).lower() for m in allowed_modes)
    if not allowed:
        raise ValueError("allowed_modes must be non-empty")
    unknown = [m for m in allowed if m not in REFERENCE_MODES]
    if unknown:
        raise ValueError(f"Unknown reference modes: {unknown}")
    needs_graph = [m for m in allowed if m in GRAPH_MODES]
    if needs_graph and graph is None:
        raise ValueError(f"graph=None but allowed_modes includes {needs_graph}")
    if "rest" in allowed and (graph is None or graph.rest_matrix is None):
        raise ValueError("'rest' requires graph built with include_rest=True")
    if "cz_ref" in allowed and (graph is None or graph.cz_idx is None):
        raise ValueError("'cz_ref' requires Cz in the channel set; got cz_idx=None")

    class RandomReferenceTransform(Transform):
        operation = staticmethod(_random_reference_op)

        def __init__(self):
            super().__init__(probability=1.0, random_state=random_state)
            self._allowed_modes = allowed
            self._graph = graph
            self._mode_rng = np.random.default_rng(random_state)

        def get_augmentation_params(self, X, y):
            n = X.shape[0]
            idx = self._mode_rng.integers(0, len(self._allowed_modes), size=n)
            modes = [self._allowed_modes[i] for i in idx]
            return {"modes": modes, "graph": self._graph}

    return RandomReferenceTransform()
