"""Reference-jitter augmentation for Phase 2.

Per-sample randomized reference selection, implemented as a braindecode
``Transform`` so it composes with their ``AugmentedDataLoader`` and skorch's
``EEGClassifier`` without any custom training loop.

Two interventions are supported:

  - **full-jitter**: each training sample independently gets a reference
    drawn uniformly from all 7 modes. The model never sees a fixed reference;
    test-time evaluation under any reference is then "in-distribution" w.r.t.
    the training distribution of references.

  - **leave-one-out (LOFO)**: same as above but with one mode held out (e.g.
    `bipolar`). Test-time evaluation on the held-out mode is the cleanest
    distribution-shift probe — the model has never seen that operator, so
    transfer to it must come from invariance, not memorization.

Implementation note. Each call to the transform decodes the batch tensor to
numpy (B, C, T), applies the existing tested ``apply_reference`` per sample,
and re-uploads. The CPU round-trip adds ~30s per 200-epoch training on T4 —
a negligible cost compared to building a parallel GPU implementation that
would need its own validation against the numpy reference. Keep it simple.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch

from refshift.reference import REFERENCE_MODES, DatasetGraph, apply_reference


# ---------------------------------------------------------------------------
# Operation: applied to a batch of (B, C, T) tensors with one mode per sample
# ---------------------------------------------------------------------------

def _random_reference_op(
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    modes: Sequence[str],
    graph: Optional[DatasetGraph],
):
    """Apply a per-sample reference operator. Returns (X_transformed, y).

    ``modes`` must have length equal to ``X.shape[0]``. The reference operator
    for sample ``i`` is ``modes[i]``. ``graph`` is required iff any mode in
    ``modes`` is one of (laplacian, bipolar, rest).
    """
    if X.ndim != 3:
        raise ValueError(f"Expected (B, C, T) tensor, got shape {tuple(X.shape)}.")
    if len(modes) != X.shape[0]:
        raise ValueError(
            f"Got {len(modes)} modes for batch of size {X.shape[0]}."
        )

    device = X.device
    X_np = X.detach().cpu().numpy().astype(np.float32, copy=False)
    out_np = np.empty_like(X_np)

    # Group indices by mode to amortize the per-mode dispatch cost. The
    # underlying primitives like _car, _median, etc. are already vectorised
    # across the batch dimension.
    by_mode: dict[str, list[int]] = {}
    for i, m in enumerate(modes):
        by_mode.setdefault(m, []).append(i)
    for mode, idxs in by_mode.items():
        sub = X_np[idxs]
        out_np[idxs] = apply_reference(sub, mode, graph=graph)

    out_t = torch.from_numpy(out_np).to(device, non_blocking=True)
    return out_t, y


# ---------------------------------------------------------------------------
# Transform: braindecode-compatible, plugs into AugmentedDataLoader
# ---------------------------------------------------------------------------

def make_random_reference_transform(
    allowed_modes: Sequence[str],
    *,
    graph: Optional[DatasetGraph] = None,
    probability: float = 1.0,
    random_state: Optional[int] = None,
):
    """Construct a braindecode ``Transform`` that re-references each sample.

    Parameters
    ----------
    allowed_modes : sequence of str
        Subset of ``REFERENCE_MODES``. Each training sample independently gets
        one of these drawn uniformly. For the full-jitter condition pass
        ``REFERENCE_MODES``; for LOFO-bipolar pass the 6 modes excluding bipolar.
    graph : DatasetGraph or None
        Required if ``allowed_modes`` contains any of {laplacian, bipolar, rest}.
    probability : float, default 1.0
        Per-sample probability of applying the operation. With 1.0 the transform
        is deterministic in *that* it always re-references; the *which* reference
        is the random part. Reducing below 1.0 mixes original-reference samples
        in, which is unusual for this use case.
    random_state : int or None
        Seed for the per-sample mode sampler. Independent from braindecode's
        own ``probability`` mask RNG (which is seeded separately by Transform).

    Returns
    -------
    braindecode.augmentation.Transform subclass instance.
    """
    # Lazy import: jitter.py shouldn't drag braindecode into Phase 1 imports.
    from braindecode.augmentation import Transform

    allowed = tuple(m.lower() for m in allowed_modes)
    if not allowed:
        raise ValueError("allowed_modes must be non-empty.")
    unknown = [m for m in allowed if m not in REFERENCE_MODES]
    if unknown:
        raise ValueError(
            f"Unknown reference modes: {unknown}. Known: {REFERENCE_MODES}"
        )
    needs_graph = any(m in ("laplacian", "bipolar", "rest") for m in allowed)
    if needs_graph and graph is None:
        raise ValueError(
            "graph=None but allowed_modes includes a graph-requiring mode "
            f"({[m for m in allowed if m in ('laplacian', 'bipolar', 'rest')]})."
        )
    if "rest" in allowed and (graph is None or graph.rest_matrix is None):
        raise ValueError(
            "allowed_modes contains 'rest' but graph.rest_matrix is None. "
            "Build the graph with include_rest=True."
        )

    class RandomReferenceTransform(Transform):
        """Per-sample uniform-random reference operator transform."""

        operation = staticmethod(_random_reference_op)

        def __init__(self):
            super().__init__(probability=probability, random_state=random_state)
            self._allowed_modes = allowed
            self._graph = graph
            # Independent numpy Generator for mode sampling. Seed is derived
            # from random_state so that two transforms built with the same
            # seed produce identical training trajectories.
            self._mode_rng = np.random.default_rng(random_state)

        def get_augmentation_params(self, X, y):
            n = X.shape[0]
            idx = self._mode_rng.integers(0, len(self._allowed_modes), size=n)
            modes = [self._allowed_modes[i] for i in idx]
            return {"modes": modes, "graph": self._graph}

    return RandomReferenceTransform()
