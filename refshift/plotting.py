"""Mismatch-matrix heatmap plotting.

Single public function: ``plot_mismatch_matrix(df, out_path, **style)``.

Style is deliberate and fixed to match the paper figures:

- viridis colormap on a 0-100 scale (accuracy %)
- black rectangle around every diagonal cell (so within-reference cells
  stand out at a glance)
- rotated x-tick labels (test reference)
- per-cell numeric labels (auto black/white against cell luminance)
- colorbar on the right labeled "Accuracy (%)"

Only matplotlib is imported, and only lazily — ``refshift`` continues to
import cleanly on machines without matplotlib, with the error surfaced
only when someone actually calls ``plot_mismatch_matrix``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd

from refshift.reference import REFERENCE_MODES


def plot_mismatch_matrix(
    df: pd.DataFrame,
    out_path: Optional[str] = None,
    *,
    title: str = "CSP+LDA: Train x Test Reference Accuracy",
    metric: str = "accuracy",
    reference_order: Tuple[str, ...] = REFERENCE_MODES,
    figsize: Tuple[float, float] = (9, 6),
    dpi: int = 140,
    vmin: float = 0.0,
    vmax: float = 100.0,
):
    """Render a mismatch heatmap and optionally save it.

    Parameters
    ----------
    df : pandas.DataFrame
        Long-form results with at least columns ``train_ref``, ``test_ref``,
        and ``<metric>``. Accuracy is expected in [0, 1]; the plot shows
        percentages.
    out_path : str or None
        If given, the figure is saved to this path (PNG/PDF inferred from
        the extension). The figure is returned either way.
    title : str
        Figure title.
    metric : str
        Column in ``df`` to plot. Default 'accuracy'.
    reference_order : tuple of str
        Row/column order. Modes that are absent from ``df`` are dropped,
        preserving the given order among those that are present.
    figsize, dpi, vmin, vmax :
        Standard matplotlib knobs. vmin/vmax are in percent (so 0-100).

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    # Aggregate to mean over subjects/seeds.
    agg = df.groupby(["train_ref", "test_ref"])[metric].mean().unstack("test_ref")
    present = [m for m in reference_order if m in agg.index and m in agg.columns]
    if not present:
        raise ValueError(
            "No overlap between reference_order and df's train/test refs. "
            f"Got train_refs={sorted(agg.index)}, "
            f"test_refs={sorted(agg.columns)}, "
            f"reference_order={list(reference_order)}"
        )
    agg = agg.reindex(index=present, columns=present)
    M = agg.to_numpy() * 100.0

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    im = ax.imshow(M, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")

    # Per-cell labels, auto-contrast against cell luminance.
    mid = 0.5 * (vmin + vmax)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            val = M[i, j]
            color = "black" if val > mid else "white"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color=color, fontsize=10)

    # Outline the diagonal.
    for k in range(M.shape[0]):
        ax.add_patch(
            Rectangle((k - 0.5, k - 0.5), 1, 1,
                      fill=False, edgecolor="black", linewidth=2.0)
        )

    ax.set_xticks(range(len(present)))
    ax.set_yticks(range(len(present)))
    ax.set_xticklabels(present, rotation=45, ha="right")
    ax.set_yticklabels(present)
    ax.set_xlabel("Test reference")
    ax.set_ylabel("Train reference")
    ax.set_title(title)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Accuracy (%)")

    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, bbox_inches="tight", dpi=dpi)
    return fig
