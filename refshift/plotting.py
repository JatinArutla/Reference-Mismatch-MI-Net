"""Mismatch-matrix heatmap. viridis 0-100, diagonal cells boxed in black,
per-cell numeric labels with auto contrast.
"""

from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd

from refshift.reference import REFERENCE_MODES


def plot_mismatch_matrix(
    df: pd.DataFrame,
    out_path: Optional[str] = None,
    *,
    title: str = "Train x Test Reference Accuracy",
    metric: str = "accuracy",
    reference_order: Tuple[str, ...] = REFERENCE_MODES,
    figsize: Tuple[float, float] = (9, 6),
    dpi: int = 140,
    vmin: float = 0.0,
    vmax: float = 100.0,
):
    """Render a heatmap from a long-form mismatch DataFrame. Returns the figure.

    Aggregates across (subject, seed). Modes absent from df are dropped while
    preserving the requested order. vmin/vmax in percent; df accuracy in [0, 1].
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    agg = df.groupby(["train_ref", "test_ref"])[metric].mean().unstack("test_ref")
    present = [m for m in reference_order if m in agg.index and m in agg.columns]
    if not present:
        raise ValueError(
            f"No overlap between reference_order and df. "
            f"Got rows={sorted(agg.index)}, cols={sorted(agg.columns)}, "
            f"order={list(reference_order)}"
        )
    agg = agg.reindex(index=present, columns=present)
    M = agg.to_numpy() * 100.0

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    im = ax.imshow(M, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")

    mid = 0.5 * (vmin + vmax)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            color = "black" if M[i, j] > mid else "white"
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    color=color, fontsize=10)

    for k in range(M.shape[0]):
        ax.add_patch(Rectangle(
            (k - 0.5, k - 0.5), 1, 1,
            fill=False, edgecolor="black", linewidth=2.0,
        ))

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
