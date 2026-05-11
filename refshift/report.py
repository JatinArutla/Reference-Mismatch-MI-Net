"""Notebook reporting helper: print matrix, save CSV, save heatmap, in one call.

`report_experiment(df, kind=...)` handles the five DataFrame shapes the
experiment runners produce. It exists so notebooks don't reimplement the
same print/save logic for every cell. Each kind has its own matrix layout,
heatmap type, and summary stats; this module is the single source of truth
for how each one is rendered.

Usage:

    from refshift import run_mismatch
    from refshift.report import report_experiment

    df = run_mismatch("iv2a", model="csp_lda", seeds=[0])
    report_experiment(
        df, kind="mismatch",
        name="iv2a_csp_lda",
        results_dir="results/", figs_dir="figs/",
        dataset="iv2a", title="CSP+LDA iv2a",
    )

Returns a dict with the printed matrix and summary stats so the notebook
can do follow-up analysis without re-deriving them.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from refshift.plotting import plot_mismatch_matrix
from refshift.reference import REFERENCE_MODES, reference_modes_for_dataset


_VALID_KINDS = ("mismatch", "jitter_full", "lofo", "ems_diag", "bandpass")


def report_experiment(
    df: pd.DataFrame,
    *,
    kind: str,
    name: str,
    results_dir: str,
    figs_dir: str,
    dataset: Optional[str] = None,
    title: Optional[str] = None,
    modes: Optional[Tuple[str, ...]] = None,
    save_csv: bool = True,
    save_heatmap: bool = True,
    print_matrix: bool = True,
) -> Dict[str, Any]:
    """Print the natural matrix for this experiment, save CSV, save heatmap.

    Parameters
    ----------
    df : pd.DataFrame
        Long-form output from one of the experiment runners.
    kind : str
        One of 'mismatch', 'jitter_full', 'lofo', 'ems_diag', 'bandpass'.
        Determines how `df` is pivoted, what summary stats are computed,
        and which heatmap shape is rendered.
    name : str
        Stem for the saved files: {results_dir}/{name}.csv and
        {figs_dir}/{name}.png. No extension.
    results_dir, figs_dir : str
        Output directories. Created if missing.
    dataset : str, optional
        Dataset id (e.g. 'iv2a'). Used to auto-resolve `modes` if not given.
    title : str, optional
        Heatmap/header title. Defaults to a short description of the kind.
    modes : tuple of str, optional
        Reference order for matrix axes. Defaults to
        reference_modes_for_dataset(dataset) or REFERENCE_MODES.
    save_csv, save_heatmap, print_matrix : bool
        Toggles for each output. All default True.

    Returns
    -------
    dict
        Keys: 'matrix' (pd.DataFrame), 'summary' (dict of stats),
        'csv_path' (str or None), 'fig_path' (str or None).
    """
    if kind not in _VALID_KINDS:
        raise ValueError(f"Unknown kind: {kind!r}. Valid: {_VALID_KINDS}")
    if save_csv:
        os.makedirs(results_dir, exist_ok=True)
    if save_heatmap:
        os.makedirs(figs_dir, exist_ok=True)

    if modes is None:
        if dataset is not None:
            modes = reference_modes_for_dataset(dataset)
        else:
            modes = REFERENCE_MODES

    title = title or f"{kind} — {name}"

    csv_path = None
    if save_csv:
        csv_path = os.path.join(results_dir, f"{name}.csv")
        df.to_csv(csv_path, index=False)

    if kind == "mismatch":
        result = _report_mismatch(df, modes, title, print_matrix)
    elif kind == "jitter_full":
        result = _report_jitter_full(df, modes, title, print_matrix)
    elif kind == "lofo":
        result = _report_lofo(df, modes, title, print_matrix)
    elif kind == "ems_diag":
        result = _report_ems(df, modes, title, print_matrix)
    elif kind == "bandpass":
        result = _report_bandpass(df, title, print_matrix)
    else:  # unreachable; guarded above
        raise AssertionError

    fig_path = None
    if save_heatmap:
        fig_path = os.path.join(figs_dir, f"{name}.png")
        _save_figure_for_kind(df, kind, modes, title, fig_path)

    result["csv_path"] = csv_path
    result["fig_path"] = fig_path
    return result


# ---------------------------------------------------------------------------
# Per-kind matrix/stat extractors
# ---------------------------------------------------------------------------

def _report_mismatch(
    df: pd.DataFrame, modes: Tuple[str, ...], title: str, do_print: bool,
) -> Dict[str, Any]:
    from refshift.experiments.mismatch import mismatch_matrix
    M = mismatch_matrix(df)
    present = [m for m in modes if m in M.index and m in M.columns]
    M = M.reindex(index=present, columns=present)
    arr = M.to_numpy()
    diag = float(np.nanmean(np.diag(arr)))
    off = float(np.nanmean(arr[~np.eye(arr.shape[0], dtype=bool)]))
    summary = {"diagonal_mean": diag, "off_diag_mean": off, "gap": diag - off}
    if do_print:
        print(f"\n=== {title} ===")
        print(M.round(3).to_string())
        print(f"\ndiagonal mean : {diag:.4f}")
        print(f"off-diag mean : {off:.4f}")
        print(f"gap           : {diag - off:+.4f}")
    return {"matrix": M, "summary": summary}


def _report_jitter_full(
    df: pd.DataFrame, modes: Tuple[str, ...], title: str, do_print: bool,
) -> Dict[str, Any]:
    means = df.groupby("test_ref")["accuracy"].mean()
    present = [m for m in modes if m in means.index]
    means = means.reindex(present)
    M = means.round(3).to_frame().T
    M.index = ["jitter_full"]
    mean = float(means.mean())
    std = float(means.std(ddof=0))
    summary = {"mean": mean, "std": std}
    if do_print:
        print(f"\n=== {title} ===")
        print(M.to_string())
        print(f"\nmean : {mean:.3f}")
        print(f"std  : {std:.3f}")
    return {"matrix": M, "summary": summary}


def _report_lofo(
    df: pd.DataFrame, modes: Tuple[str, ...], title: str, do_print: bool,
) -> Dict[str, Any]:
    M = (df.groupby(["holdout_ref", "test_ref"])["accuracy"]
           .mean().unstack())
    present_rows = [m for m in modes if m in M.index]
    present_cols = [m for m in modes if m in M.columns]
    M = M.reindex(index=present_rows, columns=present_cols).round(3)

    diag = [M.loc[r, r] for r in M.index if r in M.columns]
    off = [M.loc[i, j] for i in M.index for j in M.columns if i != j and not pd.isna(M.loc[i, j])]
    held_mean = float(np.nanmean(diag)) if diag else float("nan")
    seen_mean = float(np.nanmean(off)) if off else float("nan")
    summary = {
        "held_out_mean": held_mean,
        "seen_mean": seen_mean,
        "recovery_gap": seen_mean - held_mean,
    }
    if do_print:
        print(f"\n=== {title} ===")
        print(M.to_string())
        print(f"\nheld-out mean : {held_mean:.3f}   (model never saw test_ref in training)")
        print(f"seen mean     : {seen_mean:.3f}   (in-distribution off-diagonal)")
        print(f"recovery gap  : {seen_mean - held_mean:+.3f}  (smaller = better generalisation)")
    return {"matrix": M, "summary": summary}


def _report_ems(
    df: pd.DataFrame, modes: Tuple[str, ...], title: str, do_print: bool,
) -> Dict[str, Any]:
    means = df.groupby("reference")["accuracy"].mean()
    present = [m for m in modes if m in means.index]
    means = means.reindex(present)
    M = means.round(3).to_frame("ems_diag")
    mean = float(means.mean())
    summary = {"mean": mean}
    if do_print:
        print(f"\n=== {title} ===")
        print(M.to_string())
        print(f"\nmean : {mean:.3f}")
    return {"matrix": M, "summary": summary}


def _report_bandpass(
    df: pd.DataFrame, title: str, do_print: bool,
) -> Dict[str, Any]:
    M = (df.groupby(["train_band", "test_band"])["accuracy"]
           .mean().unstack().round(3))
    summary = {
        "diagonal_mean": float(np.nanmean([M.loc[b, b] for b in M.index if b in M.columns])),
        "off_diag_mean": float(np.nanmean(
            [M.loc[i, j] for i in M.index for j in M.columns
             if i != j and not pd.isna(M.loc[i, j])]
        )),
    }
    if do_print:
        print(f"\n=== {title} ===")
        print(M.to_string())
        print(f"\ndiagonal mean : {summary['diagonal_mean']:.4f}")
        print(f"off-diag mean : {summary['off_diag_mean']:.4f}")
    return {"matrix": M, "summary": summary}


# ---------------------------------------------------------------------------
# Heatmap rendering
# ---------------------------------------------------------------------------

def _save_figure_for_kind(
    df: pd.DataFrame, kind: str, modes: Tuple[str, ...],
    title: str, out_path: str,
) -> None:
    """Render the appropriate visualisation for this experiment kind."""
    import matplotlib.pyplot as plt

    if kind == "mismatch":
        fig = plot_mismatch_matrix(
            df, out_path=out_path, title=title, reference_order=modes,
        )
        plt.close(fig)
        return

    if kind == "lofo":
        # Same heatmap as mismatch, but holdout_ref takes the train_ref slot.
        df_for_plot = df.rename(columns={"holdout_ref": "train_ref"})
        fig = plot_mismatch_matrix(
            df_for_plot, out_path=out_path, title=title, reference_order=modes,
        )
        plt.close(fig)
        return

    if kind == "jitter_full":
        means = (df.groupby("test_ref")["accuracy"].mean()
                   .reindex([m for m in modes if m in df["test_ref"].unique()]).dropna())
        fig, ax = plt.subplots(figsize=(7, 4), dpi=140)
        ax.bar(range(len(means)), means.values * 100, color="#4c78a8")
        ax.set_xticks(range(len(means)))
        ax.set_xticklabels(means.index, rotation=45, ha="right")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 100)
        ax.axhline(
            means.values.mean() * 100, color="black", linestyle="--", linewidth=1,
            label=f"mean = {means.values.mean()*100:.1f}%",
        )
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight", dpi=140)
        plt.close(fig)
        return

    if kind == "ems_diag":
        means = (df.groupby("reference")["accuracy"].mean()
                   .reindex([m for m in modes if m in df["reference"].unique()]).dropna())
        fig, ax = plt.subplots(figsize=(7, 4), dpi=140)
        ax.bar(range(len(means)), means.values * 100, color="#54a24b")
        ax.set_xticks(range(len(means)))
        ax.set_xticklabels(means.index, rotation=45, ha="right")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 100)
        ax.axhline(
            means.values.mean() * 100, color="black", linestyle="--", linewidth=1,
            label=f"mean = {means.values.mean()*100:.1f}%",
        )
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight", dpi=140)
        plt.close(fig)
        return

    if kind == "bandpass":
        M = (df.groupby(["train_band", "test_band"])["accuracy"]
               .mean().unstack())
        fig, ax = plt.subplots(figsize=(6, 4), dpi=140)
        im = ax.imshow(M.to_numpy() * 100, cmap="viridis", vmin=0, vmax=100, aspect="auto")
        mid = 50.0
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                val = M.iloc[i, j] * 100
                if pd.isna(val):
                    continue
                color = "black" if val > mid else "white"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", color=color, fontsize=10)
        ax.set_xticks(range(len(M.columns))); ax.set_xticklabels(M.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(M.index))); ax.set_yticklabels(M.index)
        ax.set_xlabel("Test bandpass (Hz)"); ax.set_ylabel("Train bandpass (Hz)")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label="Accuracy (%)")
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight", dpi=140)
        plt.close(fig)
        return

    raise AssertionError(f"unhandled kind: {kind}")
