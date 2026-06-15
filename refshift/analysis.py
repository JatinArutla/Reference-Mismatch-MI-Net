"""Turn long-form result tables into the matrices and summaries you read.

These are the reporting helpers the notebook uses. They take the long-form
DataFrame a runner returns and produce a printed matrix plus a few summary
numbers. None of them touch the models or data; they are pure pandas/numpy on
the results table, so they are easy to read and to trust.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from refshift.references import FAMILIES, REFERENCE_MODES


def mismatch_matrix(df: pd.DataFrame, *, metric: str = "accuracy") -> pd.DataFrame:
    """Mean ``metric`` as a train_ref x test_ref table."""
    return df.groupby(["train_ref", "test_ref"])[metric].mean().unstack("test_ref")


def mismatch_std_matrix(df: pd.DataFrame, *, metric: str = "accuracy") -> pd.DataFrame:
    """Std of ``metric`` over (subject, seed) as a train_ref x test_ref table."""
    return df.groupby(["train_ref", "test_ref"])[metric].std().unstack("test_ref")


def report_matrix(df, *, title, modes=REFERENCE_MODES):
    """Print the accuracy mismatch matrix and a transfer-gap summary.

    Sections: [A] diagonal vs off-diagonal means and their gap; [B] per
    test-reference view of how hard each reference is to transfer into; [C]
    per-cell std over (subject, seed). Returns the accuracy matrix.
    """
    present = [m for m in modes if m in df["train_ref"].unique()]
    M = mismatch_matrix(df).reindex(index=present, columns=present)
    A = M.to_numpy(dtype=float)
    n = A.shape[0]
    off_mask = ~np.eye(n, dtype=bool)

    print("=" * 78)
    print(title)
    print("=" * 78)
    print("Accuracy matrix (rows = train_ref, cols = test_ref), %:")
    print((M * 100).round(1).to_string())

    diag = np.diag(A)
    off = A[off_mask]
    print("\n[A] Transfer summary")
    print(f"    diagonal mean   = {np.nanmean(diag) * 100:6.2f}%")
    print(f"    off-diag mean   = {np.nanmean(off) * 100:6.2f}%")
    print(f"    transfer gap    = {(np.nanmean(diag) - np.nanmean(off)) * 100:6.2f}%"
          "  (bigger = worse mismatch)")

    print("\n[B] Per-test-ref view (which references are hardest to transfer INTO)")
    rows = []
    for j, t in enumerate(present):
        col = A[:, j]
        matched = col[j]
        mismatched = np.nanmean(np.delete(col, j))
        rows.append({"test_ref": t, "matched_%": round(matched * 100, 1),
                     "mismatched_%": round(mismatched * 100, 1),
                     "gap_%": round((matched - mismatched) * 100, 1)})
    print(pd.DataFrame(rows).sort_values("gap_%", ascending=False).to_string(index=False))

    S = mismatch_std_matrix(df).reindex(index=present, columns=present)
    print("\n[C] Per-cell std over (subject, seed), %:")
    print((S * 100).round(1).to_string())
    print()
    return M


def report_families(df, *, title, modes=REFERENCE_MODES, families=FAMILIES):
    """Within-family transfer gaps and a cross-family transfer grid.

    [A] for each family with >=2 members, the diagonal-vs-off gap inside the
    family. [B] a family x family grid of mean transfer accuracy. Returns the
    full accuracy matrix (percent).
    """
    present = [m for m in modes if m in df["train_ref"].unique()]
    M = (df.groupby(["train_ref", "test_ref"])["accuracy"].mean()
           .unstack("test_ref").reindex(index=present, columns=present)) * 100

    groups = {}
    for name, members in families.items():
        g = [m for m in members if m in present]
        if g:
            groups[name] = g
    if "native" in present:
        groups["native"] = ["native"]

    print("=" * 78)
    print(title + "  -- family view")
    print("=" * 78)
    print("[A] Within-family transfer gap (diag - off), pp  [families with >=2 members]")
    wrows = []
    for name, g in groups.items():
        if len(g) < 2:
            continue
        d = np.nanmean([M.loc[a, a] for a in g])
        o = np.nanmean([M.loc[a, b] for a in g for b in g if a != b])
        wrows.append({"family": name, "members": ",".join(g),
                      "diag_%": round(d, 1), "off_%": round(o, 1),
                      "gap_pp": round(d - o, 1)})
    print(pd.DataFrame(wrows).to_string(index=False) if wrows else "  (none)")

    print("\n[B] Cross-family mean transfer accuracy, %  (train-family -> test-family)")
    fams = list(groups.keys())
    grid = pd.DataFrame(index=fams, columns=fams, dtype=float)
    for f1 in fams:
        for f2 in fams:
            grid.loc[f1, f2] = np.nanmean(
                [M.loc[a, b] for a in groups[f1] for b in groups[f2]]
            )
    print(grid.round(1).to_string())
    print("  (rows = trained-on family, cols = tested-on family; "
          "low off-diagonal = collapse)")
    print()
    return M


def report_jitter_full(df, *, title, modes=REFERENCE_MODES):
    """Per-test-reference accuracy of a full-jitter model, with an invariance
    summary (mean, worst reference, spread). Returns the per-ref table."""
    present = [m for m in modes if m in df["test_ref"].unique()]
    g = df.groupby("test_ref")["accuracy"]
    mean = g.mean().reindex(present)
    std = g.std().reindex(present)
    tbl = pd.DataFrame({
        "test_ref": present,
        "acc_%": (mean.values * 100).round(1),
        "std_%": (std.values * 100).round(1),
    }).sort_values("acc_%", ascending=False)

    print("=" * 78)
    print(title)
    print("=" * 78)
    print("Per-test-ref accuracy of the jitter-trained model, %:")
    print(tbl.to_string(index=False))
    a = mean.values
    print("\n[A] Invariance summary")
    print(f"    mean over test_refs = {np.nanmean(a) * 100:6.2f}%")
    print(f"    worst test_ref      = {np.nanmin(a) * 100:6.2f}%"
          f"  ({present[int(np.nanargmin(a))]})")
    print(f"    spread (max - min)  = {(np.nanmax(a) - np.nanmin(a)) * 100:6.2f}%"
          "  (smaller = more invariant)")
    print()
    return tbl


def report_loro(df, *, title, modes=REFERENCE_MODES):
    """Leave-one-reference-out: holdout_ref x test_ref matrix; diagonal is the
    unseen-reference accuracy. Returns the matrix (percent)."""
    present = [m for m in modes if m in df["holdout_ref"].unique()]
    M = (df.groupby(["holdout_ref", "test_ref"])["accuracy"].mean()
           .unstack("test_ref").reindex(index=present, columns=present))
    A = M.to_numpy(dtype=float)
    n = A.shape[0]

    print("=" * 78)
    print(title)
    print("=" * 78)
    print("Accuracy matrix (rows = holdout_ref, cols = test_ref), %:")
    print((M * 100).round(1).to_string())
    diag = np.diag(A)
    off = A[~np.eye(n, dtype=bool)]
    print("\n[A] Held-out-reference summary")
    print(f"    diagonal mean (unseen ref)  = {np.nanmean(diag) * 100:6.2f}%")
    print(f"    off-diag mean (in-mix refs) = {np.nanmean(off) * 100:6.2f}%")
    print(f"    recovery gap                = "
          f"{(np.nanmean(off) - np.nanmean(diag)) * 100:6.2f}%"
          "  (cost of holding a ref out)")
    print()
    return M


def report_lofo(df, *, title):
    """Leave-one-family-out: holdout_family x test_family matrix.

    Diagonal cells are the held-out (unseen) family's transfer; off-diagonal is
    in-distribution. Returns the family x family matrix (percent).
    """
    M = (df.groupby(["holdout_family", "test_family"])["accuracy"].mean()
           .unstack("test_family"))
    order = list(dict.fromkeys(df["holdout_family"].tolist()))
    rows = [f for f in order if f in M.index]
    cols = rows + [f for f in M.columns if f not in rows]
    M = (M.reindex(index=rows, columns=cols)) * 100
    A = M.to_numpy(dtype=float)

    print("=" * 78)
    print(title + "  -- family view")
    print("=" * 78)
    print("Accuracy matrix (rows = held-out family, cols = test family), %:")
    print(M.round(1).to_string())
    diag = [A[i, i] for i in range(len(rows)) if i < A.shape[1] and not np.isnan(A[i, i])]
    off = [A[i, j] for i in range(len(rows)) for j in range(A.shape[1])
           if i != j and not np.isnan(A[i, j])]
    held = float(np.nanmean(diag)) if diag else float("nan")
    seen = float(np.nanmean(off)) if off else float("nan")
    print("\n[A] Held-out-family summary")
    print(f"    held-out family mean = {held:6.2f}%  (whole family unseen)")
    print(f"    seen family mean     = {seen:6.2f}%  (in-distribution)")
    print(f"    family recovery gap  = {seen - held:6.2f}%"
          "  (smaller = better cross-family generalisation)")
    print()
    return M
