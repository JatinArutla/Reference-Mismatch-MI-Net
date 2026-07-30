"""Turn a long-form results table into the numbers you read.

Every runner returns one row per (subject, seed, ...). These helpers pivot that
into matrices and print the summaries. Two kinds of number appear:

  pooled gap    mean over all rows, then diagonal minus off-diagonal. Good for
                reading structure in the matrix. Descriptive only.
  transfer gap  computed per subject on that subject's own seed-averaged
                matrix, then bootstrapped over subjects. Seeds are repeated
                runs of the same subject, not independent samples, so they are
                averaged within subject first. This is the paper number.

In a balanced design the two point estimates are algebraically identical: the
mean of per-subject gaps equals the pooled gap. The transfer gap earns its place
by carrying a confidence interval, not by differing. If the two ever disagree,
some subject is missing cells -- treat a divergence as a data alarm.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from refshift.references import FAMILIES, REFERENCE_MODES


# ---------------------------------------------------------------------------
# Matrices
# ---------------------------------------------------------------------------

def mismatch_matrix(df, *, metric="accuracy"):
    """Mean ``metric`` as a train_ref x test_ref table."""
    return df.groupby(["train_ref", "test_ref"])[metric].mean().unstack("test_ref")


def mismatch_std_matrix(df, *, metric="accuracy"):
    """Std of ``metric`` over (subject, seed) as a train_ref x test_ref table."""
    return df.groupby(["train_ref", "test_ref"])[metric].std().unstack("test_ref")


# ---------------------------------------------------------------------------
# Subject-level statistics (the paper number)
# ---------------------------------------------------------------------------

def _per_subject_gap(df, metric="accuracy"):
    """One matched-minus-mismatched gap per subject, seeds averaged first."""
    cells = (df.groupby(["subject", "train_ref", "test_ref"])[metric]
               .mean().reset_index())
    gaps = {}
    for subject, sub in cells.groupby("subject"):
        M = sub.pivot(index="train_ref", columns="test_ref", values=metric)
        refs = [r for r in M.index if r in M.columns]
        A = M.reindex(index=refs, columns=refs).to_numpy(dtype=float)
        off = ~np.eye(A.shape[0], dtype=bool)
        gaps[subject] = float(np.nanmean(np.diag(A)) - np.nanmean(A[off]))
    return pd.Series(gaps)


def transfer_gap_ci(df, *, metric="accuracy", n_boot=10000, ci=95, seed=0):
    """Subject-level transfer gap with a percentile bootstrap CI over subjects.

    Returns dict(mean, lo, hi, n_subjects) in percent. With ~9 subjects the
    interval is wide; that width is the honest uncertainty.
    """
    v = _per_subject_gap(df, metric).to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"mean": np.nan, "lo": np.nan, "hi": np.nan, "n_subjects": 0}
    rng = np.random.default_rng(seed)
    boot = rng.choice(v, size=(n_boot, v.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return {"mean": round(float(v.mean()) * 100, 2), "lo": round(float(lo) * 100, 2),
            "hi": round(float(hi) * 100, 2), "n_subjects": int(v.size)}


# ---------------------------------------------------------------------------
# Reporters
# ---------------------------------------------------------------------------

def _header(title):
    print("=" * 78)
    print(title)
    print("=" * 78)


def report_matrix(df, *, title, modes=REFERENCE_MODES):
    """Mismatch matrix, transfer summary, per-test-ref view, spread, paper gap."""
    present = [m for m in modes if m in df["train_ref"].unique()]
    M = mismatch_matrix(df).reindex(index=present, columns=present)
    A = M.to_numpy(dtype=float)
    off = ~np.eye(A.shape[0], dtype=bool)

    _header(title)
    print("Accuracy matrix (rows = train_ref, cols = test_ref), %:")
    print((M * 100).round(1).to_string())

    print("\n[A] Pooled summary (descriptive; not the paper number)")
    print(f"    diagonal mean = {np.nanmean(np.diag(A)) * 100:6.2f}%")
    print(f"    off-diag mean = {np.nanmean(A[off]) * 100:6.2f}%")
    print(f"    pooled gap    = {(np.nanmean(np.diag(A)) - np.nanmean(A[off])) * 100:6.2f}%")

    print("\n[B] Per-test-ref view (which references are hardest to transfer INTO)")
    rows = [{"test_ref": t,
             "matched_%": round(A[j, j] * 100, 1),
             "mismatched_%": round(np.nanmean(np.delete(A[:, j], j)) * 100, 1),
             "gap_%": round((A[j, j] - np.nanmean(np.delete(A[:, j], j))) * 100, 1)}
            for j, t in enumerate(present)]
    print(pd.DataFrame(rows).sort_values("gap_%", ascending=False).to_string(index=False))

    S = mismatch_std_matrix(df).reindex(index=present, columns=present)
    print("\n[C] Per-cell std over (subject, seed), %:")
    print((S * 100).round(1).to_string())

    # Mismatch is directional: A->B and B->A need not cost the same. A large
    # asymmetry says the operators are not simply "far apart" in one metric.
    D = np.abs(A - A.T)
    i, j = np.unravel_index(np.nanargmax(D), D.shape)
    print(f"\n[C2] Asymmetry: mean |M - M^T| off-diagonal = {np.nanmean(D[off]) * 100:.2f}pp; "
          f"largest pair {present[i]}->{present[j]} {A[i, j] * 100:.1f}% "
          f"vs {present[j]}->{present[i]} {A[j, i] * 100:.1f}% "
          f"({D[i, j] * 100:.1f}pp)")

    if {"subject", "train_ref", "test_ref"}.issubset(df.columns):
        r = transfer_gap_ci(df)
        print(f"\n[D] TRANSFER GAP (paper number) = {r['mean']:.2f}%  "
              f"[95% CI {r['lo']:.2f}, {r['hi']:.2f}]  n={r['n_subjects']} subjects, "
              "seeds averaged within subject")
    print()
    return M


def report_families(df, *, title, modes=REFERENCE_MODES, families=FAMILIES):
    """Within-family gaps, cross-family grid, and the operator-distance signal."""
    present = [m for m in modes if m in df["train_ref"].unique()]
    M = (df.groupby(["train_ref", "test_ref"])["accuracy"].mean()
           .unstack("test_ref").reindex(index=present, columns=present)) * 100

    groups = {n: [m for m in ms if m in present] for n, ms in families.items()}
    groups = {n: g for n, g in groups.items() if g}
    if "native" in present:                      # no-op baseline, its own group
        groups["native"] = ["native"]

    _header(title + "  -- family view")
    print("[A] Within-family transfer gap (diag - off), pp  [families with >=2 members]")
    rows = []
    for name, g in groups.items():
        if len(g) < 2:
            continue
        d = np.nanmean([M.loc[a, a] for a in g])
        o = np.nanmean([M.loc[a, b] for a in g for b in g if a != b])
        rows.append({"family": name, "members": ",".join(g), "diag_%": round(d, 1),
                     "off_%": round(o, 1), "gap_pp": round(d - o, 1)})
    print(pd.DataFrame(rows).to_string(index=False) if rows else "  (none)")

    print("\n[B] Cross-family mean transfer accuracy, %  (train-family -> test-family)")
    fams = list(groups)
    grid = pd.DataFrame(index=fams, columns=fams, dtype=float)
    for f1 in fams:
        for f2 in fams:
            grid.loc[f1, f2] = np.nanmean(
                [M.loc[a, b] for a in groups[f1] for b in groups[f2]])
    print(grid.round(1).to_string())
    print("  (rows = trained-on family, cols = tested-on family; "
          "low off-diagonal = collapse)")

    if "global" in groups and "spatial" in groups:
        within = grid.loc["global", "global"]
        across = (grid.loc["global", "spatial"] + grid.loc["spatial", "global"]) / 2
        print(f"\n[C] Operator-distance signal: within-global={within:.1f}%  "
              f"global<->spatial={across:.1f}%  drop={within - across:.1f}pp")
    print()
    return M


def report_jitter_full(df, *, title, modes=REFERENCE_MODES):
    """Per-test-reference accuracy of a jitter-trained model, plus its spread."""
    present = [m for m in modes if m in df["test_ref"].unique()]
    g = df.groupby("test_ref")["accuracy"]
    mean, std = g.mean().reindex(present), g.std().reindex(present)
    tbl = pd.DataFrame({"test_ref": present,
                        "acc_%": (mean.values * 100).round(1),
                        "std_%": (std.values * 100).round(1)})

    _header(title)
    print("Per-test-ref accuracy of the jitter-trained model, %:")
    print(tbl.sort_values("acc_%", ascending=False).to_string(index=False))
    a = mean.values
    print("\n[A] Invariance summary")
    print(f"    mean over test_refs = {np.nanmean(a) * 100:6.2f}%")
    print(f"    worst test_ref      = {np.nanmin(a) * 100:6.2f}%  "
          f"({present[int(np.nanargmin(a))]})")
    print(f"    spread (max - min)  = {(np.nanmax(a) - np.nanmin(a)) * 100:6.2f}%"
          "  (smaller = more invariant)")
    print()
    return tbl


def report_loro(df, *, title, modes=REFERENCE_MODES, full_jitter=None):
    """Leave-one-reference-out: the diagonal is the unseen-reference accuracy.

    Pass ``full_jitter`` (the run_mismatch_jitter table for condition='full') to
    get the holdout cost measured properly. Comparing a held-out reference against
    the same model's other references conflates "we never trained on it" with
    "it is intrinsically harder"; the full-jitter model differs only in that one
    reference, so it is the right baseline.
    """
    present = [m for m in modes if m in df["holdout_ref"].unique()]
    M = (df.groupby(["holdout_ref", "test_ref"])["accuracy"].mean()
           .unstack("test_ref").reindex(index=present, columns=present))
    A = M.to_numpy(dtype=float)

    _header(title)
    print("Accuracy matrix (rows = holdout_ref, cols = test_ref), %:")
    print((M * 100).round(1).to_string())
    diag, off = np.diag(A), A[~np.eye(A.shape[0], dtype=bool)]
    print("\n[A] Held-out-reference summary")
    print(f"    diagonal mean (unseen ref)  = {np.nanmean(diag) * 100:6.2f}%")
    print(f"    off-diag mean (in-mix refs) = {np.nanmean(off) * 100:6.2f}%")
    print(f"    recovery gap                = "
          f"{(np.nanmean(off) - np.nanmean(diag)) * 100:6.2f}%"
          "  (cost of holding a ref out)")

    print("\n[B] Per-held-out-reference view")
    base = None
    if full_jitter is not None:
        base = full_jitter.groupby("test_ref")["accuracy"].mean()
    rows = []
    for j, h in enumerate(present):
        in_mix = np.nanmean(np.delete(A[j, :], j))
        row = {"holdout_ref": h, "unseen_acc_%": round(A[j, j] * 100, 1),
               "in_mix_acc_%": round(in_mix * 100, 1),
               "naive_cost_%": round((in_mix - A[j, j]) * 100, 1)}
        if base is not None and h in base.index:
            row["true_cost_%"] = round((base[h] - A[j, j]) * 100, 1)
        rows.append(row)
    table = pd.DataFrame(rows)
    sort_key = "true_cost_%" if "true_cost_%" in table else "naive_cost_%"
    print(table.sort_values(sort_key, ascending=False).to_string(index=False))
    if base is None:
        print("  naive_cost_% mixes holdout cost with intrinsic difficulty; pass "
              "full_jitter= for true_cost_%.")

    # A reference that drags the mix down shows up as a HIGH in-mix mean when it
    # is the one held out. Silence here means every reference pulls its weight.
    if base is not None:
        overall = float(base.mean())
        worst = max(range(len(present)),
                    key=lambda j: np.nanmean(np.delete(A[j, :], j)))
        lift = (np.nanmean(np.delete(A[worst, :], worst)) - overall) * 100
        print(f"\n[C] Mix quality: holding out {present[worst]!r} raises the other "
              f"references by {lift:+.1f}pp vs the full mix ({overall * 100:.1f}%). "
              "A large positive value means that reference is a distractor, not an "
              "augmentation.")
    print()
    return M


def report_lofo(df, *, title):
    """Leave-one-family-out: the diagonal is the unseen family's transfer."""
    M = (df.groupby(["holdout_family", "test_family"])["accuracy"].mean()
           .unstack("test_family"))
    rows_order = [f for f in dict.fromkeys(df["holdout_family"]) if f in M.index]
    cols = rows_order + [f for f in M.columns if f not in rows_order]
    M = M.reindex(index=rows_order, columns=cols) * 100
    A = M.to_numpy(dtype=float)

    _header(title + "  -- family view")
    print("Accuracy matrix (rows = held-out family, cols = test family), %:")
    print(M.round(1).to_string())
    held = np.nanmean([A[i, i] for i in range(len(rows_order)) if i < A.shape[1]])
    seen = np.nanmean([A[i, j] for i in range(len(rows_order))
                       for j in range(A.shape[1]) if i != j])
    print("\n[A] Held-out-family summary")
    print(f"    held-out family mean = {held:6.2f}%  (whole family unseen)")
    print(f"    seen family mean     = {seen:6.2f}%  (in-distribution)")
    print(f"    family recovery gap  = {seen - held:6.2f}%"
          "  (smaller = better cross-family generalisation)")
    print()
    return M
