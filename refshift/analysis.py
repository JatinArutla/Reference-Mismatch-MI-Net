"""Post-hoc analyses on mismatch-matrix results.

Four pure-numpy/scipy functions on the long-form DataFrame from run_mismatch
or the aggregate from mismatch_matrix:

    mismatch_std_matrix              per-cell std across (subject, seed)
    cluster_references               agglomerative on D = diag - sym(M)
    operator_distance_correlation    Frobenius operator distance vs transfer gap
    paired_wilcoxon_per_test_ref     paired Wilcoxon per test_ref + Holm

Plus helpers baseline_diagonal_view, baseline_col_off_diag_view to extract
comparable subsets from a baseline run_mismatch DataFrame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from refshift.reference import (
    REFERENCE_MODES,
    DatasetGraph,
    apply_reference,
    build_graph,
)


# ---------------------------------------------------------------------------
# 1. Standard-deviation matrix
# ---------------------------------------------------------------------------

def mismatch_std_matrix(
    df: pd.DataFrame,
    *,
    metric: str = "accuracy",
    reference_order: Tuple[str, ...] = REFERENCE_MODES,
) -> pd.DataFrame:
    """Per-cell std over (subject, seed). Counterpart to mismatch_matrix(..., 'mean')."""
    agg = df.groupby(["train_ref", "test_ref"])[metric].std()
    present_train = [m for m in reference_order if m in agg.index.get_level_values("train_ref").unique()]
    present_test = [m for m in reference_order if m in agg.index.get_level_values("test_ref").unique()]
    return agg.unstack("test_ref").reindex(index=present_train, columns=present_test)


# ---------------------------------------------------------------------------
# 2. Hierarchical clustering
# ---------------------------------------------------------------------------

@dataclass
class ClusterResult:
    references: List[str]
    distance_matrix: np.ndarray
    linkage: np.ndarray
    clusters: Dict[int, List[List[str]]]
    diag_mean: float


def cluster_references(
    mean_matrix: pd.DataFrame,
    *,
    method: str = "average",
    cluster_sizes: Tuple[int, ...] = (2, 3, 4),
) -> ClusterResult:
    """Agglomerative cluster references on D_ij = diag_mean - 0.5*(M_ij + M_ji).

    method='average' (UPGMA) because the distance is behavioural, not Euclidean
    (so 'ward' isn't strictly applicable). cluster_sizes lists ks for which
    fcluster assignments are reported.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    M = mean_matrix.to_numpy().astype(np.float64)
    refs = list(mean_matrix.index)
    if list(mean_matrix.columns) != refs:
        raise ValueError(
            f"cluster_references expects square matrix with matching row/col order; "
            f"got rows={refs}, cols={list(mean_matrix.columns)}"
        )

    Msym = 0.5 * (M + M.T)
    diag_mean = float(np.diag(M).mean())
    D = diag_mean - Msym
    np.fill_diagonal(D, 0.0)
    D = np.maximum(D, 0.0)  # clip numerical noise

    Z = linkage(squareform(D, checks=False), method=method)

    clusters: Dict[int, List[List[str]]] = {}
    for k in cluster_sizes:
        labels = fcluster(Z, t=k, criterion="maxclust")
        groups: Dict[int, List[str]] = {}
        for r, lab in zip(refs, labels):
            groups.setdefault(int(lab), []).append(r)
        clusters[k] = [groups[i] for i in sorted(groups)]

    return ClusterResult(
        references=refs, distance_matrix=D, linkage=Z,
        clusters=clusters, diag_mean=diag_mean,
    )


def plot_dendrogram(
    result: ClusterResult,
    out_path: Optional[str] = None,
    *,
    title: str = "Reference-operator clustering",
    figsize: Tuple[float, float] = (8, 4),
    dpi: int = 140,
):
    """Dendrogram from cluster_references output. Returns the figure."""
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    dendrogram(
        result.linkage, labels=result.references,
        leaf_rotation=45, leaf_font_size=10,
        color_threshold=0.0, above_threshold_color="black",
        ax=ax,
    )
    ax.set_ylabel("Distance  (diag mean - symmetric transfer)")
    ax.set_title(title)
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, bbox_inches="tight", dpi=dpi)
    return fig


# ---------------------------------------------------------------------------
# 3. Operator-distance vs transfer-gap correlation
# ---------------------------------------------------------------------------

def _estimate_linear_operator(
    mode: str,
    graph: DatasetGraph,
    *,
    n_times: int = 2000,
    seed: int = 0,
    n_probes: int = 1,
) -> np.ndarray:
    """Best linear approximation of the operator: A = Y @ pinv(X) on Gaussian X.

    Exact for genuinely linear ops (native, CAR, REST, kNN-Laplacian, cz_ref).
    For median (non-linear), returns the linear tangent — empirically close
    to CAR's I - J/C since median of zero-mean Gaussian is approximately zero.
    n_probes>1 averages independent probes; matters mostly for median.
    """
    C = len(graph.ch_names)
    rng = np.random.default_rng(seed)
    A_acc = np.zeros((C, C), dtype=np.float64)
    for _ in range(int(n_probes)):
        X = rng.standard_normal((1, C, n_times)).astype(np.float32)
        Y = apply_reference(X, mode, graph=graph)
        A_acc += Y[0] @ np.linalg.pinv(X[0])
    return (A_acc / float(n_probes)).astype(np.float64)


@dataclass
class OperatorDistanceResult:
    """Asymptotic stats unreliable at n=15 pairs; report perm_p and ci95 in papers."""
    references: List[str]
    distances_frobenius: np.ndarray
    transfer_gaps: np.ndarray
    spearman_rho: float
    spearman_p: float
    pearson_r: float
    pearson_p: float
    perm_p_spearman: float
    perm_p_pearson: float
    ci95_spearman: Tuple[float, float]
    ci95_pearson: Tuple[float, float]
    pair_table: pd.DataFrame


def operator_distance_correlation(
    mean_matrix: pd.DataFrame,
    ch_names: List[str],
    *,
    k_laplacian: int = 4,
    montage: str = "standard_1005",
    n_probe_times: int = 2000,
    n_probes: int = 8,
    seed: int = 0,
    n_permutations: int = 10_000,
    n_bootstrap: int = 5_000,
) -> OperatorDistanceResult:
    """Test whether Frobenius operator distance predicts transfer gap.

    Estimate each operator's linear matrix on a random Gaussian probe; compute
    pairwise Frobenius distances; correlate with gap_ij = diag_mean - 0.5*(M_ij + M_ji)
    on the upper triangle. Bootstrap CIs over pairs and permutation p-value
    over operator-label shuffles, because n=15 pairs (or 10 if cz_ref dropped)
    makes asymptotic Spearman/Pearson p-values unreliable.

    Not a Ben-David H-divergence bound: Frobenius distance is data-free; the
    correlation with empirical transfer gap is a structural finding, not tight.
    """
    from scipy.stats import pearsonr, spearmanr

    refs = list(mean_matrix.index)
    if list(mean_matrix.columns) != refs:
        raise ValueError("mean_matrix must be square with matching row/col order")

    need_rest = "rest" in refs
    graph = build_graph(ch_names, k=k_laplacian, montage=montage, include_rest=need_rest)

    ops: Dict[str, np.ndarray] = {}
    for r in refs:
        ops[r] = _estimate_linear_operator(
            r, graph, n_times=n_probe_times, seed=seed, n_probes=n_probes,
        )

    n = len(refs)
    D_op = np.zeros((n, n))
    for i, a in enumerate(refs):
        for j, b in enumerate(refs):
            D_op[i, j] = np.linalg.norm(ops[a] - ops[b], ord="fro")

    M = mean_matrix.to_numpy().astype(np.float64)
    Msym = 0.5 * (M + M.T)
    diag_mean = float(np.diag(M).mean())
    gap = diag_mean - Msym
    np.fill_diagonal(gap, 0.0)

    iu = np.triu_indices(n, k=1)
    dist_flat = D_op[iu]
    gap_flat = gap[iu]

    r_s, p_s = spearmanr(dist_flat, gap_flat)
    r_p, p_p = pearsonr(dist_flat, gap_flat)

    rng = np.random.default_rng(seed)
    n_pairs = len(dist_flat)
    boot_rho = np.empty(n_bootstrap, dtype=np.float64)
    boot_pear = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n_pairs, size=n_pairs)
        d_b, g_b = dist_flat[idx], gap_flat[idx]
        if np.std(d_b) == 0 or np.std(g_b) == 0:
            boot_rho[b] = np.nan
            boot_pear[b] = np.nan
            continue
        boot_rho[b], _ = spearmanr(d_b, g_b)
        boot_pear[b], _ = pearsonr(d_b, g_b)
    valid_rho = boot_rho[~np.isnan(boot_rho)]
    valid_pear = boot_pear[~np.isnan(boot_pear)]
    ci_rho = (
        (float(np.percentile(valid_rho, 2.5)), float(np.percentile(valid_rho, 97.5)))
        if len(valid_rho) else (float("nan"), float("nan"))
    )
    ci_pear = (
        (float(np.percentile(valid_pear, 2.5)), float(np.percentile(valid_pear, 97.5)))
        if len(valid_pear) else (float("nan"), float("nan"))
    )

    # Permutation: shuffle operator labels of the symmetric gap matrix.
    perm_count_s = 0
    perm_count_p = 0
    obs_abs_s = abs(r_s)
    obs_abs_p = abs(r_p)
    for _ in range(n_permutations):
        perm = rng.permutation(n)
        gap_perm = gap[np.ix_(perm, perm)]
        gp = gap_perm[iu]
        if np.std(gp) == 0:
            continue
        r_s_perm, _ = spearmanr(dist_flat, gp)
        r_p_perm, _ = pearsonr(dist_flat, gp)
        if abs(r_s_perm) >= obs_abs_s:
            perm_count_s += 1
        if abs(r_p_perm) >= obs_abs_p:
            perm_count_p += 1
    # Phipson-Smyth +1/+1 small-sample correction
    perm_p_s = (perm_count_s + 1) / (n_permutations + 1)
    perm_p_p = (perm_count_p + 1) / (n_permutations + 1)

    rows = []
    for i, j in zip(*iu):
        rows.append({
            "ref_i": refs[i], "ref_j": refs[j],
            "distance_frobenius": float(D_op[i, j]),
            "transfer_gap": float(gap[i, j]),
        })
    pair_table = pd.DataFrame(rows)

    return OperatorDistanceResult(
        references=refs,
        distances_frobenius=D_op,
        transfer_gaps=gap,
        spearman_rho=float(r_s), spearman_p=float(p_s),
        pearson_r=float(r_p), pearson_p=float(p_p),
        perm_p_spearman=float(perm_p_s), perm_p_pearson=float(perm_p_p),
        ci95_spearman=ci_rho, ci95_pearson=ci_pear,
        pair_table=pair_table,
    )


def plot_operator_distance_scatter(
    result: OperatorDistanceResult,
    out_path: Optional[str] = None,
    *,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (6, 5),
    dpi: int = 140,
    annotate: bool = True,
):
    """Scatter Frobenius distance vs transfer gap; returns the figure."""
    import matplotlib.pyplot as plt

    df = result.pair_table
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.scatter(df["distance_frobenius"], df["transfer_gap"], s=36, alpha=0.8)
    ax.set_xlabel(r"Operator Frobenius distance  $\|A_i - A_j\|_F$")
    ax.set_ylabel("Transfer gap  (diag - symmetric transfer)")
    if title is None:
        title = (
            f"Operator distance vs transfer gap\n"
            f"Spearman rho = {result.spearman_rho:.3f}  (p = {result.spearman_p:.1e})"
        )
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if annotate:
        for _, row in df.iterrows():
            ax.annotate(
                f"{row['ref_i']}-{row['ref_j']}",
                xy=(row["distance_frobenius"], row["transfer_gap"]),
                xytext=(3, 3), textcoords="offset points",
                fontsize=7, alpha=0.7,
            )

    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, bbox_inches="tight", dpi=dpi)
    return fig


# ---------------------------------------------------------------------------
# 4. Paired Wilcoxon for jitter / LOFO experiments
# ---------------------------------------------------------------------------

def baseline_diagonal_view(baseline_df: pd.DataFrame) -> pd.DataFrame:
    """Diagonal cells of a run_mismatch DataFrame: train_ref == test_ref.

    Returns columns subject, seed, test_ref, accuracy. Compare a jitter
    DataFrame against this for the "is jitter different from clean training"
    test.
    """
    required = {"subject", "seed", "train_ref", "test_ref", "accuracy"}
    missing = required - set(baseline_df.columns)
    if missing:
        raise ValueError(f"baseline_df missing columns: {sorted(missing)}")
    diag = baseline_df[baseline_df["train_ref"] == baseline_df["test_ref"]]
    return diag[["subject", "seed", "test_ref", "accuracy"]].reset_index(drop=True)


def baseline_col_off_diag_view(baseline_df: pd.DataFrame) -> pd.DataFrame:
    """Per (subject, seed, test_ref): mean accuracy across all train_ref != test_ref.

    Compare a LOFO DataFrame against this: both sides are "model never saw
    test_ref at training" — but the baseline saw exactly one alternative
    reference, while LOFO saw 5. The Wilcoxon then tests whether
    multi-reference training helps unseen-reference transfer.
    """
    required = {"subject", "seed", "train_ref", "test_ref", "accuracy"}
    missing = required - set(baseline_df.columns)
    if missing:
        raise ValueError(f"baseline_df missing columns: {sorted(missing)}")
    off = baseline_df[baseline_df["train_ref"] != baseline_df["test_ref"]]
    return (
        off.groupby(["subject", "seed", "test_ref"], as_index=False)["accuracy"]
           .mean()
    )


def _holm_bonferroni(p_values: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni step-down. Returns adjusted p-values."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if m == 0:
        return p
    order = np.argsort(p)
    p_sorted = p[order]
    multipliers = np.arange(m, 0, -1)
    adjusted_sorted = np.minimum(1.0, p_sorted * multipliers)
    adjusted_sorted = np.maximum.accumulate(adjusted_sorted)
    out = np.empty_like(adjusted_sorted)
    out[order] = adjusted_sorted
    return out


def paired_wilcoxon_per_test_ref(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    *,
    label_a: str = "A",
    label_b: str = "B",
    alternative: str = "two-sided",
    correction: str = "holm",
) -> pd.DataFrame:
    """Per-test-ref paired Wilcoxon of accuracy_a - accuracy_b, Holm-corrected.

    Both DataFrames need columns subject, seed, test_ref, accuracy. Inner-joined
    on (subject, seed, test_ref); per test_ref the signed-rank test runs on the
    paired differences. A 'pooled' row is appended (uncorrected — different
    question: 'is there an overall effect' vs 'which test_refs differ').
    """
    from scipy.stats import wilcoxon

    required = {"subject", "seed", "test_ref", "accuracy"}
    for name, df in (("df_a", df_a), ("df_b", df_b)):
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{name} missing columns: {sorted(missing)}")
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError(f"Unknown alternative: {alternative!r}")
    if correction not in (None, "holm"):
        raise ValueError(f"Unknown correction: {correction!r}")

    merged = pd.merge(
        df_a[["subject", "seed", "test_ref", "accuracy"]].rename(columns={"accuracy": "acc_a"}),
        df_b[["subject", "seed", "test_ref", "accuracy"]].rename(columns={"accuracy": "acc_b"}),
        on=["subject", "seed", "test_ref"], how="inner",
    )
    if merged.empty:
        raise ValueError(
            "After joining on (subject, seed, test_ref), no paired observations remain. "
            "Check df_a and df_b cover the same subjects/seeds."
        )
    merged["delta"] = merged["acc_a"] - merged["acc_b"]

    rows = []
    for ref in sorted(merged["test_ref"].unique()):
        sub = merged[merged["test_ref"] == ref]
        n = len(sub)
        if n == 0:
            continue
        if np.allclose(sub["delta"].to_numpy(), 0.0):
            stat, p = 0.0, 1.0
        else:
            res = wilcoxon(
                sub["acc_a"].to_numpy(), sub["acc_b"].to_numpy(),
                alternative=alternative, zero_method="wilcox",
            )
            stat, p = float(res.statistic), float(res.pvalue)
        rows.append({
            "test_ref": ref,
            "n_pairs": int(n),
            f"mean_{label_a}": float(sub["acc_a"].mean()),
            f"mean_{label_b}": float(sub["acc_b"].mean()),
            "median_delta": float(sub["delta"].median()),
            "mean_delta": float(sub["delta"].mean()),
            "wilcoxon_stat": stat,
            "p_value": p,
        })

    out = pd.DataFrame(rows)
    if correction == "holm" and len(out) > 0:
        out["p_adjusted"] = _holm_bonferroni(out["p_value"].to_numpy())
    else:
        out["p_adjusted"] = out["p_value"]

    if np.allclose(merged["delta"].to_numpy(), 0.0):
        pooled_stat, pooled_p = 0.0, 1.0
    else:
        res = wilcoxon(
            merged["acc_a"].to_numpy(), merged["acc_b"].to_numpy(),
            alternative=alternative, zero_method="wilcox",
        )
        pooled_stat, pooled_p = float(res.statistic), float(res.pvalue)
    pooled_row = {
        "test_ref": "pooled",
        "n_pairs": int(len(merged)),
        f"mean_{label_a}": float(merged["acc_a"].mean()),
        f"mean_{label_b}": float(merged["acc_b"].mean()),
        "median_delta": float(merged["delta"].median()),
        "mean_delta": float(merged["delta"].mean()),
        "wilcoxon_stat": pooled_stat,
        "p_value": pooled_p,
        "p_adjusted": pooled_p,
    }
    return pd.concat([out, pd.DataFrame([pooled_row])], ignore_index=True)
