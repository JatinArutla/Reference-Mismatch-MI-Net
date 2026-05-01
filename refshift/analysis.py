"""Post-hoc analyses on mismatch-matrix results.

Four analyses are exposed, each operating on the long-form DataFrame
returned by ``run_mismatch`` or the aggregate matrix returned by
``mismatch_matrix``:

1. ``mismatch_std_matrix`` — per-cell standard deviation across
   (subject, seed), so the reader can judge whether the gap in the mean
   matrix is consistent or driven by a few subjects.

2. ``cluster_references`` — hierarchical agglomerative clustering on a
   symmetric distance matrix ``D_ij = diag_mean - 0.5*(M_ij + M_ji)``
   where ``M`` is the mean transfer matrix. Returns the linkage, the
   cluster assignments at several thresholds, and plotting helpers for
   a dendrogram.

3. ``operator_distance_correlation`` — empirically estimates each
   reference operator as a linear map on the channel set via a random
   probe, computes pairwise Frobenius distances between operator
   matrices, and correlates them with the observed transfer gap
   ``gap_ij = diag_mean - 0.5*(M_ij + M_ji)``. Spearman and Pearson
   correlations are returned with p-values. This is the Ben-David-style
   framing: transfer loss is predicted by how far apart the operators
   are in matrix-valued space.

4. ``paired_wilcoxon_per_test_ref`` — paired Wilcoxon signed-rank tests
   across (subject, seed) pairs, per test reference, with Holm-Bonferroni
   multiple-comparison correction across the 6 references. Companion
   helpers ``baseline_diagonal_view`` and ``baseline_col_off_diag_view``
   extract comparable subsets from a baseline ``run_mismatch`` DataFrame
   so jitter / LOFO results can be tested against single-reference
   training results without manual indexing.

All four are pure numpy/scipy; no torch, no MOABB, and no network.
They run on the CSVs produced by ``run_mismatch`` in under a second.
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
    """Per-cell standard deviation over (subject, seed) for a mismatch run.

    This is the counterpart to ``mismatch_matrix(df, aggregate='mean')`` —
    it answers "how variable is each cell?". A cell with a low mean and
    low std is a robust finding; low mean + high std means a few subjects
    drive the effect.

    Parameters
    ----------
    df : DataFrame with columns train_ref, test_ref, subject, seed, <metric>.
    metric : column name, default "accuracy".
    reference_order : row/column order; missing modes are dropped.

    Returns
    -------
    DataFrame indexed by train_ref, columns test_ref, values = std of <metric>.
    """
    agg = df.groupby(["train_ref", "test_ref"])[metric].std()
    present_train = [m for m in reference_order if m in agg.index.get_level_values("train_ref").unique()]
    present_test = [m for m in reference_order if m in agg.index.get_level_values("test_ref").unique()]
    return agg.unstack("test_ref").reindex(index=present_train, columns=present_test)


# ---------------------------------------------------------------------------
# 2. Hierarchical clustering
# ---------------------------------------------------------------------------

@dataclass
class ClusterResult:
    """Output of ``cluster_references``.

    Attributes
    ----------
    references : list of str
        Reference names in the order they index ``distance_matrix``.
    distance_matrix : ndarray (n_refs, n_refs)
        Symmetric distance: ``diag_mean - 0.5 * (M_ij + M_ji)``. Clipped
        at 0 so numerical noise doesn't produce negative distances.
    linkage : ndarray
        scipy linkage output (pass directly to ``scipy.cluster.hierarchy.dendrogram``).
    clusters : dict {k: list of list of str}
        Cluster assignments for several values of k (number of clusters).
    diag_mean : float
        The diagonal mean used as the "perfect transfer" reference.
    """
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
    """Cluster reference operators by their symmetric transfer behaviour.

    Distance is defined as ``D_ij = diag_mean - 0.5 * (M_ij + M_ji)``:
    the larger the symmetric transfer drop relative to within-reference
    accuracy, the farther apart the two operators. This matches the
    paper's question: "which operators behave similarly under mismatch?"

    Parameters
    ----------
    mean_matrix : DataFrame
        train_ref x test_ref mean accuracy, output of ``mismatch_matrix``.
    method : str
        scipy linkage method. Default 'average' (UPGMA); reasonable for
        behavioural distances. 'ward' is an option but requires
        Euclidean distances which we don't cleanly have.
    cluster_sizes : tuple of int
        Values of k for which to report cluster assignments.
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    M = mean_matrix.to_numpy().astype(np.float64)
    refs = list(mean_matrix.index)
    if list(mean_matrix.columns) != refs:
        raise ValueError(
            "cluster_references expects a square matrix with matching row/col order; "
            f"got rows={refs}, cols={list(mean_matrix.columns)}"
        )

    Msym = 0.5 * (M + M.T)
    diag_mean = float(np.diag(M).mean())
    D = diag_mean - Msym
    np.fill_diagonal(D, 0.0)
    D = np.maximum(D, 0.0)

    condensed = squareform(D, checks=False)
    Z = linkage(condensed, method=method)

    clusters: Dict[int, List[List[str]]] = {}
    for k in cluster_sizes:
        labels = fcluster(Z, t=k, criterion="maxclust")
        groups: Dict[int, List[str]] = {}
        for r, lab in zip(refs, labels):
            groups.setdefault(int(lab), []).append(r)
        # order clusters by first-member position for stable output
        clusters[k] = [groups[i] for i in sorted(groups)]

    return ClusterResult(
        references=refs,
        distance_matrix=D,
        linkage=Z,
        clusters=clusters,
        diag_mean=diag_mean,
    )


def plot_dendrogram(
    result: ClusterResult,
    out_path: Optional[str] = None,
    *,
    title: str = "Reference-operator clustering",
    figsize: Tuple[float, float] = (8, 4),
    dpi: int = 140,
):
    """Draw a dendrogram from ``cluster_references`` output.

    Returns the matplotlib Figure.
    """
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    dendrogram(
        result.linkage,
        labels=result.references,
        leaf_rotation=45,
        leaf_font_size=10,
        color_threshold=0.0,  # all black; colour is uninformative here
        above_threshold_color="black",
        ax=ax,
    )
    ax.set_ylabel("Distance  (diag mean − symmetric transfer)")
    ax.set_title(title)
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, bbox_inches="tight", dpi=dpi)
    return fig


# ---------------------------------------------------------------------------
# 3. Operator distance ↔ transfer correlation
# ---------------------------------------------------------------------------

def _estimate_linear_operator(
    mode: str,
    graph: DatasetGraph,
    *,
    n_times: int = 2000,
    seed: int = 0,
    n_probes: int = 1,
) -> np.ndarray:
    """Empirically estimate the linear operator matrix A for a reference op.

    For each op we compute Y = op(X) on a random Gaussian probe X, then
    solve A = Y @ pinv(X). This gives the best linear approximation in
    the least-squares sense. For genuinely linear ops (native, CAR, REST,
    kNN-Laplacian, cz_ref) this recovers the operator exactly. For
    median (non-linear), it returns the linear tangent, which equals CAR
    in expectation — the honest linearization. With ``n_probes > 1`` the
    estimate is averaged across independent Gaussian probes, which
    reduces the variance of the median linearization without changing
    the result for the linear operators.
    """
    C = len(graph.ch_names)
    rng = np.random.default_rng(seed)
    A_acc = np.zeros((C, C), dtype=np.float64)
    for p in range(int(n_probes)):
        X = rng.standard_normal((1, C, n_times)).astype(np.float32)
        Y = apply_reference(X, mode, graph=graph)
        A_acc += Y[0] @ np.linalg.pinv(X[0])
    return (A_acc / float(n_probes)).astype(np.float64)


@dataclass
class OperatorDistanceResult:
    """Output of ``operator_distance_correlation``.

    spearman_rho, pearson_r are point estimates on the upper triangle.
    spearman_p, pearson_p are asymptotic p-values; with n=15 pairs (6
    operators) these are unreliable at small-sample, so we additionally
    compute a permutation p-value (perm_p_spearman, perm_p_pearson) by
    shuffling the operator labels of the gap matrix and recomputing the
    correlation many times. ci95_spearman / ci95_pearson are
    bootstrap-resampled 95% confidence intervals on the correlations
    over pairs.
    """
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
    """Test whether Frobenius distance between reference operators predicts transfer gap.

    Procedure:
      1. For each reference in ``mean_matrix``, estimate its linear
         operator matrix A (C x C) on the given channel set, averaged
         over ``n_probes`` independent Gaussian probes (relevant only
         for the median operator's linear tangent estimate).
      2. Compute pairwise Frobenius distances between operators.
      3. Compute transfer gaps from the mean matrix:
         ``gap_ij = diag_mean - 0.5*(M_ij + M_ji)``.
      4. Correlate the upper triangle (n=15 pairs for 6 references;
         n=10 if the caller drops cz_ref, e.g. on Schirrmeister2017).
      5. Bootstrap CIs over the pairs (resampling pairs with replacement).
      6. Permutation test by shuffling operator labels of the gap matrix
         while keeping the distance matrix fixed.

    We do not interpret this as a Ben-David H-divergence bound. Frobenius
    distance is a data-free quantity; its empirical correlation with
    transfer gap is an interesting structural finding, not a tight
    theoretical bound. The paper should describe it as "operator-matrix
    Frobenius distance, a data-free quantity, predicts the empirical
    transfer gap with Spearman ρ = X (95% CI [a, b], permutation p < c)".

    Parameters
    ----------
    mean_matrix : DataFrame
        train_ref x test_ref, from ``mismatch_matrix``.
    ch_names : list of str
        EEG channel names for the dataset. Needed to build neighbour
        graph and REST matrix at the correct dimensionality.
    k_laplacian, montage : passed through to ``build_graph``.
    n_probe_times : int
        Length of the Gaussian probe used to recover each operator
        matrix. 2000 is plenty for linear operators; helpful for
        averaging the median linearization.
    n_probes : int
        Number of independent probes to average over. >1 helps for the
        non-linear median operator; ignored otherwise. Default 8.
    seed : int
        RNG seed for probes, bootstrap, and permutation. Reproducible.
    n_permutations : int
        Permutations for the label-shuffle null. Default 10000.
    n_bootstrap : int
        Bootstrap resamples for the pair-wise CI. Default 5000.

    Returns
    -------
    OperatorDistanceResult with point estimates, CIs, permutation
    p-values, and the per-pair table for plotting.
    """
    from scipy.stats import pearsonr, spearmanr

    refs = list(mean_matrix.index)
    if list(mean_matrix.columns) != refs:
        raise ValueError("mean_matrix must be square with matching row/col order")

    need_rest = "rest" in refs
    graph = build_graph(
        ch_names, k=k_laplacian, montage=montage, include_rest=need_rest,
    )

    # Estimate linear operator per reference (averaged over probes for
    # median's linearization variance reduction).
    ops: Dict[str, np.ndarray] = {}
    for r in refs:
        ops[r] = _estimate_linear_operator(
            r, graph, n_times=n_probe_times, seed=seed, n_probes=n_probes,
        )

    # Pairwise Frobenius distance
    n = len(refs)
    D_op = np.zeros((n, n))
    for i, a in enumerate(refs):
        for j, b in enumerate(refs):
            D_op[i, j] = np.linalg.norm(ops[a] - ops[b], ord="fro")

    # Transfer gap
    M = mean_matrix.to_numpy().astype(np.float64)
    Msym = 0.5 * (M + M.T)
    diag_mean = float(np.diag(M).mean())
    gap = diag_mean - Msym
    np.fill_diagonal(gap, 0.0)

    # Correlate upper triangle
    iu = np.triu_indices(n, k=1)
    dist_flat = D_op[iu]
    gap_flat = gap[iu]

    r_s, p_s = spearmanr(dist_flat, gap_flat)
    r_p, p_p = pearsonr(dist_flat, gap_flat)

    # Bootstrap CIs over pairs (resample pair indices with replacement)
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
        (float(np.percentile(valid_rho, 2.5)),
         float(np.percentile(valid_rho, 97.5)))
        if len(valid_rho) else (float("nan"), float("nan"))
    )
    ci_pear = (
        (float(np.percentile(valid_pear, 2.5)),
         float(np.percentile(valid_pear, 97.5)))
        if len(valid_pear) else (float("nan"), float("nan"))
    )

    # Permutation test: shuffle operator-axis labels of the symmetric gap
    # matrix, recompute upper-triangle correlation, count exceedances.
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
    # +1 / +1 small-sample correction (Phipson & Smyth 2010)
    perm_p_s = (perm_count_s + 1) / (n_permutations + 1)
    perm_p_p = (perm_count_p + 1) / (n_permutations + 1)

    rows = []
    for i, j in zip(*iu):
        rows.append({
            "ref_i": refs[i],
            "ref_j": refs[j],
            "distance_frobenius": float(D_op[i, j]),
            "transfer_gap": float(gap[i, j]),
        })
    pair_table = pd.DataFrame(rows)

    return OperatorDistanceResult(
        references=refs,
        distances_frobenius=D_op,
        transfer_gaps=gap,
        spearman_rho=float(r_s),
        spearman_p=float(p_s),
        pearson_r=float(r_p),
        pearson_p=float(p_p),
        perm_p_spearman=float(perm_p_s),
        perm_p_pearson=float(perm_p_p),
        ci95_spearman=ci_rho,
        ci95_pearson=ci_pear,
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
    """Scatter plot of operator distance vs transfer gap, with correlation in the title.

    Returns the matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    df = result.pair_table
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.scatter(df["distance_frobenius"], df["transfer_gap"], s=36, alpha=0.8)
    ax.set_xlabel("Operator Frobenius distance  ‖A_i − A_j‖_F")
    ax.set_ylabel("Transfer gap  (diag − symmetric transfer)")
    if title is None:
        title = (
            f"Operator distance vs transfer gap\n"
            f"Spearman ρ = {result.spearman_rho:.3f}  (p = {result.spearman_p:.1e})"
        )
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if annotate:
        for _, row in df.iterrows():
            label = f"{row['ref_i']}↔{row['ref_j']}"
            ax.annotate(
                label,
                xy=(row["distance_frobenius"], row["transfer_gap"]),
                xytext=(3, 3), textcoords="offset points",
                fontsize=7, alpha=0.7,
            )

    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, bbox_inches="tight", dpi=dpi)
    return fig


# ---------------------------------------------------------------------------
# 4. Paired-Wilcoxon significance for jitter / LOFO experiments
# ---------------------------------------------------------------------------

def baseline_diagonal_view(baseline_df: pd.DataFrame) -> pd.DataFrame:
    """Extract per-(subject, seed, test_ref) diagonal accuracies from a
    ``run_mismatch`` baseline DataFrame.

    Returns a DataFrame with columns ``subject, seed, test_ref, accuracy``
    where each row corresponds to the same-reference (train_ref == test_ref)
    cell. This is the natural "clean condition" baseline to compare a jitter
    DataFrame against.
    """
    required = {"subject", "seed", "train_ref", "test_ref", "accuracy"}
    missing = required - set(baseline_df.columns)
    if missing:
        raise ValueError(f"baseline_df is missing columns: {sorted(missing)}")
    diag = baseline_df[baseline_df["train_ref"] == baseline_df["test_ref"]]
    return diag[["subject", "seed", "test_ref", "accuracy"]].reset_index(drop=True)


def baseline_col_off_diag_view(baseline_df: pd.DataFrame) -> pd.DataFrame:
    """Extract per-(subject, seed, test_ref) mean off-diagonal accuracy from a
    ``run_mismatch`` baseline DataFrame.

    For each (subject, seed, test_ref) triple, averages the accuracy across
    all rows where train_ref != test_ref (i.e. across the 6 alternative
    single-reference training models that did not see the test reference
    during training).

    This is the natural baseline to compare a LOFO jitter DataFrame against:
    both sides represent "the model never saw the test reference at training
    time" — but the baseline saw exactly one alternative reference, while
    LOFO saw 6. The Wilcoxon then tests whether multi-reference training
    confers an advantage on unseen references over single-reference training.
    """
    required = {"subject", "seed", "train_ref", "test_ref", "accuracy"}
    missing = required - set(baseline_df.columns)
    if missing:
        raise ValueError(f"baseline_df is missing columns: {sorted(missing)}")
    off = baseline_df[baseline_df["train_ref"] != baseline_df["test_ref"]]
    return (
        off.groupby(["subject", "seed", "test_ref"], as_index=False)["accuracy"]
           .mean()
    )


def _holm_bonferroni(p_values: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni step-down correction for a vector of p-values.

    Returns adjusted p-values; reject H0 at level alpha iff adjusted p < alpha.
    """
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
    """Per-test-reference paired Wilcoxon signed-rank test of df_a vs df_b.

    Both DataFrames must have columns ``subject, seed, test_ref, accuracy``
    with one row per (subject, seed, test_ref). They are inner-joined on
    those keys so only triples present in both contribute paired observations.
    For each unique test_ref, runs ``scipy.stats.wilcoxon`` on the per-pair
    differences ``accuracy_a - accuracy_b``. A "pooled" row (ignoring test_ref)
    is appended.

    Parameters
    ----------
    df_a, df_b : DataFrame
        Must contain ``subject, seed, test_ref, accuracy`` columns. Typical
        shapes:
          - jitter result: one row per (subject, seed, test_ref).
          - baseline view: produced by ``baseline_diagonal_view`` or
            ``baseline_col_off_diag_view`` from a ``run_mismatch`` DataFrame.
    label_a, label_b : str
        Column-name labels used in the returned summary.
    alternative : {'two-sided', 'greater', 'less'}
        Direction for ``scipy.stats.wilcoxon``. ``'greater'`` tests whether
        df_a's accuracy is *greater* than df_b's (i.e. df_a wins).
    correction : {'holm', None}
        Multiple-comparison correction across the per-test-ref p-values.
        Holm-Bonferroni is conservative but standard. None disables.

    Returns
    -------
    DataFrame with columns:
        test_ref, n_pairs, mean_<label_a>, mean_<label_b>, median_delta,
        mean_delta, wilcoxon_stat, p_value, p_adjusted

    The 'pooled' row uses Wilcoxon across all (subject, seed, test_ref) pairs
    flattened. Per-test-ref tests are corrected; the pooled p-value is
    reported uncorrected (it answers a different question: "is there an
    overall effect?" rather than "which test refs differ?").

    Notes
    -----
    Sign convention: ``delta = accuracy_a - accuracy_b``. A positive
    median_delta means df_a is on average higher than df_b for that test_ref.

    Wilcoxon's signed-rank test ignores zero-difference pairs by default
    (``zero_method='wilcox'``). If df_a == df_b for some pairs (rare with
    continuous accuracies), the effective n is reduced.
    """
    from scipy.stats import wilcoxon

    required = {"subject", "seed", "test_ref", "accuracy"}
    for name, df in (("df_a", df_a), ("df_b", df_b)):
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError(f"Unknown alternative: {alternative!r}")
    if correction not in (None, "holm"):
        raise ValueError(f"Unknown correction: {correction!r}")

    merged = pd.merge(
        df_a[["subject", "seed", "test_ref", "accuracy"]]
            .rename(columns={"accuracy": "acc_a"}),
        df_b[["subject", "seed", "test_ref", "accuracy"]]
            .rename(columns={"accuracy": "acc_b"}),
        on=["subject", "seed", "test_ref"],
        how="inner",
    )
    if merged.empty:
        raise ValueError(
            "After joining on (subject, seed, test_ref), no paired observations "
            "remain. Check that df_a and df_b cover the same subjects/seeds."
        )
    merged["delta"] = merged["acc_a"] - merged["acc_b"]

    rows = []
    test_refs = sorted(merged["test_ref"].unique())
    for ref in test_refs:
        sub = merged[merged["test_ref"] == ref]
        n = len(sub)
        if n == 0:
            continue
        if np.allclose(sub["delta"].to_numpy(), 0.0):
            stat, p = 0.0, 1.0
        else:
            res = wilcoxon(
                sub["acc_a"].to_numpy(),
                sub["acc_b"].to_numpy(),
                alternative=alternative,
                zero_method="wilcox",
            )
            stat = float(res.statistic)
            p = float(res.pvalue)
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
            merged["acc_a"].to_numpy(),
            merged["acc_b"].to_numpy(),
            alternative=alternative,
            zero_method="wilcox",
        )
        pooled_stat = float(res.statistic)
        pooled_p = float(res.pvalue)
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
    out = pd.concat([out, pd.DataFrame([pooled_row])], ignore_index=True)
    return out
