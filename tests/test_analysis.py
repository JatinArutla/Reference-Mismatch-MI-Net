"""Tests for refshift.analysis. All synthetic; no MOABB or network needed."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from refshift.analysis import (
    baseline_diagonal_view,
    cluster_references,
    mismatch_std_matrix,
    operator_distance_correlation,
)
from refshift.reference import REFERENCE_MODES


# ---------------------------------------------------------------------------
# Fixtures: synthetic IV-2a-like long-form DataFrame and mean matrix
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_df():
    """Long-form mismatch result with structure designed to match what we
    expect from v0.15's 8-mode set:

      - Diagonal ~0.60.
      - Within global-mean family ({native, car, median, rest}): off-diag ~0.55.
      - Within spatial-derivative family ({lap_small, lap_large}): off-diag ~0.45.
      - Spatial-derivative vs global-mean: off-diag ~0.30.
      - cz_ref vs anything: off-diag ~0.30.
      - CSD vs anything: off-diag ~0.18 (CSD's amplitude scale is far
        bigger than any other operator; BatchNorm trained on CAR-scale
        data fails badly on CSD-scale data, and vice versa).

    With this structure k=2 clustering should separate global-mean from
    the spatial-derivative outliers, and the operator-distance / transfer-gap
    correlation should be positive (large operator distance -> large gap).
    """
    rng = np.random.default_rng(0)
    refs = list(REFERENCE_MODES)
    global_mean = {"native", "car", "median", "rest"}
    spatial = {"lap_small", "lap_large"}
    rows = []
    for subj in range(1, 6):
        for train_ref in refs:
            for test_ref in refs:
                if train_ref == test_ref:
                    base = 0.60
                elif "csd" in (train_ref, test_ref):
                    # CSD's amplitude-scale outlier behaviour: bad transfer
                    # to/from anything else.
                    base = 0.18
                elif train_ref in global_mean and test_ref in global_mean:
                    base = 0.55
                elif train_ref in spatial and test_ref in spatial:
                    base = 0.45
                else:
                    # cross-family (spatial <-> global, cz_ref <-> anything)
                    base = 0.30
                acc = base + 0.02 * rng.standard_normal()
                rows.append({
                    "subject": subj, "seed": 0,
                    "train_ref": train_ref, "test_ref": test_ref,
                    "accuracy": float(np.clip(acc, 0.0, 1.0)),
                })
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_mean_matrix(synthetic_df):
    return synthetic_df.groupby(["train_ref", "test_ref"])["accuracy"].mean().unstack("test_ref").reindex(
        index=list(REFERENCE_MODES), columns=list(REFERENCE_MODES),
    )


@pytest.fixture
def iv2a_ch_names():
    return [
        "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
        "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
        "CP3", "CP1", "CPz", "CP2", "CP4",
        "P1", "Pz", "P2", "POz",
    ]


# ---------------------------------------------------------------------------
# 1. std matrix
# ---------------------------------------------------------------------------

def test_std_matrix_shape_and_order(synthetic_df):
    S = mismatch_std_matrix(synthetic_df)
    assert S.shape == (len(REFERENCE_MODES), len(REFERENCE_MODES))
    assert list(S.index) == list(REFERENCE_MODES)
    assert list(S.columns) == list(REFERENCE_MODES)
    assert np.isfinite(S.to_numpy()).all()
    # All stds should be positive (synthetic noise was nonzero)
    assert (S.to_numpy() > 0).all()


def test_std_matrix_matches_direct_groupby(synthetic_df):
    """mismatch_std_matrix must equal groupby std directly."""
    S = mismatch_std_matrix(synthetic_df)
    expected = synthetic_df.groupby(["train_ref", "test_ref"])["accuracy"].std().unstack("test_ref")
    expected = expected.reindex(index=list(REFERENCE_MODES), columns=list(REFERENCE_MODES))
    np.testing.assert_allclose(S.to_numpy(), expected.to_numpy(), atol=1e-12)


def test_std_matrix_drops_missing_modes(synthetic_df):
    """If the df only has a subset of modes, the std matrix respects that."""
    subset_df = synthetic_df[synthetic_df["train_ref"].isin(["native", "car", "cz_ref"])]
    subset_df = subset_df[subset_df["test_ref"].isin(["native", "car", "cz_ref"])]
    S = mismatch_std_matrix(subset_df)
    assert set(S.index) == {"native", "car", "cz_ref"}
    assert set(S.columns) == {"native", "car", "cz_ref"}


# ---------------------------------------------------------------------------
# 2. clustering
# ---------------------------------------------------------------------------

def test_cluster_references_isolates_csd_at_k2(synthetic_mean_matrix):
    """With v0.15 synthetic data where CSD is a large-amplitude outlier
    (transfer gap to anything else ~0.42 vs ~0.30 for any other pair),
    k=2 clustering should isolate csd from everything else.
    """
    result = cluster_references(synthetic_mean_matrix)
    assert 2 in result.clusters
    clusters_k2 = result.clusters[2]
    as_sets = [set(c) for c in clusters_k2]
    # csd should be its own cluster at k=2.
    assert {"csd"} in as_sets, (
        f"Expected csd as a singleton cluster at k=2, got {clusters_k2}"
    )
    # The other cluster should contain the remaining 7 modes.
    others = set(REFERENCE_MODES) - {"csd"}
    assert others in as_sets, (
        f"Expected {others} as the non-csd cluster at k=2, got {clusters_k2}"
    )


def test_cluster_references_global_mean_clusters_together(synthetic_mean_matrix):
    """Across k values, the four global-mean refs (native, car, median, rest)
    should fall into a single cluster (they have the smallest pairwise
    transfer-gap distance, ~0.05, in the synthetic data)."""
    result = cluster_references(synthetic_mean_matrix)
    global_mean = {"native", "car", "median", "rest"}
    # At k=4, global_mean should be one of the four clusters.
    clusters_k4 = result.clusters[4]
    as_sets = [set(c) for c in clusters_k4]
    found = any(global_mean.issubset(c) for c in as_sets)
    assert found, (
        f"global_mean refs split across clusters at k=4: {clusters_k4}"
    )


def test_cluster_references_distance_properties(synthetic_mean_matrix):
    result = cluster_references(synthetic_mean_matrix)
    D = result.distance_matrix
    # Symmetric
    np.testing.assert_allclose(D, D.T, atol=1e-12)
    # Zero diagonal
    np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-12)
    # Non-negative
    assert (D >= -1e-12).all()
    # Shape matches refs
    assert D.shape == (len(result.references), len(result.references))


def test_cluster_references_linkage_has_correct_merges(synthetic_mean_matrix):
    """Linkage output has n-1 rows for n leaves."""
    result = cluster_references(synthetic_mean_matrix)
    n = len(result.references)
    assert result.linkage.shape == (n - 1, 4)


def test_cluster_references_rejects_non_square(synthetic_mean_matrix):
    bad = synthetic_mean_matrix.drop(columns=["native"])
    with pytest.raises(ValueError, match="square"):
        cluster_references(bad)


# ---------------------------------------------------------------------------
# 3. operator-distance correlation
# ---------------------------------------------------------------------------

def test_operator_distance_correlation_positive_and_significant(
    synthetic_mean_matrix, iv2a_ch_names
):
    """With synthetic data built from a clean family structure, the
    operator-distance ↔ transfer-gap correlation should be positive and
    point in the expected direction. Loose p-value bar (0.10) rather than
    0.05: with 8 references there are n=28 upper-triangle pairs (or 21
    when cz_ref is dropped on Schirrmeister). The headline check is the
    *direction* of the correlation; sign-flips or broken math would be
    caught by the ρ>0.4 assertion.
    """
    pytest.importorskip("mne")
    result = operator_distance_correlation(
        synthetic_mean_matrix, iv2a_ch_names,
    )
    assert result.spearman_rho > 0.4, (
        f"Expected positive operator-distance/transfer-gap correlation, "
        f"got ρ={result.spearman_rho:.3f}"
    )
    assert result.spearman_p < 0.10, (
        f"Expected directionally significant correlation (p<0.10 with "
        f"n=28 pairs), got p={result.spearman_p:.3f}"
    )


def test_operator_distance_result_shapes(synthetic_mean_matrix, iv2a_ch_names):
    pytest.importorskip("mne")
    result = operator_distance_correlation(
        synthetic_mean_matrix, iv2a_ch_names,
    )
    n = len(result.references)
    assert result.distances_frobenius.shape == (n, n)
    assert result.transfer_gaps.shape == (n, n)
    # Frobenius distance is symmetric, zero diag
    np.testing.assert_allclose(
        result.distances_frobenius, result.distances_frobenius.T, atol=1e-10,
    )
    np.testing.assert_allclose(np.diag(result.distances_frobenius), 0.0, atol=1e-10)
    # Pair table should have C(n,2) = n*(n-1)/2 rows
    assert len(result.pair_table) == n * (n - 1) // 2


def test_operator_distance_identity_row_is_small(synthetic_mean_matrix, iv2a_ch_names):
    """'native' is literally the identity operator; its distance to itself is 0
    and distance to CAR should be ~sqrt(C*1/C) ~ 1.0 regardless of data.
    """
    pytest.importorskip("mne")
    result = operator_distance_correlation(
        synthetic_mean_matrix, iv2a_ch_names,
    )
    refs = result.references
    i = refs.index("native")
    j = refs.index("car")
    # native vs native = 0 by construction
    assert result.distances_frobenius[i, i] == pytest.approx(0.0, abs=1e-10)
    # native vs CAR: ||I - (I - J/C)||_F = ||J/C||_F = sqrt(C * C * 1/C^2) = 1
    assert result.distances_frobenius[i, j] == pytest.approx(1.0, abs=0.1)


def test_operator_distance_correlation_returns_ci_and_perm_p():
    """Result must include bootstrap CI and permutation p-values
    (asymptotic stats are unreliable at n=15 pairs)."""
    pytest.importorskip("mne")
    import pandas as pd
    from refshift.analysis import (
        OperatorDistanceResult,
        operator_distance_correlation,
    )

    refs = ["native", "car", "median", "lap_small", "rest", "cz_ref"]
    rng = np.random.default_rng(0)
    M = 0.4 + 0.05 * rng.standard_normal((6, 6))
    np.fill_diagonal(M, 0.7)
    df = pd.DataFrame(M, index=refs, columns=refs)
    iv2a_chs = [
        "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz",
        "C2", "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz",
        "P2", "POz",
    ]
    res = operator_distance_correlation(
        df, iv2a_chs,
        n_probe_times=200, n_probes=2,
        n_permutations=100, n_bootstrap=100, seed=0,
    )
    assert isinstance(res, OperatorDistanceResult)
    assert hasattr(res, "ci95_spearman")
    assert hasattr(res, "ci95_pearson")
    assert hasattr(res, "perm_p_spearman")
    assert hasattr(res, "perm_p_pearson")
    lo_s, hi_s = res.ci95_spearman
    if not (np.isnan(lo_s) or np.isnan(hi_s)):
        assert lo_s <= hi_s
    assert 0.0 < res.perm_p_spearman <= 1.0
    assert 0.0 < res.perm_p_pearson <= 1.0


# ---------------------------------------------------------------------------
# v0.15 backward compatibility: legacy 'laplacian' in CSVs
# ---------------------------------------------------------------------------

def test_mismatch_std_matrix_resolves_laplacian_alias():
    """v0.14 CSVs with 'laplacian' rows should analyse correctly under v0.15,
    with 'laplacian' resolved to 'lap_small' before pivoting."""
    import pandas as pd
    rng = np.random.default_rng(0)
    old_modes = ["native", "car", "median", "laplacian", "rest", "cz_ref"]
    rows = []
    for tr in old_modes:
        for te in old_modes:
            for subj in range(1, 4):
                rows.append({
                    "subject": subj, "seed": 0,
                    "train_ref": tr, "test_ref": te,
                    "accuracy": 0.6 if tr == te else 0.4 + 0.05 * rng.standard_normal(),
                })
    df = pd.DataFrame(rows)
    S = mismatch_std_matrix(df)
    # 6 modes after alias resolution; 'lap_small' present, 'laplacian' absent.
    assert S.shape == (6, 6)
    assert "lap_small" in S.index
    assert "laplacian" not in S.index
    # Original df is not mutated.
    assert "laplacian" in df["train_ref"].unique()


def test_baseline_diagonal_view_resolves_laplacian_alias():
    """Same legacy-CSV handling for the baseline view helper."""
    import pandas as pd
    df = pd.DataFrame([
        {"subject": 1, "seed": 0, "train_ref": "laplacian", "test_ref": "laplacian", "accuracy": 0.7},
        {"subject": 1, "seed": 0, "train_ref": "laplacian", "test_ref": "car", "accuracy": 0.5},
        {"subject": 1, "seed": 0, "train_ref": "car", "test_ref": "car", "accuracy": 0.65},
    ])
    diag = baseline_diagonal_view(df)
    # Diagonal cells only; both should become 'lap_small' / 'car'.
    assert set(diag["test_ref"].unique()) == {"lap_small", "car"}
