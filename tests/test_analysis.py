"""Tests for refshift.analysis. All synthetic; no MOABB or network needed."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from refshift.analysis import (
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
    """Long-form mismatch result with structure: diagonal ~0.60,
    global-mean cluster off-diag ~0.55, spatial-cluster off-diag ~0.30,
    cross-family off-diag ~0.30. 5 subjects, 1 seed, REFERENCE_MODES.
    """
    rng = np.random.default_rng(0)
    refs = list(REFERENCE_MODES)
    global_mean = {"native", "car", "median", "gs", "rest"}
    rows = []
    for subj in range(1, 6):
        for train_ref in refs:
            for test_ref in refs:
                if train_ref == test_ref:
                    base = 0.60
                elif train_ref in global_mean and test_ref in global_mean:
                    base = 0.55
                elif train_ref == "laplacian" and test_ref == "bipolar":
                    base = 0.32
                elif train_ref == "bipolar" and test_ref == "laplacian":
                    base = 0.32
                else:
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
    subset_df = synthetic_df[synthetic_df["train_ref"].isin(["native", "car", "bipolar"])]
    subset_df = subset_df[subset_df["test_ref"].isin(["native", "car", "bipolar"])]
    S = mismatch_std_matrix(subset_df)
    assert set(S.index) == {"native", "car", "bipolar"}
    assert set(S.columns) == {"native", "car", "bipolar"}


# ---------------------------------------------------------------------------
# 2. clustering
# ---------------------------------------------------------------------------

def test_cluster_references_recovers_one_cluster_plus_two_isolates(synthetic_mean_matrix):
    """With synthetic data that explicitly has {global-mean family} + laplacian
    + bipolar as three behavioural groups, k=3 clustering should recover them.
    """
    result = cluster_references(synthetic_mean_matrix)
    assert 3 in result.clusters
    clusters_k3 = result.clusters[3]
    # Flatten for easier assertion
    as_sets = [set(c) for c in clusters_k3]
    global_mean = {"native", "car", "median", "gs", "rest"}
    assert global_mean in as_sets, (
        f"Expected {global_mean} as one cluster, got {clusters_k3}"
    )
    assert {"laplacian"} in as_sets, f"laplacian should be its own cluster, got {clusters_k3}"
    assert {"bipolar"} in as_sets, f"bipolar should be its own cluster, got {clusters_k3}"


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
    operator-distance ↔ transfer-gap correlation should be strongly
    positive. Set a loose bar so we tolerate real-world noise; we just
    want to catch sign-flips and broken math.
    """
    pytest.importorskip("mne")
    result = operator_distance_correlation(
        synthetic_mean_matrix, iv2a_ch_names,
    )
    assert result.spearman_rho > 0.4, (
        f"Expected positive operator-distance/transfer-gap correlation, "
        f"got ρ={result.spearman_rho:.3f}"
    )
    assert result.spearman_p < 0.05, (
        f"Expected statistically significant correlation, got p={result.spearman_p:.3f}"
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
