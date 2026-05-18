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
    """Long-form mismatch result with KNOWN cluster structure for testing
    clustering and analysis mechanics. The specific numerical values are
    arbitrary fixture choices, NOT empirical predictions about real EEG.

    Cluster structure used for the fixture:
      - Diagonal cells:          0.60
      - Within global-mean family ({native, car, median, rest}): 0.55
      - Within spatial-derivative family ({lap_small, lap_large}): 0.45
      - CSD as own cluster:      0.18 to/from anything else
      - Cross-family (everything else): 0.30

    This deliberately makes CSD a maximally separated cluster at k=2 so the
    clustering-mechanics tests have a clean, unambiguous ground-truth to
    recover. It does NOT claim that real-world CSD transfer is 0.18 or
    that the failure is amplitude-driven. The empirical CSD behavior is an
    open scientific question handled in the full mismatch experiments;
    the synthetic_neutral_df fixture below is the scale-neutral variant
    for operator-distance tests that should not bake in any CSD outlier
    interpretation.
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
                    # Fixture choice: maximally separated cluster for testing
                    # clustering mechanics. NOT an empirical claim.
                    base = 0.18
                elif train_ref in global_mean and test_ref in global_mean:
                    base = 0.55
                elif train_ref in spatial and test_ref in spatial:
                    base = 0.45
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
def synthetic_neutral_df():
    """Scale-neutral synthetic data: CSD has the same transfer profile as
    any other spatial-derivative operator. Use this for operator-distance
    tests so they don't depend on CSD being a structural outlier.

    Cluster structure:
      - Diagonal cells:        0.60
      - Within global-mean ({native, car, median, rest}): 0.55
      - Within spatial-derivative ({lap_small, lap_large, csd}): 0.45
      - cz_ref vs anything (asymmetric global): 0.30
      - Cross-family (global <-> spatial): 0.30
    """
    rng = np.random.default_rng(0)
    refs = list(REFERENCE_MODES)
    global_mean = {"native", "car", "median", "rest"}
    spatial_deriv = {"lap_small", "lap_large", "csd"}
    rows = []
    for subj in range(1, 6):
        for train_ref in refs:
            for test_ref in refs:
                if train_ref == test_ref:
                    base = 0.60
                elif train_ref in global_mean and test_ref in global_mean:
                    base = 0.55
                elif train_ref in spatial_deriv and test_ref in spatial_deriv:
                    base = 0.45
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
def synthetic_neutral_mean_matrix(synthetic_neutral_df):
    """Scale-neutral mean matrix counterpart."""
    return synthetic_neutral_df.groupby(["train_ref", "test_ref"])["accuracy"].mean().unstack("test_ref").reindex(
        index=list(REFERENCE_MODES), columns=list(REFERENCE_MODES),
    )


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
    and distance to CAR should be ~sqrt(C*1/C) ~ 1.0 in the raw Frobenius
    metric regardless of data.

    v0.16: pass distance_metric="frobenius_raw" explicitly. The default became
    'frobenius_normed' in v0.16 to control for CSD's amplitude scale; the
    closed-form ||J/C||_F = 1 identity holds only under raw distance.
    """
    pytest.importorskip("mne")
    result = operator_distance_correlation(
        synthetic_mean_matrix, iv2a_ch_names,
        distance_metric="frobenius_raw",
    )
    refs = result.references
    i = refs.index("native")
    j = refs.index("car")
    # native vs native = 0 by construction
    assert result.distances_frobenius[i, i] == pytest.approx(0.0, abs=1e-10)
    # native vs CAR: ||I - (I - J/C)||_F = ||J/C||_F = sqrt(C * C * 1/C^2) = 1
    assert result.distances_frobenius[i, j] == pytest.approx(1.0, abs=0.1)
    assert result.distance_metric == "frobenius_raw"


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


# ---------------------------------------------------------------------------
# v0.16: distance_metric in operator_distance_correlation
# ---------------------------------------------------------------------------

def test_operator_distance_metric_normed_default(iv2a_ch_names):
    """v0.16: default distance_metric is 'frobenius_normed'. Verify the
    OperatorDistanceResult records the metric and that normed distances
    are in [0, sqrt(2)] for every pair (each operator has unit Frobenius
    norm by construction)."""
    pytest.importorskip("mne")
    import pandas as pd
    refs = ["native", "car", "lap_small", "csd"]
    rng = np.random.default_rng(0)
    M = 0.4 + 0.05 * rng.standard_normal((4, 4))
    np.fill_diagonal(M, 0.7)
    df = pd.DataFrame(M, index=refs, columns=refs)
    res = operator_distance_correlation(
        df, iv2a_ch_names,
        n_permutations=100, n_bootstrap=100,
    )
    # Default in v0.16 is frobenius_normed.
    assert res.distance_metric == "frobenius_normed"
    # Normed distances bounded in [0, sqrt(2)]: each normalized operator
    # has unit Frobenius norm, so ||A_norm_i - A_norm_j||_F <= 2 by
    # triangle inequality, and equality only at "opposite" operators.
    # In practice values stay well under sqrt(2).
    D = res.distances_frobenius
    assert D.shape == (4, 4)
    assert (D >= 0).all()
    assert (D <= np.sqrt(2.0) + 1e-9).all(), (
        f"normed distances exceed sqrt(2): max={D.max()}"
    )
    # Diagonal is exactly zero.
    np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-10)


def test_operator_distance_metric_raw_vs_normed_csd_dominates_raw(iv2a_ch_names):
    """v0.16: when CSD is included, raw Frobenius is dominated by CSD's
    amplitude scale. The scale-normed distance suppresses this, so the ratio
    of (CSD pair distance / non-CSD pair distance) should be much smaller
    under normed than under raw. This is the test that validates the new
    metric actually addresses the scale confound.
    """
    pytest.importorskip("mne")
    import pandas as pd
    refs = ["native", "car", "lap_small", "csd"]
    rng = np.random.default_rng(0)
    M = 0.4 + 0.05 * rng.standard_normal((4, 4))
    np.fill_diagonal(M, 0.7)
    df = pd.DataFrame(M, index=refs, columns=refs)

    raw = operator_distance_correlation(
        df, iv2a_ch_names,
        distance_metric="frobenius_raw",
        n_permutations=50, n_bootstrap=50,
    )
    nor = operator_distance_correlation(
        df, iv2a_ch_names,
        distance_metric="frobenius_normed",
        n_permutations=50, n_bootstrap=50,
    )

    # csd pair indices.
    rrefs = raw.references
    csd_i = rrefs.index("csd")
    car_i = rrefs.index("car")
    nat_i = rrefs.index("native")

    raw_csd_car = raw.distances_frobenius[csd_i, car_i]
    raw_nat_car = raw.distances_frobenius[nat_i, car_i]
    nor_csd_car = nor.distances_frobenius[csd_i, car_i]
    nor_nat_car = nor.distances_frobenius[nat_i, car_i]

    raw_ratio = raw_csd_car / max(raw_nat_car, 1e-9)
    nor_ratio = nor_csd_car / max(nor_nat_car, 1e-9)

    # The raw ratio is huge (CSD has Frobenius norm ~10^3 vs ~1 for others).
    # The normed ratio is small (both pairs have ||.||_F = 1 components).
    assert raw_ratio > 100, (
        f"Expected CSD-CAR raw distance >> native-CAR raw distance; "
        f"got raw_ratio={raw_ratio}"
    )
    assert nor_ratio < 10, (
        f"normed distance should suppress the scale dominance; got "
        f"nor_ratio={nor_ratio} (should be O(1), not O(1000))"
    )


def test_operator_distance_correlation_rejects_unknown_metric(iv2a_ch_names):
    """Unknown distance_metric must raise ValueError."""
    pytest.importorskip("mne")
    import pandas as pd
    refs = ["native", "car", "lap_small"]
    df = pd.DataFrame(
        np.diag([0.7, 0.7, 0.7]) + 0.4 * (1 - np.eye(3)),
        index=refs, columns=refs,
    )
    with pytest.raises(ValueError, match="Unknown distance_metric"):
        operator_distance_correlation(
            df, iv2a_ch_names,
            distance_metric="cosine_similarity",
            n_permutations=10, n_bootstrap=10,
        )


def test_operator_distance_correlation_neutral_fixture(
    synthetic_neutral_mean_matrix, iv2a_ch_names,
):
    """Scale-neutral fixture (CSD not baked as outlier). With
    frobenius_normed, the topology-driven distance should still produce a
    sensible Spearman correlation (likely positive) without CSD dominating.
    """
    pytest.importorskip("mne")
    res = operator_distance_correlation(
        synthetic_neutral_mean_matrix, iv2a_ch_names,
        n_permutations=100, n_bootstrap=100,
    )
    # Test mechanics: result has the right shape and metadata, doesn't crash.
    n = len(synthetic_neutral_mean_matrix)
    assert res.distances_frobenius.shape == (n, n)
    assert len(res.pair_table) == n * (n - 1) // 2
    assert res.distance_metric == "frobenius_normed"
    assert -1.0 <= res.spearman_rho <= 1.0
    assert 0.0 < res.perm_p_spearman <= 1.0


# ---------------------------------------------------------------------------
# v0.16: run_pre_ems_mismatch (mocked, DL-only)
# ---------------------------------------------------------------------------

def test_run_pre_ems_mismatch_produces_nxn_long_form(monkeypatch):
    """run_pre_ems_mismatch should call load_dl_data once per reference,
    train one model per train_ref, and score on all test_refs. Output is
    long-form with the expected columns and N*N cells per (subject, seed)."""
    pytest.importorskip("braindecode")
    pytest.importorskip("torch")
    import pandas as pd
    from refshift.experiments.pre_ems_mismatch import run_pre_ems_mismatch

    modes = ("native", "car", "lap_small")
    C, T, N = 4, 100, 16
    sfreq = 250.0
    ch_names = ["C1", "C2", "C3", "C4"]
    y = np.array([0, 1] * (N // 2), dtype=np.int64)
    metadata = pd.DataFrame({
        "session": ["0"] * N,
        "run": ["0"] * (N // 2) + ["1"] * (N // 2),
        "subject": [1] * N,
    })

    def fake_load_dl_data(dataset_id, subject, **kwargs):
        # Return slightly different data for each ref so train/test splits
        # are stable but predictions are not constant.
        rng = np.random.default_rng(hash(kwargs["pre_ems_reference"]) & 0xFFFF)
        X = rng.standard_normal((N, C, T)).astype(np.float32)
        return X, y, metadata, sfreq, ch_names

    class FakeDLModel:
        def __init__(self, *args, **kwargs):
            self.n_classes = kwargs.get("n_classes", 2)
        def fit(self, X, y):
            return self
        def predict(self, X):
            return np.zeros(X.shape[0], dtype=np.int64)

    class FakeDataset:
        code = "FAKE"
        subject_list = [1]

    class FakeParadigm:
        channels = ch_names

    monkeypatch.setattr(
        "refshift.experiments.pre_ems_mismatch.load_dl_data",
        fake_load_dl_data,
    )
    monkeypatch.setattr(
        "refshift.experiments.pre_ems_mismatch.resolve_dataset",
        lambda dataset_id: (FakeDataset(), FakeParadigm()),
    )
    monkeypatch.setattr(
        "refshift.experiments.pre_ems_mismatch.get_eeg_channel_names",
        lambda dataset, subject, paradigm: ch_names,
    )
    monkeypatch.setattr(
        "refshift.model.make_dl_model",
        lambda **kwargs: FakeDLModel(**kwargs),
    )
    # validate_reference_modes can pass through; ch_names has C1..C4 which
    # don't include Cz, but cz_ref isn't in our modes.

    df = run_pre_ems_mismatch(
        "fake_id",
        model="shallow",
        subjects=[1],
        seeds=[0],
        reference_modes=modes,
        progress=False,
    )
    # NxN long form: 3 train_ref * 3 test_ref * 1 subject * 1 seed = 9 rows.
    assert df.shape[0] == 9
    expected_cols = {
        "dataset", "subject", "seed", "pipeline",
        "train_ref", "test_ref", "accuracy", "kappa",
        "n_train", "n_test",
    }
    assert expected_cols.issubset(set(df.columns))
    # Every (train_ref, test_ref) cell present.
    pairs = set(zip(df["train_ref"], df["test_ref"]))
    expected_pairs = {(a, b) for a in modes for b in modes}
    assert pairs == expected_pairs
    # All rows tagged pipeline="pre_ems_mismatch".
    assert (df["pipeline"] == "pre_ems_mismatch").all()


def test_run_pre_ems_mismatch_rejects_csp_lda():
    """run_pre_ems_mismatch is DL-only because CSP+LDA doesn't use EMS."""
    from refshift.experiments.pre_ems_mismatch import run_pre_ems_mismatch
    with pytest.raises(ValueError, match="DL-only"):
        run_pre_ems_mismatch(
            "iv2a", model="csp_lda", subjects=[1], seeds=[0],
        )
