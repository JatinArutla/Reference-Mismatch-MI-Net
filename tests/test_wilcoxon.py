"""Unit tests for refshift.analysis.paired_wilcoxon_per_test_ref and the
two baseline-view extractors. All synthetic — no MOABB / network calls.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from refshift.analysis import (
    _holm_bonferroni,
    baseline_col_off_diag_view,
    baseline_diagonal_view,
    paired_wilcoxon_per_test_ref,
)


# ----- Fixtures --------------------------------------------------------------

REFS = ["native", "car", "median", "laplacian", "nn_diff", "rest"]


def _make_baseline_df(seed: int = 0, n_subjects: int = 9, n_seeds: int = 3,
                     diag_acc: float = 0.65, off_acc: float = 0.43,
                     noise_sd: float = 0.05) -> pd.DataFrame:
    """Synthetic baseline run_mismatch DataFrame with realistic IV-2a shape:
    per-(subject, seed, train_ref, test_ref) accuracy. Diagonal cells get
    diag_acc + noise; off-diagonal cells get off_acc + noise.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for subject in range(1, n_subjects + 1):
        for s in range(n_seeds):
            for tr in REFS:
                for te in REFS:
                    base = diag_acc if tr == te else off_acc
                    rows.append({
                        "subject": subject, "seed": s,
                        "train_ref": tr, "test_ref": te,
                        "accuracy": float(np.clip(base + rng.normal(0, noise_sd), 0, 1)),
                    })
    return pd.DataFrame(rows)


def _make_jitter_df(seed: int = 0, n_subjects: int = 9, n_seeds: int = 3,
                    acc_by_ref: dict | None = None,
                    noise_sd: float = 0.05) -> pd.DataFrame:
    """Synthetic jitter DataFrame: per-(subject, seed, test_ref) accuracy."""
    rng = np.random.default_rng(seed)
    if acc_by_ref is None:
        acc_by_ref = {ref: 0.65 for ref in REFS}
    rows = []
    for subject in range(1, n_subjects + 1):
        for s in range(n_seeds):
            for ref in REFS:
                base = acc_by_ref[ref]
                rows.append({
                    "subject": subject, "seed": s,
                    "test_ref": ref,
                    "accuracy": float(np.clip(base + rng.normal(0, noise_sd), 0, 1)),
                })
    return pd.DataFrame(rows)


# ----- Holm-Bonferroni -------------------------------------------------------

def test_holm_bonferroni_zero_pvalues_unchanged():
    p = np.array([0.0, 0.0, 0.0])
    out = _holm_bonferroni(p)
    np.testing.assert_array_equal(out, [0.0, 0.0, 0.0])


def test_holm_bonferroni_single_pvalue_unchanged():
    p = np.array([0.04])
    out = _holm_bonferroni(p)
    np.testing.assert_allclose(out, [0.04])


def test_holm_bonferroni_strict_ordering():
    """Smallest p gets multiplied by m, largest by 1; monotonicity enforced."""
    p = np.array([0.01, 0.02, 0.03])  # m=3
    out = _holm_bonferroni(p)
    # raw multipliers: 3*0.01=0.03, 2*0.02=0.04, 1*0.03=0.03
    # cumulative max -> 0.03, 0.04, 0.04
    np.testing.assert_allclose(out, [0.03, 0.04, 0.04], atol=1e-12)


def test_holm_bonferroni_caps_at_one():
    p = np.array([0.5, 0.6, 0.7])
    out = _holm_bonferroni(p)
    assert np.all(out <= 1.0)
    assert np.all(out >= p)


def test_holm_bonferroni_unsorted_input():
    """Result should be position-invariant w.r.t. input order, after un-sort."""
    p = np.array([0.03, 0.01, 0.02])
    out = _holm_bonferroni(p)
    # Same set of corrected values, just permuted to match input positions:
    # sorted [0.01, 0.02, 0.03] -> [0.03, 0.04, 0.04]
    # input order [0.03, 0.01, 0.02] -> [0.04, 0.03, 0.04]
    np.testing.assert_allclose(out, [0.04, 0.03, 0.04], atol=1e-12)


# ----- baseline_diagonal_view ------------------------------------------------

def test_baseline_diagonal_view_extracts_only_diagonal():
    df = _make_baseline_df()
    diag = baseline_diagonal_view(df)
    expected_n = 9 * 3 * 6  # subjects x seeds x refs
    assert len(diag) == expected_n
    assert set(diag.columns) == {"subject", "seed", "test_ref", "accuracy"}
    # Sanity: synthetic diag mean ~0.65 with noise_sd 0.05
    assert 0.60 < diag["accuracy"].mean() < 0.70


def test_baseline_diagonal_view_missing_cols_raises():
    df = pd.DataFrame({"subject": [1], "seed": [0], "accuracy": [0.5]})
    with pytest.raises(ValueError):
        baseline_diagonal_view(df)


# ----- baseline_col_off_diag_view --------------------------------------------

def test_baseline_col_off_diag_view_one_row_per_triple():
    df = _make_baseline_df()
    off = baseline_col_off_diag_view(df)
    expected_n = 9 * 3 * 6
    assert len(off) == expected_n
    # Synthetic off mean ~0.43
    assert 0.38 < off["accuracy"].mean() < 0.48


def test_baseline_col_off_diag_view_averaging_correct():
    """For a fixed (subject, seed, test_ref), the value should equal the mean
    of the 6 off-diagonal cells in the corresponding column."""
    df = _make_baseline_df()
    off = baseline_col_off_diag_view(df)
    # Pick one combination
    row = off[(off.subject == 3) & (off.seed == 0) & (off.test_ref == "car")].iloc[0]
    expected = df[
        (df.subject == 3) & (df.seed == 0) &
        (df.test_ref == "car") & (df.train_ref != "car")
    ]["accuracy"].mean()
    np.testing.assert_allclose(row["accuracy"], expected, atol=1e-12)


def test_baseline_col_off_diag_view_missing_cols_raises():
    df = pd.DataFrame({"subject": [1], "seed": [0], "accuracy": [0.5]})
    with pytest.raises(ValueError):
        baseline_col_off_diag_view(df)


# ----- paired_wilcoxon_per_test_ref ------------------------------------------

def test_wilcoxon_no_difference_high_pvalue():
    """Two identical synthetic dfs should give p ~ 1.0 everywhere."""
    df_a = _make_jitter_df(seed=42)
    df_b = df_a.copy()
    out = paired_wilcoxon_per_test_ref(df_a, df_b, label_a="A", label_b="B")
    # All test_refs + "pooled" = 7 rows
    assert len(out) == 7
    # All deltas are exactly zero -> our explicit guard returns p=1.0
    assert (out["p_value"] == 1.0).all()
    assert (out["mean_delta"] == 0.0).all()


def test_wilcoxon_clear_difference_low_pvalue():
    """A consistent +5pt boost across every (subject, seed, ref) should yield
    near-zero p with alternative='greater'."""
    df_b = _make_jitter_df(seed=0, noise_sd=0.0,
                           acc_by_ref={r: 0.55 for r in REFS})
    df_a = df_b.copy()
    df_a["accuracy"] = df_a["accuracy"] + 0.05  # uniform +5pt
    out = paired_wilcoxon_per_test_ref(
        df_a, df_b, label_a="boosted", label_b="orig", alternative="greater",
    )
    # Per-test-ref p-values should all be very small
    per_ref = out[out["test_ref"] != "pooled"]
    assert (per_ref["p_value"] < 0.01).all()
    assert (per_ref["mean_delta"] > 0.04).all()
    # Pooled p extremely small
    pooled = out[out["test_ref"] == "pooled"].iloc[0]
    assert pooled["p_value"] < 1e-10


def test_wilcoxon_columns_named_correctly():
    df_a = _make_jitter_df(seed=1)
    df_b = _make_jitter_df(seed=2)
    out = paired_wilcoxon_per_test_ref(df_a, df_b, label_a="jitter", label_b="baseline")
    assert "mean_jitter" in out.columns
    assert "mean_baseline" in out.columns
    assert "p_adjusted" in out.columns
    assert "wilcoxon_stat" in out.columns


def test_wilcoxon_pooled_row_present():
    df_a = _make_jitter_df(seed=1)
    df_b = _make_jitter_df(seed=2)
    out = paired_wilcoxon_per_test_ref(df_a, df_b)
    assert "pooled" in out["test_ref"].values
    pooled = out[out["test_ref"] == "pooled"].iloc[0]
    assert pooled["n_pairs"] == 9 * 3 * 6  # all triples


def test_wilcoxon_holm_correction_monotone():
    """Adjusted p-values should be >= raw p-values, per Holm-Bonferroni."""
    df_b = _make_jitter_df(seed=0, noise_sd=0.0,
                           acc_by_ref={r: 0.55 for r in REFS})
    df_a = df_b.copy()
    df_a.loc[df_a["test_ref"] == "nn_diff", "accuracy"] += 0.10
    out = paired_wilcoxon_per_test_ref(
        df_a, df_b, alternative="greater",
    )
    per_ref = out[out["test_ref"] != "pooled"]
    assert (per_ref["p_adjusted"] >= per_ref["p_value"] - 1e-12).all()


def test_wilcoxon_no_correction_pass_through():
    df_a = _make_jitter_df(seed=1)
    df_b = _make_jitter_df(seed=2)
    out = paired_wilcoxon_per_test_ref(df_a, df_b, correction=None)
    per_ref = out[out["test_ref"] != "pooled"]
    np.testing.assert_allclose(
        per_ref["p_adjusted"].to_numpy(), per_ref["p_value"].to_numpy(), atol=1e-12,
    )


def test_wilcoxon_inner_join_when_subjects_disjoint_raises():
    """If df_a and df_b share no (subject, seed, test_ref) triples, the
    function should raise rather than return an empty result."""
    df_a = _make_jitter_df(seed=1, n_subjects=3)
    df_b = _make_jitter_df(seed=2, n_subjects=3)
    df_b["subject"] += 100  # disjoint subject IDs
    with pytest.raises(ValueError, match="no paired observations"):
        paired_wilcoxon_per_test_ref(df_a, df_b)


def test_wilcoxon_missing_columns_raises():
    df_a = pd.DataFrame({"subject": [1], "seed": [0], "test_ref": ["car"], "accuracy": [0.5]})
    df_b = pd.DataFrame({"subject": [1], "seed": [0], "accuracy": [0.5]})  # no test_ref
    with pytest.raises(ValueError, match="missing columns"):
        paired_wilcoxon_per_test_ref(df_a, df_b)


def test_wilcoxon_unknown_alternative_raises():
    df_a = _make_jitter_df()
    df_b = _make_jitter_df()
    with pytest.raises(ValueError, match="alternative"):
        paired_wilcoxon_per_test_ref(df_a, df_b, alternative="bogus")


# ----- Integration: realistic jitter-vs-baseline shape -----------------------

def test_full_pipeline_baseline_diagonal_vs_jitter():
    """End-to-end: synthetic baseline df + synthetic jitter df, run
    full pipeline (extractor + Wilcoxon)."""
    baseline = _make_baseline_df(seed=0)
    jitter = _make_jitter_df(seed=1, acc_by_ref={r: 0.65 for r in REFS})
    diag = baseline_diagonal_view(baseline)
    out = paired_wilcoxon_per_test_ref(
        jitter, diag, label_a="jitter", label_b="baseline_diag",
    )
    # Should have 7 rows (6 refs + pooled)
    assert len(out) == 7
    # Both means should be in the 0.6-0.7 range (synthetic)
    per_ref = out[out["test_ref"] != "pooled"]
    assert (per_ref["mean_jitter"].between(0.55, 0.75)).all()
    assert (per_ref["mean_baseline_diag"].between(0.55, 0.75)).all()


def test_full_pipeline_baseline_off_vs_lofo():
    """End-to-end: LOFO NN-diff should beat baseline col-off-diag for NN-diff
    if we synthesize it that way."""
    baseline = _make_baseline_df(seed=0, diag_acc=0.65, off_acc=0.36, noise_sd=0.03)
    # LOFO NN-diff mimics the user's actual result: NN-diff test ~0.43 vs
    # other test refs ~0.65
    acc = {r: 0.65 for r in REFS}
    acc["nn_diff"] = 0.43
    lofo = _make_jitter_df(seed=1, acc_by_ref=acc, noise_sd=0.03)
    off_view = baseline_col_off_diag_view(baseline)
    out = paired_wilcoxon_per_test_ref(
        lofo, off_view, label_a="lofo", label_b="baseline_off",
        alternative="greater",
    )
    # Bipolar LOFO ~0.43 vs baseline off ~0.36 -> should be significant
    bip = out[out["test_ref"] == "nn_diff"].iloc[0]
    assert bip["mean_delta"] > 0.04  # ~7pt difference (0.43-0.36)
    assert bip["p_value"] < 0.01
