"""
Tests for refshift.experiments.

Smoke tests that run the full experiment runners with a tiny epoch count to
verify the plumbing is correct without burning compute. Real benchmark runs
use n_epochs=200, not 5.

Runtime on T4: ~3-5 minutes (dominated by 6 short ATCNet training runs
for the mismatch matrix smoke test).
"""

from __future__ import annotations

import sys
import traceback

import numpy as np
import pandas as pd

from experiments import (
    mismatch_matrix_mean,
    mismatch_matrix_std,
    run_jitter,
    run_mismatch_matrix,
)
from reference_ops import REFERENCE_MODES


# ============================================================================
# Tests
# ============================================================================

def test_mismatch_matrix_atcnet_smoke():
    """IV-2a subject 1, ATCNet, mechanistic, 5 epochs. Verify shape + finite."""
    df = run_mismatch_matrix(
        "iv2a", subject=1,
        model_type="atcnet", standardization="mechanistic",
        n_epochs=5, batch_size=32, seed=0, verbose=True,
    )
    # 6 train_refs x 6 test_refs = 36 rows
    assert len(df) == 36, f"expected 36 rows, got {len(df)}"
    assert set(df["train_ref"]) == set(REFERENCE_MODES)
    assert set(df["test_ref"])  == set(REFERENCE_MODES)
    assert df["accuracy"].notna().all()
    assert df["kappa"].notna().all()
    assert (df["accuracy"] >= 0).all() and (df["accuracy"] <= 1).all()
    # Diagonal (train_ref == test_ref) should on average beat off-diagonal
    diag = df[df["train_ref"] == df["test_ref"]]["accuracy"].mean()
    off  = df[df["train_ref"] != df["test_ref"]]["accuracy"].mean()
    assert diag >= off - 0.05, (
        f"Diagonal ({diag:.3f}) should not be much worse than off-diagonal "
        f"({off:.3f}) even at 5 epochs"
    )
    print(f"  ATCNet mismatch matrix: 36 rows, diag mean={diag:.3f}, off mean={off:.3f}")


def test_mismatch_matrix_csp_lda_smoke():
    """IV-2a subject 1, CSP+LDA. Fast (no training loop)."""
    df = run_mismatch_matrix(
        "iv2a", subject=1, model_type="csp_lda", seed=0, verbose=False,
    )
    assert len(df) == 36
    assert df["accuracy"].notna().all()
    diag = df[df["train_ref"] == df["test_ref"]]["accuracy"].mean()
    off  = df[df["train_ref"] != df["test_ref"]]["accuracy"].mean()
    # For CSP+LDA, the mismatch effect is usually strong: diag > off
    assert diag > 0.30, f"CSP+LDA diag mean accuracy {diag:.3f} too low"
    print(f"  CSP+LDA mismatch matrix: diag mean={diag:.3f}, off mean={off:.3f}")


def test_mismatch_matrix_deployment():
    """Deployment standardization pipeline runs end to end."""
    df = run_mismatch_matrix(
        "iv2a", subject=1,
        model_type="atcnet", standardization="deployment",
        n_epochs=3, batch_size=32, seed=0, verbose=False,
    )
    assert len(df) == 36
    assert (df["standardization"] == "deployment").all()
    assert df["accuracy"].notna().all()
    print("  deployment standardization smoke: OK")


def test_run_jitter_smoke():
    """Jitter runner: train on 3 refs, evaluate on 6, 5 epochs."""
    df = run_jitter(
        "iv2a", subject=1,
        training_refs=["native", "car", "laplacian"],
        n_epochs=5, batch_size=32, seed=0, verbose=True,
    )
    # 6 test_refs (default all 6), one row each
    assert len(df) == 6
    assert (df["model"] == "atcnet_jitter").all()
    assert df["accuracy"].notna().all()
    assert df["training_refs"].iloc[0] == "native|car|laplacian"
    print(f"  jitter smoke: test accs = {df['accuracy'].round(3).tolist()}")


def test_aggregation_helpers():
    """mismatch_matrix_mean and _std return 6x6 DataFrames."""
    # Build a fake long-form df
    rows = []
    rng = np.random.RandomState(0)
    for seed in range(3):
        for sub in range(1, 4):
            for tr in REFERENCE_MODES:
                for te in REFERENCE_MODES:
                    rows.append(dict(
                        dataset="iv2a", subject=sub, seed=seed,
                        train_ref=tr, test_ref=te,
                        accuracy=float(rng.uniform(0.3, 0.8)),
                        kappa=0.0, n_test=282,
                        model="atcnet", standardization="mechanistic",
                    ))
    df = pd.DataFrame(rows)
    m = mismatch_matrix_mean(df)
    s = mismatch_matrix_std(df)
    assert m.shape == (6, 6)
    assert s.shape == (6, 6)
    # Each row/col should be one of the 6 modes
    assert set(m.index)   == set(REFERENCE_MODES)
    assert set(m.columns) == set(REFERENCE_MODES)
    print(f"  aggregation helpers: mean/std shape {m.shape}")


def test_cho2017_single_session_split():
    """Single-session dataset uses stratified 80/20 split automatically."""
    df = run_mismatch_matrix(
        "cho2017", subject=1, model_type="csp_lda", seed=0, verbose=False,
    )
    assert len(df) == 36
    n_test = df["n_test"].iloc[0]
    # 200 trials * 0.2 = 40 test trials
    assert 35 <= n_test <= 45, f"unexpected n_test {n_test}"
    print(f"  cho2017 single-session split: n_test={n_test}")


# ============================================================================
# Main
# ============================================================================

TESTS = [
    ("aggregation helpers",        test_aggregation_helpers),
    ("CSP+LDA mismatch iv2a",      test_mismatch_matrix_csp_lda_smoke),
    ("CSP+LDA mismatch cho2017",   test_cho2017_single_session_split),
    ("ATCNet mismatch iv2a",       test_mismatch_matrix_atcnet_smoke),
    ("ATCNet deployment iv2a",     test_mismatch_matrix_deployment),
    ("jitter runner iv2a",         test_run_jitter_smoke),
]


def run_all():
    passed, failed = [], []
    for name, fn in TESTS:
        print(f"\n===== {name} =====")
        try:
            fn()
            passed.append(name)
        except Exception as e:
            traceback.print_exc()
            failed.append((name, repr(e)))
    print("\n" + "=" * 72)
    print(f"PASSED ({len(passed)}):")
    for n in passed:
        print(f"  PASS  {n}")
    if failed:
        print(f"\nFAILED ({len(failed)}):")
        for n, err in failed:
            print(f"  FAIL  {n}: {err}")
    else:
        print("\nAll chunk-5 tests PASSED.")
    return failed


if __name__ == "__main__":
    failed = run_all()
    sys.exit(1 if failed else 0)
