"""
Tests for refshift.data.loader.

Runs end-to-end loading for one subject per dataset, plus structural checks:
    - expected tensor shapes (N, C, T)
    - expected channel counts
    - expected sample rate
    - label ranges and class balance
    - no NaN/Inf
    - dtype and contiguity
    - cache roundtrip equivalence

Total runtime on Kaggle: ~1-2 minutes.
"""

from __future__ import annotations

import sys
import tempfile
import traceback
from collections import Counter

import numpy as np

from loader import (
    ALL_SUBJECTS,
    EXCLUDED_SUBJECTS,
    EXPECTED_N_CHANNELS,
    EXPECTED_SAMPLES,
    EXPECTED_TRIALS,
    SubjectData,
    TARGET_SFREQ_HZ,
    load_subject,
    load_subject_cache,
    save_subject_cache,
)


# ============================================================================
# Assertion helpers
# ============================================================================

def _check_array(X: np.ndarray, name: str, expected_shape_suffix: tuple):
    assert isinstance(X, np.ndarray), f"{name}: not a numpy array"
    assert X.dtype == np.float32, f"{name}: dtype is {X.dtype}, expected float32"
    assert X.flags["C_CONTIGUOUS"], f"{name}: not C-contiguous"
    assert X.ndim == 3, f"{name}: shape {X.shape} has ndim {X.ndim}, expected 3"
    assert X.shape[1:] == expected_shape_suffix, (
        f"{name}: shape {X.shape} does not match expected "
        f"[..., {expected_shape_suffix[0]}, {expected_shape_suffix[1]}]"
    )
    assert np.isfinite(X).all(), f"{name}: contains NaN or Inf"


def _check_labels(y: np.ndarray, name: str, allowed: set, min_per_class: int = 1):
    assert isinstance(y, np.ndarray), f"{name}: not a numpy array"
    assert y.dtype == np.int64, f"{name}: dtype is {y.dtype}, expected int64"
    assert y.ndim == 1, f"{name}: ndim {y.ndim}, expected 1"
    seen = set(y.tolist())
    assert seen <= allowed, f"{name}: labels {seen} include values outside {allowed}"
    counts = Counter(y.tolist())
    for c in seen:
        assert counts[c] >= min_per_class, (
            f"{name}: class {c} has only {counts[c]} trials (< {min_per_class})"
        )


def _report(result: SubjectData, title: str):
    print(f"\n---- {title} ----")
    print(f"  dataset={result.dataset_id}, subject={result.subject}")
    print(f"  ch_names: n={len(result.ch_names)}, "
          f"first 5 = {result.ch_names[:5]}")
    print(f"  sfreq={result.sfreq}")
    if result.has_session_split():
        print(f"  X_train={result.X_train.shape}, y_train={result.y_train.shape}, "
              f"classes={dict(Counter(result.y_train.tolist()))}")
        print(f"  X_test={result.X_test.shape},  y_test={result.y_test.shape},  "
              f"classes={dict(Counter(result.y_test.tolist()))}")
        print(f"  X_train range: [{result.X_train.min():.3g}, "
              f"{result.X_train.max():.3g}], mean|X|={np.abs(result.X_train).mean():.3g}")
    else:
        print(f"  X_all={result.X_all.shape}, y_all={result.y_all.shape}, "
              f"classes={dict(Counter(result.y_all.tolist()))}")
        print(f"  X_all range: [{result.X_all.min():.3g}, "
              f"{result.X_all.max():.3g}], mean|X|={np.abs(result.X_all).mean():.3g}")


# ============================================================================
# Dataset-specific tests
# ============================================================================

def test_iv2a():
    result = load_subject("iv2a", 1)
    _report(result, "IV-2a subject 1")

    C = EXPECTED_N_CHANNELS["iv2a"]
    T = EXPECTED_SAMPLES["iv2a"]
    _check_array(result.X_train, "iv2a.X_train", (C, T))
    _check_array(result.X_test,  "iv2a.X_test",  (C, T))
    _check_labels(result.y_train, "iv2a.y_train", allowed={0, 1, 2, 3}, min_per_class=50)
    _check_labels(result.y_test,  "iv2a.y_test",  allowed={0, 1, 2, 3}, min_per_class=50)

    # Trial count is a range because MOABB drops boundary epochs (last trial
    # per run). See loader module docstring on EXPECTED_TRIALS.
    lo_tr, hi_tr = EXPECTED_TRIALS["iv2a"]["train_min"], EXPECTED_TRIALS["iv2a"]["train_max"]
    lo_te, hi_te = EXPECTED_TRIALS["iv2a"]["test_min"],  EXPECTED_TRIALS["iv2a"]["test_max"]
    assert lo_tr <= result.X_train.shape[0] <= hi_tr, (
        f"iv2a train trial count: got {result.X_train.shape[0]}, "
        f"expected [{lo_tr}, {hi_tr}]"
    )
    assert lo_te <= result.X_test.shape[0] <= hi_te, (
        f"iv2a test trial count: got {result.X_test.shape[0]}, "
        f"expected [{lo_te}, {hi_te}]"
    )
    assert result.sfreq == TARGET_SFREQ_HZ
    assert len(result.ch_names) == C

    mean_abs = np.abs(result.X_train).mean()
    assert 1e-7 < mean_abs < 1e-2, (
        f"iv2a mean|X|={mean_abs:.3g} outside plausible EEG range"
    )
    print("  IV-2a: OK")


def test_openbmi():
    result = load_subject("openbmi", 1)
    _report(result, "OpenBMI subject 1")

    C = EXPECTED_N_CHANNELS["openbmi"]
    T = EXPECTED_SAMPLES["openbmi"]
    _check_array(result.X_train, "openbmi.X_train", (C, T))
    _check_array(result.X_test,  "openbmi.X_test",  (C, T))
    _check_labels(result.y_train, "openbmi.y_train", allowed={0, 1}, min_per_class=40)
    _check_labels(result.y_test,  "openbmi.y_test",  allowed={0, 1}, min_per_class=40)

    assert result.X_train.shape[0] == EXPECTED_TRIALS["openbmi"]["train"]
    assert result.X_test.shape[0]  == EXPECTED_TRIALS["openbmi"]["test"]
    assert result.sfreq == TARGET_SFREQ_HZ
    assert len(result.ch_names) == C

    assert 29 in EXCLUDED_SUBJECTS["openbmi"]
    assert 29 not in ALL_SUBJECTS["openbmi"]

    # Units sanity check: after μV→V conversion, typical EEG is ~1e-5 V
    mean_abs = np.abs(result.X_train).mean()
    assert 1e-7 < mean_abs < 1e-3, (
        f"openbmi mean|X|={mean_abs:.3g} outside plausible EEG range after μV→V"
    )
    print("  OpenBMI: OK")


def test_openbmi_excluded_subject_rejects():
    try:
        load_subject("openbmi", 29)
    except ValueError as e:
        assert "excluded" in str(e).lower() or "corrupt" in str(e).lower()
        print(f"  OpenBMI subj29 correctly rejected: {e}")
        return
    raise AssertionError("Loading OpenBMI subject 29 did not raise ValueError")


def test_cho2017():
    result = load_subject("cho2017", 1)
    _report(result, "Cho2017 subject 1")

    C = EXPECTED_N_CHANNELS["cho2017"]
    T = EXPECTED_SAMPLES["cho2017"]
    _check_array(result.X_all, "cho2017.X_all", (C, T))
    _check_labels(result.y_all, "cho2017.y_all", allowed={0, 1}, min_per_class=80)

    n_trials = result.X_all.shape[0]
    lo, hi = EXPECTED_TRIALS["cho2017"]["total_min"], EXPECTED_TRIALS["cho2017"]["total_max"]
    assert lo <= n_trials <= hi, (
        f"cho2017 n_trials={n_trials} outside expected [{lo}, {hi}]"
    )
    assert result.sfreq == TARGET_SFREQ_HZ
    assert len(result.ch_names) == C
    print("  Cho2017: OK")


def test_dreyer2023():
    result = load_subject("dreyer2023", 1)
    _report(result, "Dreyer2023 subject 1")

    C = EXPECTED_N_CHANNELS["dreyer2023"]
    T = EXPECTED_SAMPLES["dreyer2023"]
    _check_array(result.X_all, "dreyer.X_all", (C, T))
    _check_labels(result.y_all, "dreyer.y_all", allowed={0, 1}, min_per_class=30)

    expected_n = EXPECTED_TRIALS["dreyer2023"]["total"]
    assert result.X_all.shape[0] == expected_n, (
        f"dreyer n_trials={result.X_all.shape[0]}, expected {expected_n}"
    )
    counts = Counter(result.y_all.tolist())
    assert counts[0] == 40 and counts[1] == 40, (
        f"dreyer class balance {dict(counts)}, expected 40/40"
    )
    assert result.sfreq == TARGET_SFREQ_HZ
    assert len(result.ch_names) == C
    print("  Dreyer2023: OK")


def test_cache_roundtrip():
    result = load_subject("iv2a", 1)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = save_subject_cache(result, tmpdir)
        assert path.exists()
        reloaded = load_subject_cache(tmpdir, "iv2a", 1)
    assert np.array_equal(result.X_train, reloaded.X_train)
    assert np.array_equal(result.y_train, reloaded.y_train)
    assert np.array_equal(result.X_test,  reloaded.X_test)
    assert np.array_equal(result.y_test,  reloaded.y_test)
    assert result.ch_names == reloaded.ch_names
    assert result.sfreq == reloaded.sfreq
    print("  Cache roundtrip: OK")


# ============================================================================
# Main
# ============================================================================

TESTS = [
    ("iv2a loader",               test_iv2a),
    ("openbmi loader",            test_openbmi),
    ("openbmi subj 29 rejected",  test_openbmi_excluded_subject_rejects),
    ("cho2017 loader",            test_cho2017),
    ("dreyer2023 loader",         test_dreyer2023),
    ("cache roundtrip",           test_cache_roundtrip),
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
        print("\nAll loader tests PASSED.")
    return failed


if __name__ == "__main__":
    failed = run_all()
    sys.exit(1 if failed else 0)
