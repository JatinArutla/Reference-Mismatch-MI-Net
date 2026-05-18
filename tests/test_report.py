"""Unit tests for refshift.report.report_experiment.

Each test builds a synthetic long-form DataFrame matching what the runners
produce, calls report_experiment on it, and checks the returned matrix,
summary stats, and that CSV + PNG were written.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest


REFERENCE_MODES = (
    "native", "car", "median", "rest", "cz_ref",
    "lap_small", "lap_large", "csd",
)


# ---------------------------------------------------------------------------
# Synthetic frames
# ---------------------------------------------------------------------------

def _mismatch_frame(modes=REFERENCE_MODES, n_subjects=2, n_seeds=1):
    rng = np.random.default_rng(0)
    rows = []
    for s in range(1, n_subjects + 1):
        for k in range(n_seeds):
            for tr in modes:
                for te in modes:
                    acc = 0.7 if tr == te else 0.45
                    acc += float(rng.normal(0, 0.02))
                    rows.append({
                        "dataset": "TEST", "subject": s, "seed": k,
                        "train_ref": tr, "test_ref": te,
                        "accuracy": acc, "kappa": acc - 0.5,
                        "n_train": 100, "n_test": 25,
                    })
    return pd.DataFrame(rows)


def _jitter_full_frame(modes=REFERENCE_MODES, n_subjects=2, n_seeds=1):
    rng = np.random.default_rng(0)
    rows = []
    for s in range(1, n_subjects + 1):
        for k in range(n_seeds):
            for te in modes:
                rows.append({
                    "dataset": "TEST", "subject": s, "seed": k,
                    "condition": "full", "holdout_ref": "",
                    "train_modes": ",".join(modes),
                    "test_ref": te,
                    "accuracy": 0.6 + float(rng.normal(0, 0.02)),
                    "kappa": 0.1, "n_train": 100, "n_test": 25,
                })
    return pd.DataFrame(rows)


def _lofo_frame(modes=REFERENCE_MODES, n_subjects=2, n_seeds=1):
    rng = np.random.default_rng(0)
    rows = []
    for s in range(1, n_subjects + 1):
        for k in range(n_seeds):
            for h in modes:  # holdout_ref
                for te in modes:
                    # held-out test refs are harder than seen ones
                    acc = 0.45 if te == h else 0.6
                    acc += float(rng.normal(0, 0.02))
                    rows.append({
                        "dataset": "TEST", "subject": s, "seed": k,
                        "condition": "lofo", "holdout_ref": h,
                        "train_modes": ",".join(m for m in modes if m != h),
                        "test_ref": te,
                        "accuracy": acc, "kappa": acc - 0.5,
                        "n_train": 100, "n_test": 25,
                    })
    return pd.DataFrame(rows)


def _ems_frame(modes=REFERENCE_MODES, n_subjects=2, n_seeds=1):
    rng = np.random.default_rng(0)
    rows = []
    for s in range(1, n_subjects + 1):
        for k in range(n_seeds):
            for ref in modes:
                rows.append({
                    "dataset": "TEST", "subject": s, "seed": k,
                    "reference": ref,
                    "accuracy": 0.68 + float(rng.normal(0, 0.02)),
                    "kappa": 0.1, "n_train": 100, "n_test": 25,
                })
    return pd.DataFrame(rows)


def _bandpass_frame(n_subjects=2, n_seeds=1):
    rng = np.random.default_rng(0)
    bands = ["8.0-32.0", "6.0-32.0", "8.0-30.0"]
    rows = []
    for s in range(1, n_subjects + 1):
        for k in range(n_seeds):
            for tb in bands[:1]:  # one train band
                for te_b in bands:
                    acc = 0.68 if te_b == tb else 0.65
                    acc += float(rng.normal(0, 0.02))
                    rows.append({
                        "dataset": "TEST", "subject": s, "seed": k,
                        "reference": "native",
                        "train_band": tb, "test_band": te_b,
                        "accuracy": acc, "kappa": 0.1,
                        "n_train": 100, "n_test": 25,
                    })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_report_mismatch_returns_square_matrix(capsys):
    from refshift.report import report_experiment
    df = _mismatch_frame()
    with tempfile.TemporaryDirectory() as tmp:
        out = report_experiment(
            df, kind="mismatch", name="test_mismatch",
            results_dir=tmp, figs_dir=tmp, dataset="iv2a",
        )
    M = out["matrix"]
    assert M.shape == (8, 8)
    assert "diagonal_mean" in out["summary"]
    assert "gap" in out["summary"]
    # Diagonal should be higher than off-diagonal by construction
    assert out["summary"]["diagonal_mean"] > out["summary"]["off_diag_mean"]


def test_report_mismatch_writes_csv_and_png():
    from refshift.report import report_experiment
    df = _mismatch_frame()
    with tempfile.TemporaryDirectory() as tmp:
        out = report_experiment(
            df, kind="mismatch", name="test_mismatch",
            results_dir=tmp, figs_dir=tmp, dataset="iv2a",
            print_matrix=False,
        )
        assert os.path.exists(out["csv_path"])
        assert os.path.exists(out["fig_path"])
        # CSV round-trips
        rt = pd.read_csv(out["csv_path"])
        assert len(rt) == len(df)


def test_report_mismatch_respects_modes_subset():
    """Schirrmeister-style: 7 modes (no cz_ref) under v0.15's 8-mode set."""
    from refshift.report import report_experiment
    modes_no_cz = tuple(m for m in REFERENCE_MODES if m != "cz_ref")
    df = _mismatch_frame(modes=modes_no_cz)
    with tempfile.TemporaryDirectory() as tmp:
        out = report_experiment(
            df, kind="mismatch", name="test_schirr",
            results_dir=tmp, figs_dir=tmp,
            dataset="schirrmeister2017", print_matrix=False,
        )
    assert out["matrix"].shape == (7, 7)
    assert "cz_ref" not in out["matrix"].index


def test_report_jitter_full_shape_and_stats():
    from refshift.report import report_experiment
    df = _jitter_full_frame()
    with tempfile.TemporaryDirectory() as tmp:
        out = report_experiment(
            df, kind="jitter_full", name="test_jitter",
            results_dir=tmp, figs_dir=tmp, dataset="iv2a",
            print_matrix=False,
        )
    assert out["matrix"].shape == (1, 8)
    assert "mean" in out["summary"]
    assert "std" in out["summary"]


def test_report_lofo_shape_and_recovery_gap():
    from refshift.report import report_experiment
    df = _lofo_frame()
    with tempfile.TemporaryDirectory() as tmp:
        out = report_experiment(
            df, kind="lofo", name="test_lofo",
            results_dir=tmp, figs_dir=tmp, dataset="iv2a",
            print_matrix=False,
        )
    M = out["matrix"]
    assert M.shape == (8, 8)
    # By construction, held-out cells (diagonal of holdout vs test) are lower
    assert out["summary"]["recovery_gap"] > 0
    assert "held_out_mean" in out["summary"]
    assert "seen_mean" in out["summary"]


def test_report_ems_shape():
    from refshift.report import report_experiment
    df = _ems_frame()
    with tempfile.TemporaryDirectory() as tmp:
        out = report_experiment(
            df, kind="ems_diag", name="test_ems",
            results_dir=tmp, figs_dir=tmp, dataset="iv2a",
            print_matrix=False,
        )
    assert out["matrix"].shape == (8, 1)
    assert "mean" in out["summary"]


def test_report_bandpass_shape():
    from refshift.report import report_experiment
    df = _bandpass_frame()
    with tempfile.TemporaryDirectory() as tmp:
        out = report_experiment(
            df, kind="bandpass", name="test_bandpass",
            results_dir=tmp, figs_dir=tmp,
            print_matrix=False,
        )
    M = out["matrix"]
    # 1 train band × 3 test bands
    assert M.shape == (1, 3)


def test_report_invalid_kind_raises():
    from refshift.report import report_experiment
    df = _mismatch_frame()
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(ValueError, match="Unknown kind"):
            report_experiment(
                df, kind="not_a_kind", name="x",
                results_dir=tmp, figs_dir=tmp,
            )


def test_report_skip_csv_and_heatmap():
    """Both save flags off should not write either file."""
    from refshift.report import report_experiment
    df = _mismatch_frame()
    with tempfile.TemporaryDirectory() as tmp:
        out = report_experiment(
            df, kind="mismatch", name="nothing",
            results_dir=tmp, figs_dir=tmp, dataset="iv2a",
            save_csv=False, save_heatmap=False, print_matrix=False,
        )
        assert out["csv_path"] is None
        assert out["fig_path"] is None
        # No files in tmp
        assert os.listdir(tmp) == []


def test_report_jitter_handles_dataset_specific_modes():
    """Schirrmeister jitter has 7 test_refs (no cz_ref) under v0.15."""
    from refshift.report import report_experiment
    modes_no_cz = tuple(m for m in REFERENCE_MODES if m != "cz_ref")
    df = _jitter_full_frame(modes=modes_no_cz)
    with tempfile.TemporaryDirectory() as tmp:
        out = report_experiment(
            df, kind="jitter_full", name="schirr_jitter",
            results_dir=tmp, figs_dir=tmp,
            dataset="schirrmeister2017", print_matrix=False,
        )
    M = out["matrix"]
    assert M.shape == (1, 7)
    assert "cz_ref" not in M.columns
