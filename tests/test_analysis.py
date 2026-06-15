"""Tests for the analysis reporters on synthetic long-form result tables."""

import numpy as np
import pandas as pd

from refshift.analysis import (
    mismatch_matrix,
    mismatch_std_matrix,
    report_families,
    report_lofo,
    report_loro,
    report_matrix,
)
from refshift.references import FAMILIES, REFERENCE_MODES


def _mismatch_frame(seeds=(0, 1)):
    rows = []
    rng = np.random.default_rng(0)
    for s in (1, 2):
        for k in seeds:
            for tr in REFERENCE_MODES:
                for te in REFERENCE_MODES:
                    acc = 0.7 if tr == te else 0.5
                    acc += float(rng.normal(0, 0.02))
                    rows.append({"subject": s, "seed": k, "train_ref": tr,
                                 "test_ref": te, "accuracy": acc,
                                 "kappa": acc - 0.25, "n_train": 100, "n_test": 100})
    return pd.DataFrame(rows)


def _lofo_frame():
    mode_fam = {m: f for f, ms in FAMILIES.items() for m in ms}
    universe = [m for ms in FAMILIES.values() for m in ms]
    rng = np.random.default_rng(0)
    rows = []
    for s in (1, 2):
        for hf in FAMILIES:
            for te in universe:
                tf = mode_fam[te]
                acc = 0.45 if tf == hf else 0.6
                acc += float(rng.normal(0, 0.02))
                rows.append({"subject": s, "seed": 0, "condition": "lofo",
                             "holdout_family": hf, "test_family": tf,
                             "test_ref": te, "accuracy": acc})
    return pd.DataFrame(rows)


def test_mismatch_matrix_shape():
    df = _mismatch_frame()
    M = mismatch_matrix(df)
    assert M.shape == (len(REFERENCE_MODES), len(REFERENCE_MODES))
    S = mismatch_std_matrix(df)
    assert S.shape == M.shape


def test_report_matrix_runs_and_has_positive_gap():
    df = _mismatch_frame()
    M = report_matrix(df, title="t")
    # Diagonal (matched) built higher than off-diagonal, so gap should be > 0.
    A = M.to_numpy(dtype=float)
    diag = np.diag(A).mean()
    off = A[~np.eye(A.shape[0], dtype=bool)].mean()
    assert diag > off


def test_report_families_runs():
    df = _mismatch_frame()
    M = report_families(df, title="t")
    assert M.shape == (len(REFERENCE_MODES), len(REFERENCE_MODES))


def test_report_lofo_family_matrix():
    df = _lofo_frame()
    M = report_lofo(df, title="t")
    assert M.shape == (3, 3)
    assert list(M.index) == ["global", "single", "spatial"]


def test_report_loro_runs():
    # Reuse the mismatch frame shape but relabel train_ref as holdout_ref.
    df = _mismatch_frame().rename(columns={"train_ref": "holdout_ref"})
    M = report_loro(df, title="t")
    assert M.shape == (len(REFERENCE_MODES), len(REFERENCE_MODES))
