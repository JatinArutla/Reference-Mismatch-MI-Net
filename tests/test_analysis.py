"""Reporting helpers, on synthetic result tables."""

import numpy as np
import pandas as pd

from refshift.analysis import (
    mismatch_matrix,
    report_families,
    report_jitter_full,
    report_lofo,
    report_loro,
    report_matrix,
    transfer_gap_ci,
)
from refshift.references import FAMILIES, REFERENCE_MODES


def _mismatch_frame(matched=0.70, mismatched=0.50, seeds=(0, 1)):
    rng = np.random.default_rng(0)
    return pd.DataFrame([
        {"subject": s, "seed": k, "train_ref": tr, "test_ref": te,
         "accuracy": (matched if tr == te else mismatched) + rng.normal(0, 0.01)}
        for s in (1, 2, 3) for k in seeds
        for tr in REFERENCE_MODES for te in REFERENCE_MODES
    ])


def test_mismatch_matrix_shape():
    M = mismatch_matrix(_mismatch_frame())
    assert M.shape == (len(REFERENCE_MODES), len(REFERENCE_MODES))


def test_report_matrix_recovers_the_gap():
    M = report_matrix(_mismatch_frame(), title="t")
    A = M.to_numpy(dtype=float)
    assert np.diag(A).mean() > A[~np.eye(A.shape[0], dtype=bool)].mean()


def test_transfer_gap_averages_seeds_within_subject():
    # Seeds are repeated runs of the same subject, so three subjects x two
    # seeds must give n_subjects=3, not 6, and recover the 20pp gap.
    r = transfer_gap_ci(_mismatch_frame(matched=0.70, mismatched=0.50))
    assert r["n_subjects"] == 3
    assert 19.0 < r["mean"] < 21.0
    assert r["lo"] <= r["mean"] <= r["hi"]


def test_report_families_and_jitter_run():
    df = _mismatch_frame()
    assert report_families(df, title="t").shape[0] == len(REFERENCE_MODES)
    jit = df.rename(columns={"train_ref": "unused"})[["subject", "test_ref", "accuracy"]]
    assert len(report_jitter_full(jit, title="t")) == len(REFERENCE_MODES)


def test_report_loro_diagonal_is_the_holdout():
    rng = np.random.default_rng(1)
    df = pd.DataFrame([
        {"subject": s, "holdout_ref": h, "test_ref": te,
         "accuracy": (0.50 if te == h else 0.65) + rng.normal(0, 0.01)}
        for s in (1, 2) for h in REFERENCE_MODES for te in REFERENCE_MODES
    ])
    A = report_loro(df, title="t").to_numpy(dtype=float)
    assert np.diag(A).mean() < A[~np.eye(A.shape[0], dtype=bool)].mean()


def test_report_lofo_family_matrix():
    mode_fam = {m: f for f, ms in FAMILIES.items() for m in ms}
    rng = np.random.default_rng(2)
    df = pd.DataFrame([
        {"subject": s, "holdout_family": hf, "test_family": mode_fam[te],
         "test_ref": te, "accuracy": (0.45 if mode_fam[te] == hf else 0.60)
         + rng.normal(0, 0.01)}
        for s in (1, 2) for hf in FAMILIES for te in mode_fam
    ])
    M = report_lofo(df, title="t")
    assert M.shape == (3, 3)
    assert list(M.index) == ["global", "single", "spatial"]
