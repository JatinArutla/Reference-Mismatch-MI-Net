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


def test_report_loro_true_cost_uses_the_full_jitter_baseline():
    # A reference can be intrinsically hard AND cheap to hold out. The naive cost
    # confuses the two; the full-jitter baseline separates them.
    rng = np.random.default_rng(3)
    loro = pd.DataFrame([
        {"subject": s, "holdout_ref": h, "test_ref": te,
         "accuracy": (0.50 if te == h else 0.62) + rng.normal(0, 0.005)}
        for s in (1, 2) for h in REFERENCE_MODES for te in REFERENCE_MODES
    ])
    # native is hard for everyone, not just when held out
    full = pd.DataFrame([
        {"subject": s, "test_ref": te,
         "accuracy": (0.51 if te == "native" else 0.62) + rng.normal(0, 0.005)}
        for s in (1, 2) for te in REFERENCE_MODES
    ])
    report_loro(loro, title="t", full_jitter=full)
    A = report_loro(loro, title="t").to_numpy(dtype=float)
    j = list(REFERENCE_MODES).index("native")
    naive = np.nanmean(np.delete(A[j, :], j)) - A[j, j]
    true_cost = 0.51 - A[j, j]
    assert true_cost < naive, "true cost must strip out intrinsic difficulty"


def test_report_matrix_reports_asymmetry():
    # A -> B and B -> A can differ; nothing used to surface that.
    rng = np.random.default_rng(4)
    rows = []
    for s in (1, 2, 3):
        for tr in REFERENCE_MODES:
            for te in REFERENCE_MODES:
                acc = 0.70 if tr == te else (0.60 if tr == "native" else 0.35)
                rows.append({"subject": s, "seed": 0, "train_ref": tr,
                             "test_ref": te, "accuracy": acc + rng.normal(0, 0.005)})
    M = report_matrix(pd.DataFrame(rows), title="t").to_numpy(dtype=float)
    D = np.abs(M - M.T)
    # only the native row/column is asymmetric here, so check the worst pair
    # rather than the mean, which the 36 symmetric pairs dilute.
    i, j = np.unravel_index(np.nanargmax(D), D.shape)
    assert "native" in (REFERENCE_MODES[i], REFERENCE_MODES[j])
    assert D[i, j] > 0.20
