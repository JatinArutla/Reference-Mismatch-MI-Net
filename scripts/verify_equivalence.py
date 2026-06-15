"""Verify the lean refshift reproduces the original repo's numbers.

Run this ONCE on a machine that has both repos installed and the IV-2a data
available (e.g. your Kaggle kernel). It checks two things:

  1. PREPROCESSING EQUIVALENCE
     The lean preprocess.load_windows should produce the same windowed trials
     as the original data.load_dl_data (zscore path). The reference operators
     are byte-identical by construction (proved by the unit tests), so if the
     preprocessed input matches, every downstream deep-learning number matches
     up to GPU/seed nondeterminism.

  2. RESULT-CSV DIFF (optional)
     If you point it at an old result CSV and a freshly produced lean CSV for
     the same experiment, it reports the maximum per-cell accuracy difference.

Usage (in a notebook or shell, with both packages importable):

    from verify_equivalence import check_preprocessing, diff_result_csvs
    for ds in ("iv2a", "openbmi", "cho2017", "dreyer2023", "schirrmeister2017"):
        check_preprocessing(ds, subject=1)
    diff_result_csvs("old/iv2a_csp_lda.csv", "lean/iv2a_csp_lda.csv")

This script lives outside the package on purpose: it depends on BOTH the old
and new code being importable, which is only true during the migration.

NOTE: call ``setup_kaggle_env()`` (from either repo) once before running these
checks, so MOABB finds the symlinked Kaggle datasets instead of downloading.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def check_preprocessing(dataset_id: str = "iv2a", subject: int = 1,
                        tol: float = 1e-4) -> bool:
    """Compare lean load_windows against the original load_dl_data (zscore).

    Prints shapes, label/channel agreement, and the max abs difference in X.
    Returns True if everything matches within ``tol``.
    """
    # Original pipeline.
    from refshift.data import load_dl_data
    Xo, yo, _mo, so, cho = load_dl_data(
        dataset_id, subject, normalization="zscore", cache_dir=None,
    )
    # Lean pipeline. Import under its installed name; adjust if you installed
    # the lean package under a different distribution name.
    from refshift.preprocess import load_windows
    Xn, yn, _mn, sn, chn = load_windows(dataset_id, subject)

    shapes_ok = Xo.shape == Xn.shape
    labels_ok = np.array_equal(yo, yn)
    chans_ok = list(cho) == list(chn)
    sfreq_ok = float(so) == float(sn)
    max_diff = (
        float(np.abs(Xo.astype(np.float64) - Xn.astype(np.float64)).max())
        if shapes_ok else float("inf")
    )

    print("=== preprocessing equivalence (%s subject %d) ===" % (dataset_id, subject))
    print("  shapes      old=%s lean=%s  %s" % (Xo.shape, Xn.shape, "OK" if shapes_ok else "MISMATCH"))
    print("  labels      %s" % ("OK" if labels_ok else "MISMATCH"))
    print("  channels    %s" % ("OK" if chans_ok else "MISMATCH"))
    print("  sfreq       old=%s lean=%s  %s" % (so, sn, "OK" if sfreq_ok else "MISMATCH"))
    print("  max |dX|    %.3e  %s" % (max_diff, "OK" if max_diff <= tol else "TOO LARGE"))
    print("  trials/class old=%s lean=%s" % (np.bincount(yo).tolist(), np.bincount(yn).tolist()))

    passed = shapes_ok and labels_ok and chans_ok and sfreq_ok and max_diff <= tol
    print("  RESULT: %s" % ("PASS" if passed else "FAIL"))
    return passed


def diff_result_csvs(old_csv: str, lean_csv: str, *, key=("train_ref", "test_ref"),
                     value: str = "accuracy", tol: float = 0.02) -> bool:
    """Diff two result CSVs cell-by-cell on the mean of ``value``.

    Aggregates each CSV to a matrix keyed by ``key`` and reports the maximum
    absolute difference. For CSP+LDA (deterministic) expect ~0; for deep nets
    expect differences within seed/GPU noise (a few percent). Returns True if
    within ``tol``.
    """
    old = pd.read_csv(old_csv)
    lean = pd.read_csv(lean_csv)
    key = list(key)
    Mo = old.groupby(key)[value].mean()
    Ml = lean.groupby(key)[value].mean()
    common = Mo.index.intersection(Ml.index)
    if len(common) == 0:
        print("No overlapping keys between the two CSVs.")
        return False
    diff = (Mo.loc[common] - Ml.loc[common]).abs()
    max_diff = float(diff.max())
    print("=== result CSV diff (%s) ===" % value)
    print("  cells compared : %d" % len(common))
    print("  max |diff|     : %.4f" % max_diff)
    print("  mean |diff|    : %.4f" % float(diff.mean()))
    passed = max_diff <= tol
    print("  RESULT: %s (tol=%.3f)" % ("PASS" if passed else "FAIL", tol))
    return passed
