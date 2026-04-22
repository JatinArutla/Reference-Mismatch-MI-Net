"""Run the 6x6 CSP+LDA mismatch matrix on BCI IV-2a.

Output: one CSV of per-cell results and two printed 6x6 tables
(mean accuracy, std across subjects).

This script uses CrossSession evaluation: each subject's first session
becomes the train set, the second session becomes the test set. That
matches the handoff's choice for IV-2a/OpenBMI (session-split datasets).

Target behavior (structural, not exact):
    - Diagonal cells:   moderately high
    - Off-diagonal cells within global-mean family (native/car/median/gs):
                         close to diagonal
    - Off-diagonal cells crossing global-mean -> spatial-differential
                         (laplacian/bipolar): noticeably lower
    - Bipolar column:    the most depressed off-diagonal column

The handoff's v1 target for CSP+LDA IV-2a diagonal mean was ~0.45-0.55
with bipolar column off-diagonals near chance (0.25). Those numbers were
obtained under v1's from-scratch pipeline. Under MOABB's filter-on-raw
they should lift somewhat; the structural pattern is what matters.

Usage:
    python scripts/run_mismatch_iv2a_csp_lda.py                 # all 9 subjects
    python scripts/run_mismatch_iv2a_csp_lda.py --subjects 1 3  # quick
    python scripts/run_mismatch_iv2a_csp_lda.py --out results/iv2a_csp.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from refshift.mismatch import mismatch_matrix, run_mismatch_matrix


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subjects", type=int, nargs="+", default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--out", type=Path, default=Path("results/iv2a_csp_lda.csv"))
    args = p.parse_args(argv)

    from moabb.datasets import BNCI2014_001
    from moabb.paradigms import MotorImagery

    paradigm = MotorImagery(n_classes=4)
    dataset = BNCI2014_001()

    df = run_mismatch_matrix(
        paradigm=paradigm,
        dataset=dataset,
        subjects=args.subjects,
        seeds=args.seeds,
        split_strategy="auto",   # IV-2a has 2 sessions, so -> cross-session
        verbose=True,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print()
    print(f"Saved per-cell results to {args.out}  ({len(df)} rows)")
    print()
    print("Mean accuracy (train_ref rows, test_ref cols):")
    mean_tbl = mismatch_matrix(df, metric="accuracy", aggregate="mean")
    with pd.option_context("display.precision", 3, "display.width", 120):
        print(mean_tbl.round(3))
    print()
    print("Std accuracy (across subjects x seeds):")
    std_tbl = mismatch_matrix(df, metric="accuracy", aggregate="std")
    with pd.option_context("display.precision", 3, "display.width", 120):
        print(std_tbl.round(3))

    # A minimal structural sanity check.
    n = mean_tbl.shape[0]
    diag_mask = np.eye(n, dtype=bool)
    diag_mean = mean_tbl.values[diag_mask].mean()
    off_mean = mean_tbl.values[~diag_mask].mean()
    print()
    print(f"Diagonal mean:     {diag_mean:.3f}")
    print(f"Off-diagonal mean: {off_mean:.3f}")
    print(f"Gap (diag - off):  {diag_mean - off_mean:+.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
