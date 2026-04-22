"""Calibration against MOABB's published CSP+LDA benchmark on BCI IV-2a.

Target: 65.99 +/- 15.47 (MOABB published; 4-class, WithinSession, 9 subjects).

This script runs two conditions:

    1. Bare MOABB canonical pipeline: Covariances(oas) -> CSP(6) -> LDA(svd).
       Must reproduce MOABB's published number to within ~2%.

    2. Same pipeline wrapped with ReferenceTransformer('native').
       This is an identity transformer (except for a fresh copy). Accuracy
       must match condition 1 to within ~0.5% (floating-point noise).

If (1) matches MOABB and (2) matches (1), the reference transformer is
correctly integrated into MOABB's pipeline and we can trust downstream
mismatch-matrix numbers from refshift.mismatch.

Usage:
    python scripts/calibrate_csp_lda.py
    python scripts/calibrate_csp_lda.py --subjects 1 2 3    # quick check

Prerequisites:
    pip install -r requirements.txt
    # MOABB 1.5.0 is pinned; other versions may drift from 65.99.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from refshift.pipelines import make_csp_lda_pipeline


def run_calibration(subjects, random_state: int = 42):
    """Return (mean, std, per_subject_df) tuple for both conditions."""
    from moabb.datasets import BNCI2014_001
    from moabb.evaluations import WithinSessionEvaluation
    from moabb.paradigms import MotorImagery

    paradigm = MotorImagery(n_classes=4)  # defaults: 8-32 Hz, no resample, all EEG
    dataset = BNCI2014_001()
    if subjects is not None:
        dataset.subject_list = list(subjects)

    pipelines = {
        "CSP+LDA (bare)":
            make_csp_lda_pipeline(reference_mode=None),
        "CSP+LDA (ReferenceTransformer='native')":
            make_csp_lda_pipeline(reference_mode="native"),
    }

    evaluation = WithinSessionEvaluation(
        paradigm=paradigm,
        datasets=[dataset],
        overwrite=True,
        random_state=random_state,
    )
    results = evaluation.process(pipelines)
    return results


def _format_summary(results) -> str:
    lines = []
    for pipe_name, group in results.groupby("pipeline"):
        mean = 100 * group["score"].mean()
        std = 100 * group["score"].std()
        lines.append(f"  {pipe_name:42s}  {mean:5.2f} +/- {std:5.2f}")
    return "\n".join(lines)


def _check_targets(results, verbose: bool = True) -> int:
    """Return 0 on success, nonzero on failure."""
    bare = results[results["pipeline"] == "CSP+LDA (bare)"]["score"]
    with_native = results[
        results["pipeline"] == "CSP+LDA (ReferenceTransformer='native')"
    ]["score"]
    bare_mean = 100 * bare.mean()
    with_native_mean = 100 * with_native.mean()

    # Target 1: bare CSP+LDA within 2% of MOABB's 65.99%
    moabb_target = 65.99
    tol_moabb = 2.0
    moabb_ok = abs(bare_mean - moabb_target) <= tol_moabb

    # Target 2: identity-transformer result matches bare within 0.5%
    tol_identity = 0.5
    identity_ok = abs(with_native_mean - bare_mean) <= tol_identity

    if verbose:
        print()
        print(f"Target 1 (MOABB 65.99% +/- {tol_moabb}%): "
              f"got {bare_mean:.2f}% --> {'PASS' if moabb_ok else 'FAIL'}")
        print(f"Target 2 (identity transformer within {tol_identity}% "
              f"of bare): delta={with_native_mean - bare_mean:+.2f}% --> "
              f"{'PASS' if identity_ok else 'FAIL'}")

    return 0 if (moabb_ok and identity_ok) else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subjects", type=int, nargs="+", default=None,
                   help="Subset of subjects (default: all 9).")
    p.add_argument("--random-state", type=int, default=42)
    args = p.parse_args(argv)

    print("CSP+LDA calibration on BCI IV-2a (BNCI2014_001), WithinSession.")
    print(f"Subjects: {args.subjects or 'all 9'}   random_state={args.random_state}")

    results = run_calibration(args.subjects, random_state=args.random_state)

    print()
    print("Per-pipeline summary (mean +/- std across subjects x sessions):")
    print(_format_summary(results))
    return _check_targets(results)


if __name__ == "__main__":
    sys.exit(main())
