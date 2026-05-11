"""MOABB calibration: verifies bare CSP+LDA matches the published 65.99% on
IV-2a, and that prepending ReferenceTransformer('native') is a true identity.

The identity check (within 0.5%) is the cheapest end-to-end correctness test
for the reference-transformer wiring: any accidental side effect (dtype change,
contiguity, non-clone-ability) would break this and corrupt every cell of the
mismatch matrix.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import pandas as pd

from refshift.experiments._datasets import resolve_dataset
from refshift.model import make_csp_lda_pipeline


IV2A_CSP_LDA_TARGET = 65.99
IV2A_CSP_LDA_TOL = 2.0
IDENTITY_TOL = 0.5


def calibrate_csp_lda(
    dataset_id: str = "iv2a",
    *,
    subjects: Optional[List[int]] = None,
    random_state: int = 42,
    overwrite: bool = True,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, bool]:
    """Run MOABB WithinSessionEvaluation on bare and identity-prefixed CSP+LDA.

    Returns (per-fold results, summary table, passed bool). For iv2a, passed
    requires both the MOABB target and the identity-equivalence check; for
    other datasets, only identity-equivalence.
    """
    from moabb.evaluations import WithinSessionEvaluation

    dataset, paradigm = resolve_dataset(dataset_id)
    if subjects is not None:
        dataset.subject_list = list(subjects)

    pipelines = {
        "CSP+LDA (bare)": make_csp_lda_pipeline(reference_mode=None),
        "CSP+LDA (ReferenceTransformer='native')":
            make_csp_lda_pipeline(reference_mode="native"),
    }

    evaluation = WithinSessionEvaluation(
        paradigm=paradigm, datasets=[dataset],
        overwrite=overwrite, random_state=random_state,
    )
    results = evaluation.process(pipelines)

    summary = (
        results.groupby("pipeline")["score"]
               .agg(["mean", "std", "count"])
               .assign(mean=lambda d: 100 * d["mean"],
                       std=lambda d: 100 * d["std"])
               .round(2)
    )

    bare_mean = 100 * results[results["pipeline"] == "CSP+LDA (bare)"]["score"].mean()
    ident_mean = 100 * results[
        results["pipeline"] == "CSP+LDA (ReferenceTransformer='native')"
    ]["score"].mean()

    moabb_ok = (
        abs(bare_mean - IV2A_CSP_LDA_TARGET) <= IV2A_CSP_LDA_TOL
        if dataset_id.lower() == "iv2a" else True
    )
    identity_ok = abs(ident_mean - bare_mean) <= IDENTITY_TOL
    passed = bool(moabb_ok and identity_ok)

    if verbose:
        print()
        print("Per-pipeline summary (mean +/- std across subjects x sessions):")
        for name, row in summary.iterrows():
            print(f"  {name:42s}  {row['mean']:5.2f} +/- {row['std']:5.2f}")
        print()
        if dataset_id.lower() == "iv2a":
            print(
                f"Target 1 (MOABB {IV2A_CSP_LDA_TARGET}% +/- {IV2A_CSP_LDA_TOL}%): "
                f"got {bare_mean:.2f}% --> {'PASS' if moabb_ok else 'FAIL'}"
            )
        print(
            f"Target 2 (identity within {IDENTITY_TOL}%): "
            f"delta={ident_mean - bare_mean:+.2f}% --> "
            f"{'PASS' if identity_ok else 'FAIL'}"
        )

    return results, summary, passed
