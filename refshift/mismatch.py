"""6x6 mismatch-matrix runner for CSP+LDA.

Design
------
We do NOT use MOABB's ``WithinSessionEvaluation`` / ``CrossSessionEvaluation``
for the mismatch matrix. Those evaluators train once and score once per fold;
they do not support "train once per train_ref, score six times per train_ref"
without writing wrapper code that reaches into their private ``_fit_cv`` and
``_build_scored_result`` internals.

Instead we call ``paradigm.get_data(dataset, subjects=[sub])`` directly.
That function carries all the preprocessing we care about (SetRawAnnotations
-> bandpass on Raw -> epoching -> resample -> scaling). From there we
split train/test ourselves, train one CSP+LDA per train_ref, then score
each fitted classifier against all six test_refs by applying the operator
to X_test on the fly.

The sanity check (calibrate_csp_lda.py) uses MOABB's full
WithinSessionEvaluation to replicate the published 65.99% benchmark, so
the pipeline components are independently validated against MOABB.
Once calibration passes, this runner is the efficient path for the matrix.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score

from refshift.data import (
    encode_labels,
    get_eeg_channel_names,
    load_paradigm_data,
    split_train_test,
)
from refshift.pipelines import make_csp_lda_pipeline
from refshift.reference import REFERENCE_MODES, apply_reference, build_graph


def run_mismatch_matrix(
    paradigm,
    dataset,
    *,
    subjects: Optional[List[int]] = None,
    reference_modes: tuple = REFERENCE_MODES,
    seeds: List[int] = (0,),
    split_strategy: str = "auto",
    n_filters: int = 6,
    laplacian_k: int = 4,
    montage: str = "standard_1005",
    verbose: bool = False,
) -> pd.DataFrame:
    """Run the 6x6 mismatch matrix for CSP+LDA on a MOABB dataset.

    For each (subject, seed):
      1. Load epoched data once via paradigm.get_data.
      2. Split into train/test (session split if >1 session, else 80/20).
      3. Pre-compute all six test-set reference variants (cheap ndarray ops).
      4. For each train_ref: apply operator to X_train, fit CSP+LDA, then
         score the fitted classifier on each of the six pre-computed test
         variants.

    CSP+LDA is deterministic given data and the CSP implementation in
    pyriemann; the ``seed`` loop only matters for the stratified 80/20
    case (Cho2017 / Dreyer2023), where it controls the shuffle.

    Parameters
    ----------
    paradigm : moabb.paradigms.base.BaseParadigm
        Already-configured paradigm (e.g. ``MotorImagery(n_classes=4)`` for
        IV-2a or ``LeftRightImagery()`` for 2-class datasets).
    dataset : moabb.datasets.base.BaseDataset
    subjects : list of int or None
        Defaults to ``dataset.subject_list``.
    reference_modes : iterable of str
        Reference modes used for both train and test. Defaults to all six.
    seeds : list of int
        Seeds for the stratified-split case. Ignored for session-split.
    split_strategy : {'auto','session','stratify'}
    n_filters : int
        CSP filters. MOABB default is 6.
    laplacian_k : int
        Neighbors for the Laplacian operator. Default 4.
    montage : str
        MNE standard montage name used to build the neighbor graph.

    Returns
    -------
    pandas.DataFrame
        Long-form with columns: dataset, subject, seed, train_ref,
        test_ref, accuracy, kappa, n_train, n_test.
    """
    modes = tuple(reference_modes)
    if subjects is None:
        subjects = list(dataset.subject_list)

    needs_graph = any(m in ("laplacian", "bipolar") for m in modes)
    graph = None
    if needs_graph:
        ch_names = get_eeg_channel_names(dataset)
        graph = build_graph(ch_names, k=laplacian_k, montage=montage)

    rows = []
    for subject in subjects:
        if verbose:
            print(f"[{dataset.code}] subject {subject}: loading...")
        X, y_raw, metadata = load_paradigm_data(paradigm, dataset, subject)
        y_int, _ = encode_labels(y_raw)

        for seed in seeds:
            X_tr, y_tr, X_te, y_te = split_train_test(
                X, y_int, metadata,
                strategy=split_strategy, seed=seed,
            )

            # Precompute all six test variants once — each operator is O(N*C*T).
            X_te_by_ref = {
                m: apply_reference(X_te, m, graph=graph) for m in modes
            }

            for train_ref in modes:
                X_tr_ref = apply_reference(X_tr, train_ref, graph=graph)
                pipe = make_csp_lda_pipeline(
                    reference_mode=None,  # already applied above; keep pipeline bare
                    n_filters=n_filters,
                )
                pipe.fit(X_tr_ref, y_tr)

                for test_ref in modes:
                    y_pred = pipe.predict(X_te_by_ref[test_ref])
                    rows.append({
                        "dataset":   dataset.code,
                        "subject":   subject,
                        "seed":      seed,
                        "train_ref": train_ref,
                        "test_ref":  test_ref,
                        "accuracy":  float(accuracy_score(y_te, y_pred)),
                        "kappa":     float(cohen_kappa_score(y_te, y_pred)),
                        "n_train":   int(len(y_tr)),
                        "n_test":    int(len(y_te)),
                    })

            if verbose:
                done_cells = len(modes) ** 2
                acc_so_far = np.mean([r["accuracy"] for r in rows[-done_cells:]])
                print(
                    f"[{dataset.code}] subject {subject} seed {seed} done "
                    f"(mean over {done_cells} cells: {acc_so_far:.3f})"
                )

    return pd.DataFrame(rows)


def mismatch_matrix(
    df: pd.DataFrame,
    *,
    metric: str = "accuracy",
    aggregate: str = "mean",
) -> pd.DataFrame:
    """Pivot a long-form result DataFrame into a train_ref x test_ref table.

    Aggregates across (subject, seed) within each (train_ref, test_ref)
    cell. ``aggregate`` is one of {'mean', 'std'}.
    """
    grouped = df.groupby(["train_ref", "test_ref"])[metric]
    if aggregate == "mean":
        return grouped.mean().unstack("test_ref")
    if aggregate == "std":
        return grouped.std().unstack("test_ref")
    raise ValueError(f"Unknown aggregate: {aggregate!r}")
