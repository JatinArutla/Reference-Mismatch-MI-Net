"""Full NxN mismatch matrix with reference applied BEFORE EMS.

The standard run_mismatch pipeline applies the reference operator AFTER
per-channel exponential-moving standardisation (EMS). Because EMS is
adaptive and per-channel, it does not commute with channel-mixing reference
operators: the standardised view of CSD(X) is NOT the same as CSD applied
to EMS-standardised X. This is a real preprocessing-order confound: the
mismatch matrix observed in run_mismatch reflects both the operator topology
AND the operator's interaction with EMS.

run_pre_ems_diagonal addresses this by running the diagonal cells (train_ref
== test_ref) under the operator-before-EMS pipeline, but that only verifies
whether each operator's WITHIN-DISTRIBUTION accuracy is robust to the order.
It does NOT answer the cross-operator question: under the
operator-before-EMS pipeline, does training under CAR still fail to transfer
to a CSD test set?

run_pre_ems_mismatch fills that gap. For each (train_ref, test_ref) cell:

  1. Load (subject, train_ref) via load_dl_data(pre_ems_reference=train_ref)
     -- reference is applied to the filtered raw before per-channel EMS.
  2. Train the DL model on the train half of that load.
  3. For each test_ref:
     a. Load (subject, test_ref) the same way.
     b. Use the test half (or matched run) as the test set.
     c. Predict.

The cache hits hard here: each (subject, ref) load is reused across N rows
of the matrix. Total loads per (subject, seed): N (not N**2).

Important caveats:
  - Only DL models. CSP+LDA does not use EMS, so the EMS-order question
    does not apply to it; CSP+LDA cross-reference results come from run_mismatch.
  - The split policy for train/test must produce identical splits across
    different reference settings; we rely on the deterministic seed in
    split_train_test for this.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score

from refshift.experiments._datasets import (
    get_eeg_channel_names,
    resolve_dataset,
)
from refshift.experiments._dl_runner import free_cuda
from refshift.experiments._split import split_train_test
from refshift.reference import (
    REFERENCE_MODES,
    _GRAPH_MODES,
    build_graph,
    canonical_mode_tuple,
    reference_modes_for_dataset,
    validate_reference_modes,
)


def run_pre_ems_mismatch(
    dataset_id: str,
    *,
    model: str = "shallow",
    subjects: Optional[List[int]] = None,
    seeds: Sequence[int] = (0,),
    reference_modes: Optional[Sequence[str]] = None,
    classes: Optional[Sequence[str]] = None,
    split_strategy: str = "auto",
    normalization: str = "ems",
    laplacian_k: int = 4,
    k_large_skip: int = 4,
    k_large_use: int = 4,
    montage: str = "standard_1005",
    progress: bool = True,
    dl_max_epochs: int = 200,
    dl_batch_size: int = 32,
    dl_lr: Optional[float] = None,
    dl_weight_decay: float = 0.0,
    dl_device: Optional[str] = None,
    dl_verbose: int = 0,
    dl_l_freq: float = 8.0,
    dl_h_freq: float = 32.0,
    dl_resample: float = 250.0,
    dl_trial_start_offset_s: float = 0.0,
    dl_trial_stop_offset_s: float = 0.0,
    dl_cache_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Full NxN mismatch matrix with reference applied BEFORE EMS.

    DL-only; CSP+LDA does not use EMS, so the EMS-order question does not
    apply. Returns the same long-form DataFrame shape as run_mismatch, with
    columns dataset, subject, seed, train_ref, test_ref, accuracy, kappa,
    n_train, n_test, plus a "pipeline" column = "pre_ems_mismatch" so the
    result can be concatenated with run_mismatch output for comparison.

    For each (subject, seed) the workflow is:

      Step 1: for each reference r in modes, call load_dl_data with
              pre_ems_reference=r. This produces (X_r, y, metadata). The
              cache key includes pre_ems_reference so each ref gets its
              own preprocessed entry. Each load applies r to the
              filtered raw, then EMS, then windows.

      Step 2: split each (X_r, y, metadata) deterministically into train/test
              using the same seed; X_tr_r, y_tr, X_te_r, y_te. Crucially,
              the split policy is independent of the reference (it depends
              only on metadata), so y_tr and y_te are the same across refs.

      Step 3: for each train_ref: fit a fresh DL model on (X_tr_{train_ref},
              y_tr). For each test_ref: predict on X_te_{test_ref}; report
              accuracy and kappa.

    Compute scaling: with N refs and S subjects and K seeds, this requires
    N*S subject loads (cached after the first call) and N*S*K model trainings.
    For the default 8-mode set, 5 subjects, 3 seeds: 40 loads (one-time) and
    120 trainings. Comparable cost to run_mismatch (which does N*S*K
    trainings for the same matrix, but only S*K loads).

    Use this runner alongside run_mismatch to disentangle:
      - "is the operator topology driving the cross-reference failure?"  -- this runner
      - "is the operator-EMS interaction driving it?"                    -- standard run_mismatch

    If the family structure of the mismatch matrix is similar under both
    pipelines, the v0.14/v0.15 results are not preprocessing-order artefacts.
    If they differ substantially, the headline finding needs the
    pre-EMS pipeline as the primary report.

    normalization in {"zscore", "ems", "none"}; default "ems" (NOT "zscore"
    like run_mismatch). This is the EMS-ordering control, so it defaults to the
    normalizer whose ordering it tests. An ordering comparison against
    run_mismatch is only valid when both are run under the SAME normalization;
    set this and run_mismatch's normalization to matching values explicitly.
    """
    from refshift.data import load_dl_data
    from refshift.model import SUPPORTED_DL_MODELS, make_dl_model

    model_lc = model.lower()
    if model_lc == "csp_lda":
        raise ValueError(
            "run_pre_ems_mismatch is DL-only. CSP+LDA does not use EMS, so "
            "the EMS-order question doesn't apply; use run_mismatch instead."
        )
    if model_lc not in SUPPORTED_DL_MODELS:
        raise ValueError(
            f"Unknown DL model {model!r}; expected one of {SUPPORTED_DL_MODELS}"
        )

    from refshift.data import NORMALIZATIONS
    if normalization not in NORMALIZATIONS:
        raise ValueError(f"normalization={normalization!r} not in {NORMALIZATIONS}")

    if reference_modes is None:
        modes = reference_modes_for_dataset(dataset_id)
    else:
        modes = canonical_mode_tuple(reference_modes)

    dataset, paradigm = resolve_dataset(dataset_id)
    if subjects is None:
        subjects = list(dataset.subject_list)
    seeds_list = list(seeds)

    # Probe graph for early validation; not used for any actual preprocessing
    # (load_dl_data builds its own graph internally for each pre_ems_reference).
    needs_graph = any(m in _GRAPH_MODES for m in modes)
    probe_graph = None
    if needs_graph:
        ch_names = get_eeg_channel_names(
            dataset, subject=subjects[0], paradigm=paradigm,
        )
        probe_graph = build_graph(
            ch_names, k_small=laplacian_k,
            k_large_skip=k_large_skip, k_large_use=k_large_use,
            montage=montage,
            include_rest=("rest" in modes),
            include_csd=("csd" in modes),
        )
    validate_reference_modes(modes, probe_graph, dataset_id=dataset_id)

    try:
        from tqdm.auto import tqdm as _tqdm
    except ImportError:
        def _tqdm(it, **kwargs):
            return it

    # Outer loop: (subject, seed). Inner: train_ref, then test_ref.
    job_pairs = [(s, k) for s in subjects for k in seeds_list]
    iterator = _tqdm(
        job_pairs, desc=f"[{dataset.code}] {model_lc} pre-EMS mismatch",
        disable=not progress, leave=True,
    )

    rows: List[dict] = []
    for subject, seed in iterator:
        # Step 1: load preprocessed (X, y, metadata) per reference for this subject.
        # The cache key includes pre_ems_reference so loads are independent
        # entries on disk. They share dataset / subject / sfreq / l_freq / h_freq.
        per_ref_data: dict = {}
        for ref in modes:
            X_full, y_full, metadata, sfreq, ch_names_subj = load_dl_data(
                dataset_id, subject,
                resample=dl_resample,
                l_freq=dl_l_freq, h_freq=dl_h_freq,
                normalization=normalization,
                trial_start_offset_s=dl_trial_start_offset_s,
                trial_stop_offset_s=dl_trial_stop_offset_s,
                cache_dir=dl_cache_dir,
                pre_ems_reference=ref,
                pre_ems_laplacian_k=laplacian_k,
                pre_ems_k_large_skip=k_large_skip,
                pre_ems_k_large_use=k_large_use,
                pre_ems_montage=montage,
                classes=classes,
            )
            per_ref_data[ref] = (X_full, y_full, metadata, sfreq, ch_names_subj)

        # Step 2: split each into train/test. Splits depend on (metadata, seed,
        # dataset_id) only, so they are identical across refs by construction.
        # We verify this with an assertion below.
        per_ref_split: dict = {}
        ref0 = modes[0]
        X0, y0, meta0, sfreq0, _ = per_ref_data[ref0]
        X0_tr, y0_tr, X0_te, y0_te = split_train_test(
            X0, y0, meta0, strategy=split_strategy, seed=seed,
            dataset_id=dataset_id,
        )
        for ref in modes:
            Xr, yr, meta_r, sfreq_r, _ = per_ref_data[ref]
            Xr_tr, yr_tr, Xr_te, yr_te = split_train_test(
                Xr, yr, meta_r, strategy=split_strategy, seed=seed,
                dataset_id=dataset_id,
            )
            # Sanity: y_tr / y_te must match across refs (split is metadata-driven).
            # If this fails, the pre-EMS pipeline is producing different trial
            # counts per reference, which would invalidate the matrix.
            assert np.array_equal(yr_tr, y0_tr), (
                f"split mismatch: train labels differ between {ref0!r} and {ref!r} "
                f"for subject {subject}, seed {seed}. Reference operators should "
                f"not change trial counts."
            )
            assert np.array_equal(yr_te, y0_te), (
                f"split mismatch: test labels differ between {ref0!r} and {ref!r} "
                f"for subject {subject}, seed {seed}."
            )
            per_ref_split[ref] = (Xr_tr, yr_tr, Xr_te, yr_te, sfreq_r)

        # Step 3: for each train_ref, fit a fresh model on its X_tr; score on
        # every test_ref's X_te. y_tr and y_te are the same across refs (just
        # asserted), so we can use the canonical ones from ref0.
        y_tr = y0_tr
        y_te = y0_te
        n_classes = int(max(int(y_tr.max()), int(y_te.max()))) + 1

        for train_ref in modes:
            X_tr_ref, _, _, _, sfreq_tr = per_ref_split[train_ref]

            net = make_dl_model(
                model=model_lc,
                n_channels=X_tr_ref.shape[1],
                n_classes=n_classes,
                n_times=X_tr_ref.shape[2],
                sfreq=float(sfreq_tr),
                seed=int(seed),
                max_epochs=dl_max_epochs,
                batch_size=dl_batch_size,
                lr=dl_lr,
                weight_decay=dl_weight_decay,
                device=dl_device,
                verbose=dl_verbose,
            )
            net.fit(
                X_tr_ref.astype(np.float32, copy=False),
                y_tr.astype(np.int64, copy=False),
            )

            for test_ref in modes:
                _, _, X_te_ref, _, _ = per_ref_split[test_ref]
                y_pred = net.predict(X_te_ref.astype(np.float32, copy=False))
                rows.append({
                    "dataset":   dataset.code,
                    "subject":   int(subject),
                    "seed":      int(seed),
                    "pipeline":  "pre_ems_mismatch",
                    "train_ref": train_ref,
                    "test_ref":  test_ref,
                    "accuracy":  float(accuracy_score(y_te, y_pred)),
                    "kappa":     float(cohen_kappa_score(y_te, y_pred)),
                    "n_train":   int(len(y_tr)),
                    "n_test":    int(len(y_te)),
                })

            del net
            free_cuda()

        # Free preprocessed tensors before next subject to keep memory bounded.
        del per_ref_data, per_ref_split

    return pd.DataFrame(rows)
