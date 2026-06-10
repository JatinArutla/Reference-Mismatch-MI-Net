"""NxN train-test reference mismatch matrix.

For each (subject, seed):
  1. Load epoched data (CSP+LDA: paradigm.get_data; DL: load_dl_data via braindecode).
  2. Train/test split.
  3. Pre-compute all N test variants once.
  4. For each train_ref: train one model on apply_reference(X_tr, train_ref);
     score on each of the N test variants.

N is determined by the resolved reference_modes (8 by default, 7 for
Schirrmeister2017 which drops cz_ref). The CSP+LDA and DL branches share
the inner train-once-evaluate-many loop via _train_and_evaluate.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score

from refshift.experiments._datasets import (
    build_cache_config,
    get_eeg_channel_names,
    resolve_dataset,
)
from refshift.experiments._dl_runner import (
    free_cuda,
    iter_per_subject_dl_jobs,
    setup_dl_run,
)
from refshift.experiments._split import encode_labels, split_train_test
from refshift.model import make_csp_lda_pipeline
from refshift.reference import (
    REFERENCE_MODES,
    _GRAPH_MODES,
    apply_reference,
    apply_reference_then_ea,
    build_graph,
    canonical_mode_tuple,
    reference_modes_for_dataset,
    validate_reference_modes,
)


def mismatch_matrix(
    df: pd.DataFrame,
    *,
    metric: str = "accuracy",
    aggregate: str = "mean",
) -> pd.DataFrame:
    """Pivot long-form results to a train_ref x test_ref table.

    Legacy 'laplacian' rows from v0.14 CSVs are resolved to 'lap_small'
    before pivoting, so old CSVs analyse correctly under v0.15.
    """
    from refshift.reference import _resolve_alias

    df = df.copy()
    for col in ("train_ref", "test_ref"):
        if col in df.columns:
            df[col] = df[col].map(lambda m: _resolve_alias(m) if isinstance(m, str) else m)
    grouped = df.groupby(["train_ref", "test_ref"])[metric]
    if aggregate == "mean":
        return grouped.mean().unstack("test_ref")
    if aggregate == "std":
        return grouped.std().unstack("test_ref")
    raise ValueError(f"Unknown aggregate: {aggregate!r}")


def _train_and_evaluate(
    *,
    fit_pipe,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te_by_ref: dict,
    y_te: np.ndarray,
    modes: tuple,
    row_template: dict,
    rows: list,
    n_train: int,
    n_test: int,
):
    """For each train_ref, fit a fresh pipeline and score on every test_ref.

    fit_pipe is a callable train_ref -> fitted estimator. This indirection
    lets the CSP+LDA and DL branches share this inner loop while still
    constructing different model factories.
    """
    for train_ref in modes:
        pipe = fit_pipe(train_ref)
        for test_ref in modes:
            y_pred = pipe.predict(X_te_by_ref[test_ref])
            rows.append({
                **row_template,
                "train_ref": train_ref,
                "test_ref":  test_ref,
                "accuracy":  float(accuracy_score(y_te, y_pred)),
                "kappa":     float(cohen_kappa_score(y_te, y_pred)),
                "n_train":   n_train,
                "n_test":    n_test,
            })
        del pipe
        free_cuda()


def run_mismatch(
    dataset_id: str,
    *,
    model: str = "csp_lda",
    subjects: Optional[List[int]] = None,
    seeds: List[int] = (0,),
    reference_modes: Optional[Sequence[str]] = None,
    classes: Optional[Sequence[str]] = None,
    split_strategy: str = "auto",
    normalization: str = "zscore",
    apply_ea: bool = False,
    ea_eps: float = 1e-12,
    n_filters: int = 6,
    csp_trace_normalize: bool = False,
    laplacian_k: int = 4,
    k_large_skip: int = 4,
    k_large_use: int = 4,
    montage: str = "standard_1005",
    cache: bool = True,
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
    """Run the NxN train-test reference mismatch matrix on one dataset.

    model in {'csp_lda', 'eegnet', 'shallow'}. classes is threaded through
    both the CSP+LDA and DL paths (DL plumbing added in v0.14.2). subjects=None
    uses the dataset's full subject list with known-bad subjects excluded.

    reference_modes accepts any iterable (set, tuple, list, frozenset).
    Set in your notebook to restrict the matrix to a subset:

        REFERENCES = {"native", "car", "csd"}
        df = run_mismatch("iv2a", model="csp_lda", reference_modes=REFERENCES)

    The output matrix is always laid out in canonical REFERENCE_MODES order
    regardless of input iteration order. The legacy name 'laplacian' is
    accepted as an alias for 'lap_small'.

    Spatial-derivative parameters:
        laplacian_k:   number of nearest neighbours for lap_small (alias k_small);
                       default 4.
        k_large_skip:  number of nearest neighbours to skip for lap_large; default 4.
        k_large_use:   number of neighbours used by lap_large; default 4.
        With defaults (k_small=4, skip=4, use=4) lap_small uses ranks 0..3 and
        lap_large uses ranks 4..7 -- disjoint neighbour sets for every channel.

    CSP+LDA-only parameters:
        csp_trace_normalize: insert a per-trial trace normalisation step
            between Covariances(oas) and CSP. Set to True for the scale-control
            ablation that addresses the CSD-amplitude-scale confound. Default
            False matches v0.14/v0.15 behaviour (MOABB CSP.yml).

    normalization in {"zscore", "ems", "none"}; default "zscore". DL-ONLY:
    selects the per-channel standardisation of the continuous filtered raw,
    applied before windowing and the reference operator.
        "zscore" static per-channel z-score over the recording.
        "ems"    adaptive exponential_moving_standardize (the pre-v0.17 default).
        "none"   no standardisation.
    The CSP+LDA path ignores this argument entirely: it has no per-channel
    standardisation by design (covariance-based, calibrated against MOABB's
    CSP.yml), so `run_mismatch("iv2a")` reproduces the classical pipeline
    regardless of the normalization default.

    apply_ea (default False): if True, apply Euclidean Alignment (He & Wu 2020,
    MIRepNet-exact form) to each (subject, split) block of trials AFTER the
    reference operator and before the model. EA whitens per-subject second-order
    statistics to the identity. Applies on BOTH the CSP+LDA and DL paths (unlike
    `normalization`, which is DL-only and per-channel). The reference->EA ordering
    answers "does EA absorb the reference-induced shift": the whitening is fit on
    the already-referenced trials. EA is fit per split (train and test get
    independently estimated reference covariances), matching MIRepNet's own
    behaviour. ea_eps is a small diagonal ridge guarding the inverse-sqrt against
    rank-deficient reference covariances (cz_ref, Laplacians zero/reduce rank).
    """
    model_lc = model.lower()
    from refshift.data import NORMALIZATIONS
    if normalization not in NORMALIZATIONS:
        raise ValueError(f"normalization={normalization!r} not in {NORMALIZATIONS}")
    if model_lc == "csp_lda":
        is_dl = False
    else:
        from refshift.model import SUPPORTED_DL_MODELS
        if model_lc not in SUPPORTED_DL_MODELS:
            raise NotImplementedError(
                f"model={model!r} not supported. Known: 'csp_lda', {SUPPORTED_DL_MODELS}"
            )
        is_dl = True

    if reference_modes is None:
        modes = reference_modes_for_dataset(dataset_id)
    else:
        modes = canonical_mode_tuple(reference_modes)
    rows: List[dict] = []

    if is_dl:
        from refshift.model import make_dl_model

        ctx = setup_dl_run(
            dataset_id, subjects=subjects, seeds=seeds,
            reference_modes_for_graph=modes,
            laplacian_k=laplacian_k,
            k_large_skip=k_large_skip, k_large_use=k_large_use,
            montage=montage, progress=progress,
        )
        validate_reference_modes(modes, ctx.graph, dataset_id=dataset_id)
        for subject, seed, X_tr, y_tr, X_te, y_te, sfreq in iter_per_subject_dl_jobs(
            ctx, split_strategy=split_strategy,
            desc=f"[{ctx.dataset_code}] {model_lc} mismatch",
            progress=progress,
            normalization=normalization,
            dl_resample=dl_resample,
            dl_l_freq=dl_l_freq, dl_h_freq=dl_h_freq,
            dl_trial_start_offset_s=dl_trial_start_offset_s,
            dl_trial_stop_offset_s=dl_trial_stop_offset_s,
            dl_cache_dir=dl_cache_dir,
            classes=classes,
        ):
            X_te_by_ref = {
                m: apply_reference_then_ea(
                    X_te, m, graph=ctx.graph, apply_ea=apply_ea, ea_eps=ea_eps
                )
                for m in modes
            }
            n_classes = int(max(int(y_tr.max()), int(y_te.max()))) + 1

            def _fit_dl(train_ref):
                X_tr_ref = apply_reference_then_ea(
                    X_tr, train_ref, graph=ctx.graph, apply_ea=apply_ea, ea_eps=ea_eps
                )
                pipe = make_dl_model(
                    model=model_lc,
                    n_channels=X_tr_ref.shape[1],
                    n_classes=n_classes,
                    n_times=X_tr_ref.shape[2],
                    sfreq=float(sfreq),
                    seed=int(seed),
                    max_epochs=dl_max_epochs,
                    batch_size=dl_batch_size,
                    lr=dl_lr,
                    weight_decay=dl_weight_decay,
                    device=dl_device,
                    verbose=dl_verbose,
                )
                pipe.fit(X_tr_ref, y_tr)
                return pipe

            _train_and_evaluate(
                fit_pipe=_fit_dl,
                X_tr=X_tr, y_tr=y_tr, X_te_by_ref=X_te_by_ref, y_te=y_te,
                modes=modes,
                row_template={"dataset": ctx.dataset_code, "subject": subject, "seed": seed},
                rows=rows,
                n_train=int(len(y_tr)), n_test=int(len(y_te)),
            )

        return pd.DataFrame(rows)

    # CSP+LDA path
    dataset, paradigm = resolve_dataset(dataset_id, classes=classes)
    if subjects is None:
        subjects = list(dataset.subject_list)
    seeds = list(seeds)

    needs_graph = any(m in _GRAPH_MODES for m in modes)
    needs_rest = "rest" in modes
    needs_csd = "csd" in modes
    graph = None
    if needs_graph:
        ch_names = get_eeg_channel_names(dataset, subject=subjects[0], paradigm=paradigm)
        graph = build_graph(
            ch_names, k_small=laplacian_k,
            k_large_skip=k_large_skip, k_large_use=k_large_use,
            montage=montage,
            include_rest=needs_rest, include_csd=needs_csd,
        )
        if progress:
            cz_msg = (
                f", cz_idx={graph.cz_idx}" if graph.cz_idx is not None
                else ", cz_idx=None"
            )
            rest_msg = (
                f", REST cond={graph.rest_cond:.2e}"
                if graph.rest_cond is not None else ""
            )
            csd_msg = (
                f", CSD cond={graph.csd_cond:.2e}"
                if graph.csd_cond is not None else ""
            )
            print(
                f"[{dataset.code}] graph: C={len(graph.ch_names)}"
                f"{cz_msg}{rest_msg}{csd_msg}"
            )
    validate_reference_modes(modes, graph, dataset_id=dataset_id)

    cache_kwargs = {"cache_config": build_cache_config()} if cache else {}

    try:
        from tqdm.auto import tqdm as _tqdm
    except ImportError:
        def _tqdm(it, **kwargs):
            return it

    jobs = [(s, k) for s in subjects for k in seeds]
    iterator = _tqdm(
        jobs, desc=f"[{dataset.code}] {model_lc} mismatch",
        disable=not progress, leave=True,
    )

    last_subject: Optional[int] = None
    X = y_int = metadata = None

    for subject, seed in iterator:
        if subject != last_subject:
            X, y_raw, metadata = paradigm.get_data(
                dataset=dataset, subjects=[subject], **cache_kwargs,
            )
            y_int, _ = encode_labels(y_raw)
            if graph is not None:
                # paradigm.get_data returns X with channel axis matching
                # paradigm.channels (when set) or the underlying raw's EEG order.
                # The graph was built from the same source, so order must match.
                # Count + order: order check catches MOABB-version drift that
                # would otherwise corrupt every cell of the mismatch matrix.
                ch_names_subj = get_eeg_channel_names(
                    dataset, subject=subject, paradigm=paradigm,
                )
                assert list(ch_names_subj) == graph.ch_names, (
                    f"Channel order mismatch for subject {subject}: "
                    f"data first 5={ch_names_subj[:5]}, "
                    f"graph first 5={graph.ch_names[:5]}"
                )
                assert X.shape[1] == len(graph.ch_names), (
                    f"Channel count mismatch for subject {subject}: "
                    f"data={X.shape[1]}, graph={len(graph.ch_names)}"
                )
            last_subject = subject

        X_tr, y_tr, X_te, y_te = split_train_test(
            X, y_int, metadata, strategy=split_strategy, seed=seed,
            dataset_id=dataset_id,
        )
        X_te_by_ref = {
            m: apply_reference_then_ea(
                X_te, m, graph=graph, apply_ea=apply_ea, ea_eps=ea_eps
            )
            for m in modes
        }

        def _fit_csp(train_ref):
            X_tr_ref = apply_reference_then_ea(
                X_tr, train_ref, graph=graph, apply_ea=apply_ea, ea_eps=ea_eps
            )
            pipe = make_csp_lda_pipeline(
                reference_mode=None, n_filters=n_filters,
                trace_normalize=csp_trace_normalize,
            )
            pipe.fit(X_tr_ref, y_tr)
            return pipe

        _train_and_evaluate(
            fit_pipe=_fit_csp,
            X_tr=X_tr, y_tr=y_tr, X_te_by_ref=X_te_by_ref, y_te=y_te,
            modes=modes,
            row_template={"dataset": dataset.code, "subject": subject, "seed": seed},
            rows=rows,
            n_train=int(len(y_tr)), n_test=int(len(y_te)),
        )

    return pd.DataFrame(rows)
