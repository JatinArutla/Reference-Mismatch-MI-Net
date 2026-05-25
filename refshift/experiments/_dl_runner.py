"""Shared scaffolding for the DL runners.

Three of the four DL runners (mismatch DL branch, mismatch_jitter,
bandpass_mismatch train side) share the same setup: resolve dataset, build
the neighbour graph if any reference mode in the run needs one, iterate over
(subject, seed) jobs with subject-level data caching, split per iteration.

run_pre_ems_diagonal does NOT use this scaffolding because each (subject,
seed, ref) cell triggers its own load_dl_data call (the reference is part
of the preprocessing for that ablation).
"""

from __future__ import annotations

from typing import List, NamedTuple, Optional

from refshift.experiments._datasets import (
    get_eeg_channel_names,
    resolve_dataset,
)
from refshift.reference import _GRAPH_MODES, build_graph


class DLRunContext(NamedTuple):
    dataset_id: str
    dataset_code: str
    subjects: List[int]
    seeds: List[int]
    graph: Optional["DatasetGraph"]  # type: ignore[name-defined]


def setup_dl_run(
    dataset_id: str,
    *,
    subjects: Optional[List[int]],
    seeds: List[int],
    reference_modes_for_graph: tuple,
    laplacian_k: int = 4,
    k_large_skip: int = 4,
    k_large_use: int = 4,
    montage: str = "standard_1005",
    progress: bool = True,
) -> DLRunContext:
    """Resolve dataset and build the neighbour graph if any mode needs one.

    The graph is built iff any of the declared modes is in _GRAPH_MODES.
    REST is included only when 'rest' is among them (the spherical-model
    forward solution is the slow part). CSD is included only when 'csd' is
    among them. lap_large uses k_large_skip / k_large_use parameters
    (defaults match build_graph: 4/4).
    """
    from refshift.reference import _resolve_alias

    dataset, paradigm = resolve_dataset(dataset_id)
    if subjects is None:
        subjects = list(dataset.subject_list)

    resolved_modes = tuple(_resolve_alias(m) for m in reference_modes_for_graph)
    needs_graph = any(m in _GRAPH_MODES for m in resolved_modes)
    needs_rest = "rest" in resolved_modes
    needs_csd = "csd" in resolved_modes
    graph = None
    if needs_graph:
        ch_names = get_eeg_channel_names(
            dataset, subject=subjects[0], paradigm=paradigm,
        )
        graph = build_graph(
            ch_names, k_small=laplacian_k,
            k_large_skip=k_large_skip, k_large_use=k_large_use,
            montage=montage,
            include_rest=needs_rest, include_csd=needs_csd,
        )
        if progress:
            cz_msg = (
                f", cz_idx={graph.cz_idx}" if graph.cz_idx is not None
                else ", cz_idx=None (no Cz channel)"
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

    return DLRunContext(
        dataset_id=dataset_id,
        dataset_code=dataset.code,
        subjects=list(subjects),
        seeds=list(seeds),
        graph=graph,
    )


def iter_per_subject_dl_jobs(
    ctx: DLRunContext,
    *,
    split_strategy: str = "auto",
    desc: str = "",
    progress: bool = True,
    normalization: str = "zscore",
    dl_resample: float = 250.0,
    dl_l_freq: float = 8.0,
    dl_h_freq: float = 32.0,
    dl_trial_start_offset_s: float = 0.0,
    dl_trial_stop_offset_s: float = 0.0,
    dl_cache_dir: Optional[str] = None,
    classes: Optional[tuple] = None,
):
    """Yield (subject, seed, X_tr, y_tr, X_te, y_te, sfreq) per job.

    The underlying load_dl_data call only fires when the subject changes;
    seeds for the same subject reuse the in-memory tensor. The channel-order
    assertion against ctx.graph runs once per subject reload.

    classes=None loads all classes; pass a tuple like ('left_hand','right_hand')
    to restrict to a 2-class subset (handled inside load_dl_data, see its
    docstring for the int<->class mapping rules).
    """
    from refshift.data import load_dl_data
    from refshift.experiments._split import split_train_test

    try:
        from tqdm.auto import tqdm as _tqdm
    except ImportError:
        def _tqdm(it, **kwargs):
            return it

    jobs = [(s, k) for s in ctx.subjects for k in ctx.seeds]
    iterator = _tqdm(
        jobs, desc=desc or f"[{ctx.dataset_code}]",
        disable=not progress, leave=True,
    )

    last_subject: Optional[int] = None
    X = y_int = metadata = None
    sfreq: Optional[float] = None

    for subject, seed in iterator:
        if subject != last_subject:
            X, y_int, metadata, sfreq, ch_names_subj = load_dl_data(
                ctx.dataset_id, subject,
                resample=dl_resample,
                l_freq=dl_l_freq, h_freq=dl_h_freq,
                normalization=normalization,
                trial_start_offset_s=dl_trial_start_offset_s,
                trial_stop_offset_s=dl_trial_stop_offset_s,
                cache_dir=dl_cache_dir,
                classes=classes,
            )
            if ctx.graph is not None:
                assert list(ch_names_subj) == ctx.graph.ch_names, (
                    f"Channel order mismatch for subject {subject}: "
                    f"data={ch_names_subj[:5]}... graph={ctx.graph.ch_names[:5]}..."
                )
            last_subject = subject

        X_tr, y_tr, X_te, y_te = split_train_test(
            X, y_int, metadata,
            strategy=split_strategy, seed=seed, dataset_id=ctx.dataset_id,
        )
        yield subject, seed, X_tr, y_tr, X_te, y_te, sfreq


def free_cuda():
    """Best-effort CUDA cache release between trainings."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
