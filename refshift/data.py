"""Phase 2 DL data loader: braindecode + MOABB preprocessing with disk cache.

Pipeline:
    MOABBDataset
    -> pick_types EEG
    -> [Schirrmeister: pick 44-channel motor subset]
    -> V -> uV
    -> resample to common rate (default 250 Hz)
    -> bandpass l_freq..h_freq
    -> [optional pre_ems_reference: applied to filtered raw, before EMS]
    -> exponential_moving_standardize
    -> create_windows_from_events
    -> (X, y, metadata) tensors

Reference operators in the standard pipeline are applied to the windowed X
*after* this function returns. EMS is per-channel and adaptive, so it does
not commute with channel-mixing reference operators: CAR(EMS(X)) != EMS(CAR(X))
in general. The pre_ems_reference argument flips the order to "filter then
reference then EMS" for the EMS-control ablation. Fix in v0.14: the reference
operator is now applied AFTER the bandpass filter (it was earlier applied to
the broadband raw, which mattered for the non-linear median operator).

The disk cache keys on all parameters that affect the preprocessed output
(see _CACHE_KEY_PARAMS). Reference operators are post-cache, so all 6
reference variants share one cache entry per (subject, preprocess_params).
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


_DATASET_ID_TO_MOABB = {
    "iv2a": "BNCI2014_001",
    "openbmi": "Lee2019_MI",
    "cho2017": "Cho2017",
    "dreyer2023": "Dreyer2023",
    "schirrmeister2017": "Schirrmeister2017",
}


def _moabb_code(dataset_id: str) -> str:
    key = dataset_id.lower()
    if key not in _DATASET_ID_TO_MOABB:
        raise ValueError(
            f"Unknown dataset_id: {dataset_id!r}. "
            f"Known: {tuple(_DATASET_ID_TO_MOABB)}"
        )
    return _DATASET_ID_TO_MOABB[key]


def _scale_volts_to_microvolts(data):
    """Multiply by 1e6. Module-level so braindecode's Preprocessor can pickle it."""
    return data * 1e6


# ---------------------------------------------------------------------------
# Disk cache for preprocessed (X, y, metadata, sfreq, ch_names) tensors
# ---------------------------------------------------------------------------

_CACHE_KEY_PARAMS = (
    "dataset_id", "subject", "resample", "l_freq", "h_freq",
    "ems_factor_new", "ems_init_block_size",
    "trial_start_offset_s", "trial_stop_offset_s",
    "pre_ems_reference",
    "pre_ems_laplacian_k", "pre_ems_montage",
    "classes",
)


def _cache_path(cache_dir: str, params: dict) -> str:
    """<cache_dir>/<dataset_id>/sub-<NNN>/<hash>.npz"""
    relevant = {k: params[k] for k in _CACHE_KEY_PARAMS}
    key = hashlib.sha1(
        json.dumps(relevant, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    subdir = os.path.join(
        cache_dir, params["dataset_id"], f"sub-{int(params['subject']):03d}",
    )
    os.makedirs(subdir, exist_ok=True)
    return os.path.join(subdir, f"{key}.npz")


def load_dl_data(
    dataset_id: str,
    subject: int,
    *,
    resample: float = 250.0,
    l_freq: float = 8.0,
    h_freq: float = 32.0,
    ems_factor_new: float = 1e-3,
    ems_init_block_size: int = 1000,
    trial_start_offset_s: float = 0.0,
    trial_stop_offset_s: float = 0.0,
    preload: bool = True,
    n_jobs: int = 1,
    cache_dir: Optional[str] = None,
    pre_ems_reference: Optional[str] = None,
    pre_ems_laplacian_k: int = 4,
    pre_ems_montage: str = "standard_1005",
    classes: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, float, List[str]]:
    """Load one subject's MI data through the canonical preprocess pipeline.

    pre_ems_reference, pre_ems_laplacian_k, pre_ems_montage are only used when
    pre_ems_reference is not None. They control the graph built inside
    _apply_pre_ems_ref and are part of the cache key, so distinct settings
    cache to distinct entries.

    classes=None loads the dataset's full class set. classes=(c1, c2, ...) keeps
    only trials whose label is in the tuple and re-indexes y to 0..len(classes)-1
    in the order the classes appear in the tuple. classes is part of the cache
    key so 2-class and 4-class entries don't collide.

    The integer<->class mapping is built explicitly from
    dataset_full_classes(dataset_id) regardless of which class subset is kept,
    so per-subject windowing always assigns the same integer to the same class.

    Returns
    -------
    X : (N, C, T) float32
    y : (N,) int64
    metadata : DataFrame with columns ['session', 'run', 'subject']
    sfreq : float (= resample)
    ch_names : EEG channel order matching X's axis 1
    """
    from refshift.experiments._datasets import (
        dataset_full_classes,
        resolve_classes,
    )

    kept_classes = resolve_classes(dataset_id, classes)
    full_classes = dataset_full_classes(dataset_id)
    # mapping for braindecode: every dataset-defined class -> a stable int.
    # We then filter to kept_classes and re-index to 0..len(kept_classes)-1.
    full_mapping = {name: i for i, name in enumerate(full_classes)}
    kept_int_set = {full_mapping[c] for c in kept_classes}
    # Stable re-index: kept class at position i in kept_classes -> i
    full_int_to_new_int = {
        full_mapping[name]: i for i, name in enumerate(kept_classes)
    }

    params = {
        "dataset_id": str(dataset_id).lower(),
        "subject": int(subject),
        "resample": float(resample),
        "l_freq": float(l_freq),
        "h_freq": float(h_freq),
        "ems_factor_new": float(ems_factor_new),
        "ems_init_block_size": int(ems_init_block_size),
        "trial_start_offset_s": float(trial_start_offset_s),
        "trial_stop_offset_s": float(trial_stop_offset_s),
        "pre_ems_reference": str(pre_ems_reference) if pre_ems_reference else None,
        "pre_ems_laplacian_k": int(pre_ems_laplacian_k),
        "pre_ems_montage": str(pre_ems_montage),
        "classes": ",".join(kept_classes),
    }

    cache_path = _cache_path(cache_dir, params) if cache_dir is not None else None
    if cache_path is not None and os.path.exists(cache_path):
        try:
            npz = np.load(cache_path, allow_pickle=True)
            metadata = pd.DataFrame({
                "session": npz["metadata_session"],
                "run": npz["metadata_run"],
                "subject": npz["metadata_subject"],
            })
            return (npz["X"], npz["y"], metadata,
                    float(npz["sfreq"].item()), list(npz["ch_names"]))
        except Exception:
            pass  # corrupt cache: fall through and overwrite

    # Cache miss: run the full preprocess.
    from refshift.compat import make_braindecode_dataset
    from braindecode.preprocessing import (
        Preprocessor,
        create_windows_from_events,
        exponential_moving_standardize,
        preprocess,
    )

    dataset = make_braindecode_dataset(dataset_id, int(subject))

    preprocessors = [
        Preprocessor("pick_types", eeg=True, meg=False, stim=False),
    ]

    # Schirrmeister: 44-channel motor subset (paper Section 2.7.1, Cz excluded
    # as it served as recording reference). ordered=True is critical: the
    # neighbour graph is built from paradigm.channels in user-supplied order
    # via _get_eeg_channel_names; ordered=False would return raw-channel order
    # and break graph alignment.
    if dataset_id == "schirrmeister2017":
        from refshift.experiments._datasets import SCHIRRMEISTER_MOTOR_CHANNELS
        preprocessors.append(Preprocessor(
            "pick_channels",
            ch_names=list(SCHIRRMEISTER_MOTOR_CHANNELS),
            ordered=True,
        ))

    preprocessors.extend([
        Preprocessor(_scale_volts_to_microvolts, apply_on_array=True),
        Preprocessor("resample", sfreq=float(resample)),
        Preprocessor("filter", l_freq=l_freq, h_freq=h_freq),
    ])

    # Pre-EMS reference: applied to the filtered raw, before EMS. Order matters
    # for the median operator (non-linear). For linear operators (CAR, REST,
    # Laplacian, CSD, cz_ref) the result is identical regardless of where in
    # the linear-filtering chain the reference sits.
    if pre_ems_reference is not None:
        from refshift.reference import (
            _GRAPH_MODES,
            _resolve_alias,
            apply_reference,
            build_graph,
        )

        # Resolve legacy 'laplacian' alias if present in pre_ems_reference.
        canonical_pre_ems = _resolve_alias(pre_ems_reference)

        def _apply_pre_ems_ref(raw):
            ch_names = list(raw.info["ch_names"])
            needs_graph = canonical_pre_ems in _GRAPH_MODES
            graph = build_graph(
                ch_names,
                k=int(pre_ems_laplacian_k),
                montage=str(pre_ems_montage),
                include_rest=(canonical_pre_ems == "rest"),
                include_csd=(canonical_pre_ems == "csd"),
            ) if needs_graph else None
            data = raw.get_data()  # (C, T_total)
            # apply_reference takes (N, C, T); add and remove the singleton.
            new_data = apply_reference(data[None, :, :], canonical_pre_ems, graph=graph)[0]
            raw._data[:] = new_data.astype(raw._data.dtype, copy=False)

        preprocessors.append(Preprocessor(_apply_pre_ems_ref, apply_on_array=False))

    preprocessors.append(Preprocessor(
        exponential_moving_standardize,
        factor_new=ems_factor_new,
        init_block_size=ems_init_block_size,
    ))

    preprocess(dataset, preprocessors, n_jobs=n_jobs)

    sfreq = float(dataset.datasets[0].raw.info["sfreq"])
    for ds in dataset.datasets:
        if ds.raw.info["sfreq"] != sfreq:
            raise RuntimeError(
                f"Inconsistent sfreq across runs for subject {subject}: "
                f"{sfreq} vs {ds.raw.info['sfreq']}"
            )
    ch_names = list(dataset.datasets[0].raw.info["ch_names"])

    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=int(round(sfreq * trial_start_offset_s)),
        trial_stop_offset_samples=int(round(sfreq * trial_stop_offset_s)),
        preload=preload,
        mapping=full_mapping,
    )

    Xs: List[np.ndarray] = []
    ys: List[int] = []
    rows: List[dict] = []
    for ds_wind in windows_dataset.datasets:
        desc = ds_wind.description
        sess = str(desc["session"]) if "session" in desc else "0"
        run = str(desc["run"]) if "run" in desc else "0"
        subj = int(desc["subject"]) if "subject" in desc else int(subject)
        for i in range(len(ds_wind)):
            x, y, _ind = ds_wind[i]
            y_int = int(y)
            if y_int not in kept_int_set:
                # Outside the requested class subset; drop.
                continue
            Xs.append(np.asarray(x, dtype=np.float32))
            ys.append(full_int_to_new_int[y_int])
            rows.append({"session": sess, "run": run, "subject": subj})

    if not Xs:
        raise RuntimeError(
            f"No windows extracted for {dataset_id} subject {subject} "
            f"with classes={kept_classes}. "
            "Check MOABB cache / Kaggle symlinks / trial offset values."
        )

    X = np.stack(Xs).astype(np.float32, copy=False)
    y = np.array(ys, dtype=np.int64)
    metadata = pd.DataFrame(rows)

    if cache_path is not None:
        try:
            np.savez(
                cache_path[:-len(".npz")],  # np.savez auto-appends .npz
                X=X, y=y, sfreq=np.float64(sfreq),
                metadata_session=metadata["session"].to_numpy(),
                metadata_run=metadata["run"].to_numpy(),
                metadata_subject=metadata["subject"].to_numpy(),
                ch_names=np.asarray(ch_names, dtype=object),
            )
        except OSError:
            pass

    return X, y, metadata, sfreq, ch_names
