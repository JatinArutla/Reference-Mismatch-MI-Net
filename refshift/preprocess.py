"""Turn a raw subject into windowed trials ready for a model.

This is the whole deep-learning preprocessing pipeline, in order:

    1. pick EEG channels            (drop EOG / stim)
    2. [fixed channel subset]       (Schirrmeister only: 44 motor channels)
    3. volts -> microvolts          (MOABB returns volts; models expect uV)
    4. resample                     (to the dataset's rate, default 250 Hz)
    5. bandpass 8-32 Hz             (the motor-imagery mu/beta band)
    6. per-channel z-score          (centre and scale each channel over time)
    7. cut into trials              (one window per cue, labelled 0..K-1)

Two ordering choices matter:

  * Filtering happens BEFORE the reference operator. References (CAR, Laplacian,
    ...) are applied later, to the windowed trials, so one preprocessed copy
    serves every reference.
  * Standardisation is per-channel and does NOT commute with channel-mixing
    references. We z-score here, then reference afterwards, identically for
    every reference, which is what makes the mismatch matrix a fair comparison.

The preprocessed (X, y, metadata, sfreq, ch_names) tuple is cached to disk per
(dataset, subject, pipeline-params), so repeated runs and multiple seeds reuse
it instead of re-fetching and re-filtering.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from refshift.datasets import (
    classes_for,
    make_braindecode_dataset,
    spec,
    split_strategy_for,
)

# Default pipeline constants. resample is overridden per dataset by its spec.
BANDPASS_LOW_HZ: float = 8.0
BANDPASS_HIGH_HZ: float = 32.0


def _volts_to_microvolts(data):
    """Scale V -> uV. Module-level (not a lambda) so braindecode can pickle it."""
    return data * 1e6


def _zscore_per_channel(data, eps: float = 1e-7):
    """Centre and scale each channel over time: (x - mean) / std, per channel.

    ``data`` is one continuous recording (channels, time). ``eps`` floors the
    divisor so a flat channel maps to zeros instead of NaN. Module-level so it
    pickles for braindecode's Preprocessor.
    """
    data = np.asarray(data, dtype=np.float64)
    mean = data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, keepdims=True)
    return (data - mean) / np.maximum(std, eps)


# ---------------------------------------------------------------------------
# Disk cache for the preprocessed tensors
# ---------------------------------------------------------------------------

def _cache_path(cache_dir: str, dataset_id: str, subject: int, params: dict) -> str:
    """<cache_dir>/<dataset_id>/sub-<NNN>/<hash>.npz"""
    key = hashlib.sha1(
        json.dumps(params, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    subdir = os.path.join(cache_dir, dataset_id, f"sub-{int(subject):03d}")
    os.makedirs(subdir, exist_ok=True)
    return os.path.join(subdir, f"{key}.npz")


def _load_cache(path: str):
    try:
        npz = np.load(path, allow_pickle=True)
        metadata = pd.DataFrame({
            "session": npz["meta_session"],
            "run": npz["meta_run"],
            "subject": npz["meta_subject"],
        })
        return (npz["X"], npz["y"], metadata,
                float(npz["sfreq"].item()), list(npz["ch_names"]))
    except Exception:
        return None  # corrupt or missing: caller recomputes


def _save_cache(path: str, X, y, metadata, sfreq, ch_names):
    try:
        np.savez(
            path[:-len(".npz")],  # np.savez appends .npz
            X=X, y=y, sfreq=np.float64(sfreq),
            meta_session=metadata["session"].to_numpy(),
            meta_run=metadata["run"].to_numpy(),
            meta_subject=metadata["subject"].to_numpy(),
            ch_names=np.asarray(ch_names, dtype=object),
        )
    except OSError:
        pass  # cache is best-effort


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def load_windows(
    dataset_id: str,
    subject: int,
    *,
    l_freq: float = BANDPASS_LOW_HZ,
    h_freq: float = BANDPASS_HIGH_HZ,
    cache_dir: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, float, List[str]]:
    """Preprocess one subject and return windowed trials.

    Returns
    -------
    X : (n_trials, n_channels, n_times) float32, z-scored and bandpassed but
        NOT yet referenced.
    y : (n_trials,) int64, labels 0..K-1 in the dataset's canonical class order.
    metadata : DataFrame with 'session', 'run', 'subject' columns.
    sfreq : float, sampling rate after resampling.
    ch_names : channel names matching axis 1 of X (graph must use this order).
    """
    s = spec(dataset_id)
    classes = classes_for(dataset_id)
    resample_hz = float(s.resample_hz)

    params = {
        "dataset_id": dataset_id.lower(), "subject": int(subject),
        "resample": resample_hz, "l_freq": float(l_freq), "h_freq": float(h_freq),
        "channels": list(s.channels) if s.channels else None,
        "classes": list(classes),
    }
    path = _cache_path(cache_dir, dataset_id, subject, params) if cache_dir else None
    if path and os.path.exists(path):
        cached = _load_cache(path)
        if cached is not None:
            return cached

    from braindecode.preprocessing import (
        Preprocessor,
        create_windows_from_events,
        preprocess,
    )

    dataset = make_braindecode_dataset(dataset_id, subject)

    preprocessors = [Preprocessor("pick_types", eeg=True, meg=False, stim=False)]
    # Fixed channel subset (Schirrmeister motor channels). ordered=True keeps
    # the user-supplied order so the neighbour graph aligns with X's channels.
    if s.channels:
        preprocessors.append(Preprocessor(
            "pick_channels", ch_names=list(s.channels), ordered=True,
        ))
    preprocessors.extend([
        Preprocessor(_volts_to_microvolts, apply_on_array=False),
        Preprocessor("resample", sfreq=resample_hz),
        Preprocessor("filter", l_freq=l_freq, h_freq=h_freq),
        Preprocessor(_zscore_per_channel, apply_on_array=False),
    ])
    preprocess(dataset, preprocessors, n_jobs=1)

    sfreq = float(dataset.datasets[0].raw.info["sfreq"])
    for ds in dataset.datasets:
        if ds.raw.info["sfreq"] != sfreq:
            raise RuntimeError(
                f"Inconsistent sfreq across runs for {dataset_id} subject "
                f"{subject}: {sfreq} vs {ds.raw.info['sfreq']}"
            )
    ch_names = list(dataset.datasets[0].raw.info["ch_names"])

    # Map each class name to a stable integer in canonical order.
    mapping = {name: i for i, name in enumerate(classes)}
    windows = create_windows_from_events(
        dataset,
        trial_start_offset_samples=0,
        trial_stop_offset_samples=0,
        preload=True,
        mapping=mapping,
    )

    Xs: List[np.ndarray] = []
    ys: List[int] = []
    rows: List[dict] = []
    for ds_wind in windows.datasets:
        desc = ds_wind.description
        sess = str(desc["session"]) if "session" in desc else "0"
        run = str(desc["run"]) if "run" in desc else "0"
        for i in range(len(ds_wind)):
            x, label, _crop = ds_wind[i]
            Xs.append(np.asarray(x, dtype=np.float32))
            ys.append(int(label))
            rows.append({"session": sess, "run": run, "subject": int(subject)})

    if not Xs:
        raise RuntimeError(
            f"No windows extracted for {dataset_id} subject {subject}. "
            "Check the MOABB download / cache."
        )

    X = np.stack(Xs).astype(np.float32, copy=False)
    y = np.array(ys, dtype=np.int64)
    metadata = pd.DataFrame(rows)

    if path:
        _save_cache(path, X, y, metadata, sfreq, ch_names)
    return X, y, metadata, sfreq, ch_names


def split_train_test(
    X: np.ndarray, y: np.ndarray, metadata: pd.DataFrame, dataset_id: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split one subject into train/test using the dataset's strategy.

    'session' datasets (iv2a, openbmi, cho2017, dreyer2023): first session is
    train, the rest test. 'run' datasets (schirrmeister2017): first run
    (e.g. '0train') is train, the rest ('1test') test.
    """
    strategy = split_strategy_for(dataset_id)

    if strategy == "session":
        sessions = sorted(metadata["session"].unique())
        if len(sessions) < 2:
            raise RuntimeError(
                f"{dataset_id}: expected >=2 sessions for a session split, "
                f"found {sessions}."
            )
        train_mask = (metadata["session"] == sessions[0]).to_numpy()
    elif strategy == "run":
        runs = sorted(metadata["run"].unique())
        if len(runs) < 2:
            raise RuntimeError(
                f"{dataset_id}: expected >=2 runs for a run split, found {runs}."
            )
        train_mask = (metadata["run"] == runs[0]).to_numpy()
    else:
        raise ValueError(f"Unknown split strategy {strategy!r} for {dataset_id}")

    return X[train_mask], y[train_mask], X[~train_mask], y[~train_mask]
