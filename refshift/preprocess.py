"""Turn a raw subject into windowed trials, and split them into train/test.

The pipeline, in order:

    1. pick EEG channels        (drop EOG / EMG / stim)
    2. [channel subset]         (Schirrmeister only: 44 motor channels)
    3. volts -> microvolts      (MOABB stores volts; the models expect uV)
    4. resample to 250 Hz
    5. bandpass 8-32 Hz         (the motor-imagery mu/beta band)
    6. cut one window per cue, labelled 0..K-1

Two ordering choices matter:

  * Filtering happens BEFORE the reference operator, so one preprocessed copy
    serves every reference. Operators are applied later, to the windows.
  * Per-channel z-scoring happens AFTER the reference, in the runner. The
    operator therefore acts on raw voltage, which is how a real re-reference is
    applied. The windows returned here are NOT z-scored and NOT referenced.

Results are cached to disk per subject, so extra seeds reuse them.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from refshift.datasets import classes_for, make_braindecode_dataset, spec

BANDPASS_LOW_HZ = 8.0
BANDPASS_HIGH_HZ = 32.0


def _volts_to_microvolts(data):
    """Module-level (not a lambda) so braindecode can pickle it."""
    return data * 1e6


def _cache_path(cache_dir: str, dataset_id: str, subject: int, params: dict) -> str:
    key = hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]
    subdir = os.path.join(cache_dir, dataset_id, f"sub-{int(subject):03d}")
    os.makedirs(subdir, exist_ok=True)
    return os.path.join(subdir, f"{key}.npz")


def load_windows(
    dataset_id: str,
    subject: int,
    *,
    cache_dir: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, float, List[str]]:
    """Preprocess one subject.

    Returns (X, y, metadata, sfreq, ch_names) where X is
    (n_trials, n_channels, n_times) float32 bandpassed microvolts, y is
    int64 labels in the dataset's canonical class order, and metadata has
    'session' and 'run' columns. ch_names matches axis 1 of X, and the
    neighbour graph must be built from that order.
    """
    s = spec(dataset_id)
    classes = classes_for(dataset_id)

    params = {
        "dataset": dataset_id.lower(), "subject": int(subject),
        "resample": s.resample_hz, "band": [BANDPASS_LOW_HZ, BANDPASS_HIGH_HZ],
        "channels": list(s.channels) if s.channels else None,
        "classes": list(classes),
    }
    path = _cache_path(cache_dir, dataset_id, subject, params) if cache_dir else None
    if path and os.path.exists(path):
        npz = np.load(path, allow_pickle=True)
        metadata = pd.DataFrame({"session": npz["session"], "run": npz["run"]})
        return (npz["X"], npz["y"], metadata,
                float(npz["sfreq"]), list(npz["ch_names"]))

    from braindecode.preprocessing import (
        Preprocessor, create_windows_from_events, preprocess,
    )

    dataset = make_braindecode_dataset(dataset_id, subject)

    steps = [Preprocessor("pick_types", eeg=True, meg=False, stim=False)]
    if s.channels:
        # ordered=True keeps our order so the neighbour graph aligns with X.
        steps.append(Preprocessor("pick_channels", ch_names=list(s.channels),
                                  ordered=True))
    steps += [
        Preprocessor(_volts_to_microvolts, apply_on_array=True),
        Preprocessor("resample", sfreq=s.resample_hz),
        Preprocessor("filter", l_freq=BANDPASS_LOW_HZ, h_freq=BANDPASS_HIGH_HZ),
    ]
    preprocess(dataset, steps, n_jobs=1)

    sfreq = float(dataset.datasets[0].raw.info["sfreq"])
    ch_names = list(dataset.datasets[0].raw.info["ch_names"])

    windows = create_windows_from_events(
        dataset, trial_start_offset_samples=0, trial_stop_offset_samples=0,
        preload=True, mapping={name: i for i, name in enumerate(classes)},
    )

    Xs, ys, sessions, runs = [], [], [], []
    for ds in windows.datasets:
        session = str(ds.description.get("session", "0"))
        run = str(ds.description.get("run", "0"))
        for i in range(len(ds)):
            x, label, _ = ds[i]
            Xs.append(np.asarray(x, dtype=np.float32))
            ys.append(int(label))
            sessions.append(session)
            runs.append(run)

    if not Xs:
        raise RuntimeError(f"No windows for {dataset_id} subject {subject}.")

    X = np.stack(Xs)
    y = np.array(ys, dtype=np.int64)
    metadata = pd.DataFrame({"session": sessions, "run": runs})

    if path:
        np.savez(path[:-4], X=X, y=y, sfreq=sfreq,
                 session=metadata["session"].to_numpy(),
                 run=metadata["run"].to_numpy(),
                 ch_names=np.asarray(ch_names, dtype=object))
    return X, y, metadata, sfreq, ch_names


def split_train_test(X, y, metadata, dataset_id):
    """Split one subject into train and test, the way the dataset intends.

    iv2a and openbmi have two recording sessions: the first is train.
    schirrmeister2017 has one session with a '0train' and a '1test' run.
    dreyer2023 has one session with 2 calibration runs then 4 online runs;
    the calibration runs are train.
    cho2017 has a single session with a single run, so it cannot be split
    this way at all and is not currently runnable.
    """
    key = dataset_id.lower()

    if key == "schirrmeister2017":
        train = metadata["run"] == "0train"
    elif key == "dreyer2023":
        train = metadata["run"].str.contains("acquisition")
    else:
        sessions = sorted(metadata["session"].unique())
        if len(sessions) < 2:
            raise RuntimeError(
                f"{dataset_id}: needs two sessions to split, found {sessions}. "
                "cho2017 has only one session and one run, so it has no "
                "held-out block; use a different dataset."
            )
        train = metadata["session"] == sessions[0]

    train = train.to_numpy()
    if not train.any() or train.all():
        raise RuntimeError(f"{dataset_id}: train/test split produced an empty side.")
    return X[train], y[train], X[~train], y[~train]
