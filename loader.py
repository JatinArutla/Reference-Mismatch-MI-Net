"""
refshift.data.loader — unified subject loader for all four datasets.

Loads one subject at a time and returns epoched arrays resampled to 250 Hz.
No bandpass is applied here; that happens in the preprocessing module.

Datasets:
    iv2a       — MOABB BNCI2014_001, session split
    openbmi    — scipy.io direct (Lee2019 `smt` field), session split
    cho2017    — MOABB Cho2017, single session (no pre-built split)
    dreyer2023 — direct EDF + events.tsv reader, acquisition runs only

Label convention (0-indexed):
    0 = left_hand
    1 = right_hand
    2 = feet      (iv2a only)
    3 = tongue    (iv2a only)

Units:  Volts (MNE/MOABB convention). OpenBMI smt values are converted μV→V
        inside the loader so all datasets share the same scale.
Dtype:  float32 for X, int64 for y.
Shape:  X is [N, C, T], y is [N].
"""

from __future__ import annotations

import os
import re
import warnings
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Optional

import numpy as np


# ============================================================================
# Constants
# ============================================================================

TARGET_SFREQ_HZ = 250.0

WINDOWS = {
    "iv2a":       (2.0, 6.0),
    "openbmi":    (0.0, 4.0),
    "cho2017":    (0.0, 3.0),
    "dreyer2023": (0.0, 5.0),
}

EXPECTED_SAMPLES = {
    "iv2a":       1000,
    "openbmi":    1000,
    "cho2017":     750,
    "dreyer2023": 1250,
}

EXPECTED_N_CHANNELS = {
    "iv2a":       22,
    "openbmi":    62,
    "cho2017":    64,
    "dreyer2023": 27,
}

# Trial counts are ranges because MOABB occasionally drops boundary epochs
# (last trial per run) when the window extends past the end of the recording.
# This matches braindecode's behavior too; it's not a bug in our code.
EXPECTED_TRIALS = {
    "iv2a":       {"train_min": 270, "train_max": 288,
                   "test_min":  270, "test_max":  288},
    "openbmi":    {"train": 100, "test": 100},
    "cho2017":    {"total_min": 200, "total_max": 240},
    "dreyer2023": {"total": 80},
}

EXCLUDED_SUBJECTS = {
    "openbmi": {29},
}

ALL_SUBJECTS = {
    "iv2a":       list(range(1, 10)),
    "openbmi":    [s for s in range(1, 55) if s not in EXCLUDED_SUBJECTS["openbmi"]],
    "cho2017":    list(range(1, 53)),
    "dreyer2023": list(range(1, 88)),
}

# Kaggle dataset roots; overridable via env vars
OPENBMI_DEFAULT_ROOT = Path(os.environ.get(
    "REFSHIFT_OPENBMI_ROOT",
    "/kaggle/input/datasets/imaginer369/openbmi-dataset",
))

DREYER_DEFAULT_ROOT = Path(os.environ.get(
    "REFSHIFT_DREYER_ROOT",
    "/kaggle/input/datasets/delhialli/dreyer2023/MNE-Dreyer2023-data",
))

DREYER_ACQUISITION_TASKS = ("R1acquisition", "R2acquisition")

# MI cue event codes in the Dreyer events.tsv (trial_type column)
DREYER_MI_CUE_CODES = {769: "left_hand", 770: "right_hand"}


# ============================================================================
# Return type
# ============================================================================

@dataclass
class SubjectData:
    """Container for one subject's epoched data.

    For session-split datasets (iv2a, openbmi), X_train/y_train and
    X_test/y_test are populated and X_all/y_all are None. For single-session
    datasets (cho2017, dreyer2023), X_all/y_all are populated and the
    train/test fields are None. Downstream code decides the split policy.
    """
    dataset_id: str
    subject: int
    ch_names: list
    sfreq: float
    X_train: Optional[np.ndarray] = None
    y_train: Optional[np.ndarray] = None
    X_test:  Optional[np.ndarray] = None
    y_test:  Optional[np.ndarray] = None
    X_all:   Optional[np.ndarray] = None
    y_all:   Optional[np.ndarray] = None

    def has_session_split(self) -> bool:
        return self.X_train is not None


# ============================================================================
# Shared helpers
# ============================================================================

def _resample_epochs(X: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """Resample [N, C, T] along the time axis via polyphase filter."""
    if abs(fs_in - fs_out) < 1e-9:
        return np.ascontiguousarray(X, dtype=np.float32)
    from scipy.signal import resample_poly
    frac = Fraction(fs_out / fs_in).limit_denominator(1000)
    up, down = frac.numerator, frac.denominator
    out = resample_poly(X, up, down, axis=-1)
    return np.ascontiguousarray(out, dtype=np.float32)


def _epoch_raw(raw, event_name_to_label: dict, tmin: float, tmax: float):
    """Extract fixed-length epochs from a Raw. Returns (X, y, ch_names).

    `reject_by_annotation=False` keeps artifact-flagged trials (deep-learning
    convention, matching ATCNet / EEGNet / braindecode defaults for IV-2a).
    Only EEG channels are kept. Trims MNE's inclusive-tmax endpoint so the
    output is exactly `(tmax - tmin) * sfreq` samples.
    """
    import mne
    present = set(raw.annotations.description)
    active = {k: v for k, v in event_name_to_label.items() if k in present}
    if not active:
        raise RuntimeError(
            f"No annotations matching {list(event_name_to_label)} "
            f"(present: {sorted(present)[:20]})"
        )
    mne_event_id = {name: i + 1 for i, name in enumerate(active)}
    events, _ = mne.events_from_annotations(raw, event_id=mne_event_id, verbose=False)
    if len(events) == 0:
        raise RuntimeError("events_from_annotations returned no events")

    epochs = mne.Epochs(
        raw, events, event_id=mne_event_id,
        tmin=tmin, tmax=tmax,
        baseline=None,
        preload=True,
        picks="eeg",
        reject_by_annotation=False,  # keep artifact-flagged trials
        verbose="ERROR",
    )
    X = epochs.get_data(copy=False).astype(np.float32)
    expected_T = int(round((tmax - tmin) * epochs.info["sfreq"]))
    if X.shape[-1] > expected_T:
        X = X[..., :expected_T]
    X = np.ascontiguousarray(X)

    inv_mne = {v: k for k, v in mne_event_id.items()}
    y = np.array(
        [active[inv_mne[code]] for code in epochs.events[:, 2]],
        dtype=np.int64,
    )
    ch_names = epochs.ch_names
    return X, y, ch_names


def _ensure_moabb_cache_symlinks(verbose: bool = False):
    """Symlink Kaggle input files into MOABB's expected cache layout.

    Safe to call multiple times. Only sets up what's present on the local
    filesystem. Only needed for iv2a and cho2017 (openbmi and dreyer2023
    load directly from the Kaggle input).
    """
    mne_data = Path(os.environ.get("MNE_DATA", "/kaggle/working/mne_data"))
    mne_data.mkdir(parents=True, exist_ok=True)
    os.environ["MNE_DATA"] = str(mne_data)

    counters = {"bnci": 0, "cho": 0}

    def _link(src: Path, dst: Path):
        if dst.exists():
            return False
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.symlink(src, dst)
            return True
        except FileExistsError:
            return False

    bnci_src = Path("/kaggle/input/datasets/delhialli/four-class-motor-imagery-bnci-001-2014")
    bnci_dst = mne_data / "MNE-bnci-data" / "~bci" / "database" / "001-2014"
    if bnci_src.exists():
        for f in bnci_src.glob("*.mat"):
            if _link(f, bnci_dst / f.name):
                counters["bnci"] += 1

    cho_src = Path("/kaggle/input/datasets/delhialli/cho2017")
    cho_dst = (mne_data / "MNE-gigadb-data" / "gigadb-datasets" / "live" / "pub"
               / "10.5524" / "100001_101000" / "100295" / "mat_data")
    if cho_src.exists():
        for f in cho_src.glob("*.mat"):
            if _link(f, cho_dst / f.name):
                counters["cho"] += 1

    if verbose:
        print(f"MOABB cache root: {mne_data}")
        print(f"  New symlinks: bnci={counters['bnci']}, cho={counters['cho']}")


# ============================================================================
# Per-dataset loaders
# ============================================================================

def _load_iv2a(subject: int) -> SubjectData:
    from moabb.datasets import BNCI2014_001

    _ensure_moabb_cache_symlinks()
    ds = BNCI2014_001()
    data = ds.get_data(subjects=[subject])
    sessions = data[subject]

    train_key = next(k for k in sessions if "train" in k)
    test_key  = next(k for k in sessions if "test"  in k)

    tmin, tmax = WINDOWS["iv2a"]
    label_map = {"left_hand": 0, "right_hand": 1, "feet": 2, "tongue": 3}

    def epoch_session(sess_dict):
        Xs, ys, chs = [], [], None
        for run_key in sorted(sess_dict.keys()):
            raw = sess_dict[run_key]
            try:
                X, y, ch_names = _epoch_raw(raw, label_map, tmin, tmax)
            except RuntimeError:
                continue  # baseline runs with no MI events
            Xs.append(X); ys.append(y)
            if chs is None:
                chs = ch_names
        if not Xs:
            raise RuntimeError("No epochs extracted from session")
        return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0), chs

    X_tr, y_tr, ch_names = epoch_session(sessions[train_key])
    X_te, y_te, _        = epoch_session(sessions[test_key])

    native_sfreq = sessions[train_key][next(iter(sessions[train_key]))].info["sfreq"]
    X_tr = _resample_epochs(X_tr, native_sfreq, TARGET_SFREQ_HZ)
    X_te = _resample_epochs(X_te, native_sfreq, TARGET_SFREQ_HZ)

    return SubjectData(
        dataset_id="iv2a", subject=subject,
        ch_names=list(ch_names), sfreq=TARGET_SFREQ_HZ,
        X_train=X_tr, y_train=y_tr, X_test=X_te, y_test=y_te,
    )


def _load_openbmi(subject: int, root: Optional[Path] = None) -> SubjectData:
    """OpenBMI loader using `scipy.io.loadmat` directly.

    Reads the pre-epoched `smt` field from each session's .mat file. `smt`
    has shape (T=4000, N=100, C=62) at 1000 Hz, which is exactly the MOABB
    `interval=[0, 4]` window relative to cue onset. We transpose to
    (N, C, T) = (100, 62, 4000), convert μV→V, then resample to 250 Hz
    giving (100, 62, 1000).

    Session 1 = train, session 2 = test. Uses only the offline training
    subrun (`EEG_MI_train`), matching MOABB's default `train_run=True,
    test_run=False`. This is the standard convention for MOABB-compatible
    Lee2019_MI benchmarks.

    Label mapping: y_dec=1 is right_hand (→ our label 1), y_dec=2 is
    left_hand (→ our label 0). This matches MOABB's Lee2019 event map
    `{"left_hand": 2, "right_hand": 1}` combined with our global
    convention `{"left_hand": 0, "right_hand": 1}`.
    """
    import scipy.io

    if subject in EXCLUDED_SUBJECTS["openbmi"]:
        raise ValueError(
            f"OpenBMI subject {subject} is excluded "
            f"(reason: sess01 file is corrupt in Kaggle copy)."
        )

    root = Path(root) if root is not None else OPENBMI_DEFAULT_ROOT
    if not root.exists():
        raise FileNotFoundError(f"OpenBMI root not found: {root}")

    def load_session(sess_num: int):
        path = root / f"sess{sess_num:02d}_subj{subject:02d}_EEG_MI.mat"
        if not path.exists():
            raise FileNotFoundError(f"OpenBMI file not found: {path}")
        mat = scipy.io.loadmat(
            path, squeeze_me=True, struct_as_record=False,
            verify_compressed_data_integrity=False,
        )
        train = mat["EEG_MI_train"]

        # smt is (T=4000, N=100, C=62) at 1000 Hz
        smt = np.asarray(train.smt)  # (4000, 100, 62)
        X = np.transpose(smt, (1, 2, 0)).astype(np.float32)  # (100, 62, 4000)

        # y_dec: 1=right_hand, 2=left_hand (Lee2019 convention)
        # Our convention: 0=left_hand, 1=right_hand
        y_dec = np.atleast_1d(train.y_dec).astype(int)
        y = (y_dec == 1).astype(np.int64)  # right=1, left=0

        # Channel names (62 EEG; EMG is stored separately in EMG_index)
        chan = [str(c).strip() for c in np.atleast_1d(train.chan).tolist()]

        # smt values are in microvolts; convert to Volts to match MOABB
        # convention used by the other datasets (~1e-5 amplitude scale)
        X = X * 1e-6

        native_fs = float(np.atleast_1d(train.fs).item())
        return X, y, chan, native_fs

    X_tr, y_tr, ch_names, native_sfreq = load_session(1)
    X_te, y_te, _,        _            = load_session(2)

    X_tr = _resample_epochs(X_tr, native_sfreq, TARGET_SFREQ_HZ)
    X_te = _resample_epochs(X_te, native_sfreq, TARGET_SFREQ_HZ)

    return SubjectData(
        dataset_id="openbmi", subject=subject,
        ch_names=list(ch_names), sfreq=TARGET_SFREQ_HZ,
        X_train=X_tr, y_train=y_tr, X_test=X_te, y_test=y_te,
    )


def _load_cho2017(subject: int) -> SubjectData:
    from moabb.datasets import Cho2017

    _ensure_moabb_cache_symlinks()
    ds = Cho2017()
    data = ds.get_data(subjects=[subject])
    sessions = data[subject]

    sess_key = next(iter(sessions))
    run_dict = sessions[sess_key]
    raw = next(iter(run_dict.values()))

    tmin, tmax = WINDOWS["cho2017"]
    label_map = {"left_hand": 0, "right_hand": 1}
    X, y, ch_names = _epoch_raw(raw, label_map, tmin, tmax)
    X = _resample_epochs(X, raw.info["sfreq"], TARGET_SFREQ_HZ)

    return SubjectData(
        dataset_id="cho2017", subject=subject,
        ch_names=list(ch_names), sfreq=TARGET_SFREQ_HZ,
        X_all=X, y_all=y,
    )


def _load_dreyer2023(subject: int, root: Optional[Path] = None) -> SubjectData:
    """Dreyer loader using `mne.io.read_raw_edf` + `events.tsv` directly.

    Avoids `mne_bids.read_raw_bids` because it tries to create `.lock` sidecar
    files, which fails on Kaggle's read-only input filesystem. Net behavior
    mirrors MOABB's Dreyer2023 loader (acquisition runs only; 769→left_hand,
    770→right_hand).
    """
    import mne
    import pandas as pd

    root = Path(root) if root is not None else DREYER_DEFAULT_ROOT
    if not root.exists():
        raise FileNotFoundError(f"Dreyer root not found: {root}")

    tmin, tmax = WINDOWS["dreyer2023"]
    label_map = {"left_hand": 0, "right_hand": 1}

    Xs, ys, ch_names = [], [], None
    native_sfreq = None

    for task in DREYER_ACQUISITION_TASKS:
        sub_dir = root / f"sub-{subject:02d}" / "eeg"
        edf_file    = sub_dir / f"sub-{subject:02d}_task-{task}_eeg.edf"
        events_file = sub_dir / f"sub-{subject:02d}_task-{task}_events.tsv"
        if not edf_file.exists():
            raise FileNotFoundError(f"Dreyer EDF missing: {edf_file}")
        if not events_file.exists():
            raise FileNotFoundError(f"Dreyer events.tsv missing: {events_file}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = mne.io.read_raw_edf(str(edf_file), preload=True, verbose="ERROR")

        events_df = pd.read_csv(events_file, sep="\t")
        mi = events_df[events_df["trial_type"].isin(DREYER_MI_CUE_CODES.keys())].copy()
        if len(mi) == 0:
            raise RuntimeError(f"No MI cue events (769/770) in {events_file.name}")
        mi["description"] = mi["trial_type"].map(DREYER_MI_CUE_CODES)

        raw.set_annotations(mne.Annotations(
            onset=mi["onset"].values.astype(float),
            duration=np.zeros(len(mi), dtype=float),
            description=mi["description"].values,
        ))

        ch_type_map = {}
        for ch in raw.ch_names:
            if "EOG" in ch:
                ch_type_map[ch] = "eog"
            elif "EMG" in ch:
                ch_type_map[ch] = "emg"
        if ch_type_map:
            raw.set_channel_types(ch_type_map)

        X, y, chs = _epoch_raw(raw, label_map, tmin, tmax)
        Xs.append(X); ys.append(y)
        if ch_names is None:
            ch_names = chs
        if native_sfreq is None:
            native_sfreq = raw.info["sfreq"]

    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    X = _resample_epochs(X, native_sfreq, TARGET_SFREQ_HZ)

    return SubjectData(
        dataset_id="dreyer2023", subject=subject,
        ch_names=list(ch_names), sfreq=TARGET_SFREQ_HZ,
        X_all=X, y_all=y,
    )


# ============================================================================
# Public API
# ============================================================================

_LOADERS = {
    "iv2a":       _load_iv2a,
    "openbmi":    _load_openbmi,
    "cho2017":    _load_cho2017,
    "dreyer2023": _load_dreyer2023,
}


def load_subject(dataset_id: str, subject: int) -> SubjectData:
    if dataset_id not in _LOADERS:
        raise ValueError(
            f"Unknown dataset {dataset_id!r}. Known: {sorted(_LOADERS)}"
        )
    return _LOADERS[dataset_id](subject)


# ============================================================================
# Simple cache (subject-level npz)
# ============================================================================

def cache_path(cache_root, dataset_id: str, subject: int) -> Path:
    return Path(cache_root) / dataset_id / f"subject_{subject:03d}.npz"


def save_subject_cache(data: SubjectData, cache_root) -> Path:
    path = cache_path(cache_root, data.dataset_id, data.subject)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"ch_names": np.array(data.ch_names, dtype=object),
              "sfreq": np.array(data.sfreq, dtype=np.float64)}
    for name in ("X_train", "y_train", "X_test", "y_test", "X_all", "y_all"):
        val = getattr(data, name)
        if val is not None:
            arrays[name] = val
    np.savez_compressed(path, **arrays)
    return path


def load_subject_cache(cache_root, dataset_id: str, subject: int) -> SubjectData:
    path = cache_path(cache_root, dataset_id, subject)
    if not path.exists():
        raise FileNotFoundError(f"No cached file at {path}")
    d = np.load(path, allow_pickle=True)
    return SubjectData(
        dataset_id=dataset_id,
        subject=subject,
        ch_names=list(d["ch_names"]),
        sfreq=float(d["sfreq"]),
        X_train=d["X_train"] if "X_train" in d.files else None,
        y_train=d["y_train"] if "y_train" in d.files else None,
        X_test =d["X_test"]  if "X_test"  in d.files else None,
        y_test =d["y_test"]  if "y_test"  in d.files else None,
        X_all  =d["X_all"]   if "X_all"   in d.files else None,
        y_all  =d["y_all"]   if "y_all"   in d.files else None,
    )
