"""Dataset registry for the five motor-imagery datasets this repo runs.

Everything dataset-specific lives in one config table, ``DATASETS``. Each entry
is a small dataclass describing how to load that dataset, what its classes are,
which channels to keep, and how to split train/test. The rest of the codebase
asks this module questions (``classes_for``, ``split_strategy_for``, ...) rather
than hardcoding per-dataset behaviour, so adding a sixth dataset means adding
one ``DatasetSpec`` here and nothing else.

The five datasets:
    iv2a              BCI IV-2a (BNCI2014_001). 4 classes, 22 ch, 2 sessions.
    openbmi           Lee2019_MI. 2 classes (left/right), 2 sessions. Needs a
                      compat fix so both sessions load (see _make_openbmi).
    cho2017           Cho2017. 2 classes.
    dreyer2023        Dreyer2023. 2 classes.
    schirrmeister2017 Schirrmeister2017. 4 classes, 44-channel motor subset,
                      Cz used as recording reference (so cz_ref is excluded),
                      run-based train/test split, resampled to 250 Hz.

There is one wrinkle worth knowing: the two pipelines pull data differently.
  * The deep-learning path loads via braindecode (see preprocess.load_windows).
  * The CSP+LDA path loads via MOABB's paradigm (see moabb_paradigm), because
    that is what the calibration check compares against.
Both ultimately use the same MOABB dataset objects described here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# Class sets, in the canonical order used to map class name -> integer label.
# Strings match MOABB's event_id keys.
IV2A_CLASSES = ("left_hand", "right_hand", "feet", "tongue")
SCHIRR_CLASSES = ("left_hand", "right_hand", "feet", "rest")
LR_CLASSES = ("left_hand", "right_hand")

# Schirrmeister 44-channel motor subset (paper Section 2.7.1). Cz is excluded
# because it was the recording reference; this also makes CSP much faster.
SCHIRRMEISTER_MOTOR_CHANNELS = (
    "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6",
    "C5", "C3", "C1", "C2", "C4", "C6",
    "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
    "FFC5h", "FFC3h", "FFC1h", "FFC2h", "FFC4h", "FFC6h",
    "FCC5h", "FCC3h", "FCC1h", "FCC2h", "FCC4h", "FCC6h",
    "CCP5h", "CCP3h", "CCP1h", "CCP2h", "CCP4h", "CCP6h",
    "CPP5h", "CPP3h", "CPP1h", "CPP2h", "CPP4h", "CPP6h",
)


@dataclass(frozen=True)
class DatasetSpec:
    """How to load and handle one dataset.

    moabb_code     MOABB dataset name (e.g. 'BNCI2014_001').
    classes        class set in canonical label order.
    paradigm       'motor_imagery' (>=2 classes) or 'left_right' (binary).
    channels       fixed channel subset to keep, or None for all EEG channels.
    split          'session' (session 1 -> train) or 'run' (first run -> train).
    resample_hz    resampling rate.
    bad_subjects   subjects to drop (corrupt data in the public release).
    excludes_cz    True if the dataset has no Cz channel (so cz_ref is invalid).
    """
    moabb_code: str
    classes: Tuple[str, ...]
    paradigm: str
    split: str
    channels: Optional[Tuple[str, ...]] = None
    resample_hz: float = 250.0
    bad_subjects: frozenset = field(default_factory=frozenset)
    excludes_cz: bool = False


# The single source of truth for per-dataset behaviour.
DATASETS = {
    "iv2a": DatasetSpec(
        moabb_code="BNCI2014_001", classes=IV2A_CLASSES,
        paradigm="motor_imagery", split="session",
    ),
    "openbmi": DatasetSpec(
        moabb_code="Lee2019_MI", classes=LR_CLASSES,
        paradigm="left_right", split="session",
        bad_subjects=frozenset({29}),  # corrupt .mat in the GigaDB release
    ),
    "cho2017": DatasetSpec(
        moabb_code="Cho2017", classes=LR_CLASSES,
        paradigm="left_right", split="session",
    ),
    "dreyer2023": DatasetSpec(
        moabb_code="Dreyer2023", classes=LR_CLASSES,
        paradigm="left_right", split="session",
    ),
    "schirrmeister2017": DatasetSpec(
        moabb_code="Schirrmeister2017", classes=SCHIRR_CLASSES,
        paradigm="motor_imagery", split="run",
        channels=SCHIRRMEISTER_MOTOR_CHANNELS, excludes_cz=True,
    ),
}

DATASET_IDS = tuple(DATASETS)


def spec(dataset_id: str) -> DatasetSpec:
    """Look up a dataset's config, with a helpful error on a typo."""
    key = dataset_id.lower()
    if key not in DATASETS:
        raise ValueError(f"Unknown dataset_id: {dataset_id!r}. Known: {DATASET_IDS}")
    return DATASETS[key]


def classes_for(dataset_id: str) -> Tuple[str, ...]:
    """The dataset's full class set, in canonical label order."""
    return spec(dataset_id).classes


def split_strategy_for(dataset_id: str) -> str:
    """'session' or 'run' -- how this dataset's train/test split is made."""
    return spec(dataset_id).split


def excludes_cz(dataset_id: str) -> bool:
    """True if the dataset has no Cz channel (Schirrmeister), so cz_ref is invalid."""
    return spec(dataset_id).excludes_cz


def subject_list(dataset_id: str) -> List[int]:
    """The dataset's usable subjects, with known-bad subjects removed."""
    dataset, _ = moabb_dataset(dataset_id)
    return list(dataset.subject_list)


# ---------------------------------------------------------------------------
# MOABB dataset/paradigm construction (with the OpenBMI compat fix)
# ---------------------------------------------------------------------------

def _make_openbmi():
    """Lee2019_MI configured to return both sessions' calibration trials.

    MOABB 1.5.0 stores sessions=(1, 2) but writes session keys as '0','1', so
    its own filter silently drops session '0'. Setting _selected_sessions=None
    bypasses the filter and recovers all the calibration trials.
    """
    from moabb.datasets import Lee2019_MI
    ds = Lee2019_MI()
    ds._selected_sessions = None
    return ds


def moabb_dataset(dataset_id: str):
    """Return ``(moabb_dataset, paradigm)`` for the CSP+LDA / calibration path.

    The paradigm is MotorImagery (using the dataset's class count) or
    LeftRightImagery for the binary datasets. Known-bad subjects are removed
    from the dataset's subject list.
    """
    s = spec(dataset_id)
    key = dataset_id.lower()

    if s.paradigm == "left_right":
        from moabb.paradigms import LeftRightImagery
        paradigm = LeftRightImagery()
        if key == "openbmi":
            ds = _make_openbmi()
        else:
            from moabb.datasets import Cho2017, Dreyer2023
            ds = {"cho2017": Cho2017, "dreyer2023": Dreyer2023}[key]()
    else:  # motor_imagery
        from moabb.paradigms import MotorImagery
        if key == "iv2a":
            from moabb.datasets import BNCI2014_001
            ds = BNCI2014_001()
            paradigm = MotorImagery(n_classes=len(s.classes))
        elif key == "schirrmeister2017":
            from moabb.datasets import Schirrmeister2017
            ds = Schirrmeister2017()
            paradigm = MotorImagery(
                n_classes=len(s.classes),
                channels=s.channels, resample=s.resample_hz,
            )
        else:
            raise ValueError(f"No motor_imagery wiring for {dataset_id!r}")

    if s.bad_subjects:
        ds.subject_list = [x for x in ds.subject_list if x not in s.bad_subjects]
    return ds, paradigm


def moabb_paradigm_channels(dataset_id: str, subject: int) -> List[str]:
    """EEG channel names in the order MOABB's paradigm output uses.

    When a fixed channel subset is set (Schirrmeister), that order is used.
    Otherwise the subject's raw EEG channel order is returned. The neighbour
    graph must be built from this exact order.
    """
    s = spec(dataset_id)
    if s.channels:
        return list(s.channels)
    dataset, _ = moabb_dataset(dataset_id)
    raws = dataset.get_data(subjects=[subject])
    raw = next(iter(next(iter(raws[subject].values())).values()))
    types = raw.get_channel_types()
    return [ch for ch, t in zip(raw.ch_names, types) if t == "eeg"]


# ---------------------------------------------------------------------------
# braindecode dataset construction (for the deep-learning path)
# ---------------------------------------------------------------------------

def make_braindecode_dataset(dataset_id: str, subject: int):
    """braindecode dataset for one subject (raw, unprocessed).

    OpenBMI is assembled by hand because braindecode's MOABBDataset builds the
    MOABB dataset internally with no way to inject the _selected_sessions fix.
    """
    from braindecode.datasets import MOABBDataset

    s = spec(dataset_id)
    if dataset_id.lower() != "openbmi":
        return MOABBDataset(dataset_name=s.moabb_code, subject_ids=[int(subject)])

    from braindecode.datasets.base import BaseConcatDataset, RawDataset
    from braindecode.datasets.moabb import fetch_data_with_moabb
    raws, description = fetch_data_with_moabb(_make_openbmi(), subject_ids=[int(subject)])
    return BaseConcatDataset([
        RawDataset(raw, row) for raw, (_, row) in zip(raws, description.iterrows())
    ])
