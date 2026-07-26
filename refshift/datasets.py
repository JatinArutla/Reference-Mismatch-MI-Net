"""The datasets this project runs on, and how to load them.

One table, ``DATASETS``, holds everything dataset-specific: the MOABB dataset
name, the class set, an optional channel subset, and any subjects to drop.

    iv2a               BCI IV-2a (BNCI2014_001). 4 classes, 22 ch, 2 sessions.
    openbmi            Lee2019_MI. 2 classes, 2 sessions.
    schirrmeister2017  Schirrmeister2017. 4 classes, 44-channel motor subset,
                       recorded against Cz (so cz_ref is excluded), 2 runs.
    cho2017            Cho2017. 2 classes. NOT RUNNABLE: see split_train_test.
    dreyer2023         Dreyer2023. 2 classes, 6 runs in one session.

Data is loaded through braindecode (see preprocess.load_windows). MOABB is used
directly only by calibrate_csp_lda, which compares against MOABB's own baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

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
    moabb_code: str                                 # e.g. 'BNCI2014_001'
    classes: Tuple[str, ...]                        # canonical label order
    channels: Optional[Tuple[str, ...]] = None      # None = all EEG channels
    resample_hz: float = 250.0
    bad_subjects: frozenset = field(default_factory=frozenset)


DATASETS = {
    "iv2a": DatasetSpec("BNCI2014_001", IV2A_CLASSES),
    "openbmi": DatasetSpec(
        "Lee2019_MI", LR_CLASSES,
        bad_subjects=frozenset({29}),               # corrupt .mat in GigaDB
    ),
    "schirrmeister2017": DatasetSpec(
        "Schirrmeister2017", SCHIRR_CLASSES, channels=SCHIRRMEISTER_MOTOR_CHANNELS,
    ),
    "cho2017": DatasetSpec("Cho2017", LR_CLASSES),
    "dreyer2023": DatasetSpec("Dreyer2023", LR_CLASSES),
}

DATASET_IDS = tuple(DATASETS)


def spec(dataset_id: str) -> DatasetSpec:
    key = dataset_id.lower()
    if key not in DATASETS:
        raise ValueError(f"Unknown dataset_id: {dataset_id!r}. Known: {DATASET_IDS}")
    return DATASETS[key]


def classes_for(dataset_id: str) -> Tuple[str, ...]:
    return spec(dataset_id).classes


def subject_list(dataset_id: str) -> List[int]:
    """Usable subjects, with known-bad ones removed."""
    dataset, _ = moabb_dataset(dataset_id)
    return list(dataset.subject_list)


def _make_openbmi():
    """Lee2019_MI configured to return BOTH sessions.

    DO NOT SIMPLIFY. Lee2019_MI defaults to sessions=(1, 2) but writes its
    session keys as '0' and '1', so MOABB's own filter keeps only '1' and
    silently drops half the trials, which also leaves one session and breaks
    the train/test split. Setting _selected_sessions=None bypasses the filter.
    """
    from moabb.datasets import Lee2019_MI
    ds = Lee2019_MI()
    ds._selected_sessions = None
    return ds


def moabb_dataset(dataset_id: str):
    """Return ``(moabb_dataset, paradigm)``. Used by calibrate_csp_lda."""
    s = spec(dataset_id)

    if dataset_id.lower() == "openbmi":
        from moabb.paradigms import LeftRightImagery
        return _make_openbmi(), LeftRightImagery()

    import moabb.datasets as md
    ds = getattr(md, s.moabb_code)()
    if s.bad_subjects:
        ds.subject_list = [x for x in ds.subject_list if x not in s.bad_subjects]

    if len(s.classes) == 2:
        from moabb.paradigms import LeftRightImagery
        return ds, LeftRightImagery()

    from moabb.paradigms import MotorImagery
    if s.channels:
        return ds, MotorImagery(n_classes=len(s.classes),
                                channels=list(s.channels), resample=s.resample_hz)
    return ds, MotorImagery(n_classes=len(s.classes))


def make_braindecode_dataset(dataset_id: str, subject: int):
    """braindecode dataset for one subject (raw, unprocessed)."""
    from braindecode.datasets import MOABBDataset

    s = spec(dataset_id)
    if dataset_id.lower() != "openbmi":
        return MOABBDataset(dataset_name=s.moabb_code, subject_ids=[int(subject)])

    # MOABBDataset builds the MOABB object internally with no way to inject the
    # _selected_sessions fix, so assemble OpenBMI by hand. See _make_openbmi.
    from braindecode.datasets.base import BaseConcatDataset, RawDataset
    from braindecode.datasets.moabb import fetch_data_with_moabb
    raws, description = fetch_data_with_moabb(_make_openbmi(), subject_ids=[int(subject)])
    return BaseConcatDataset([
        RawDataset(raw, row) for raw, (_, row) in zip(raws, description.iterrows())
    ])
