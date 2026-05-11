"""Dataset resolver: short id -> (MOABB dataset, paradigm).

Handles the four MOABB datasets used in the paper, plus Schirrmeister2017's
44-channel motor subset, plus the binary-reduction ablation via the classes
parameter. Known-bad subjects are filtered out by default; users can still
override via subjects=[...] explicitly.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple


DATASET_IDS = ("iv2a", "openbmi", "cho2017", "dreyer2023", "schirrmeister2017")


# OpenBMI subject 29: corrupt .mat in GigaDB release; loadmat raises mid-stream.
_KNOWN_BAD_SUBJECTS: dict = {
    "openbmi": frozenset({29}),
}


# Datasets where train/test is run-based (one session, two runs).
RUN_SPLIT_DATASETS = frozenset({"schirrmeister2017"})


# Schirrmeister2017 motor subset: 44 channels, Cz excluded as recording reference
# (paper Section 2.7.1). Schirrmeister et al. report this gave better accuracy
# than the full 128-channel cap. CSP scales as O(C^3), so this also cuts CSP+LDA
# per-subject runtime from ~13 min to ~1 min on CPU.
SCHIRRMEISTER_MOTOR_CHANNELS = (
    "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6",
    "C5", "C3", "C1", "C2", "C4", "C6",
    "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
    "FFC5h", "FFC3h", "FFC1h", "FFC2h", "FFC4h", "FFC6h",
    "FCC5h", "FCC3h", "FCC1h", "FCC2h", "FCC4h", "FCC6h",
    "CCP5h", "CCP3h", "CCP1h", "CCP2h", "CCP4h", "CCP6h",
    "CPP5h", "CPP3h", "CPP1h", "CPP2h", "CPP4h", "CPP6h",
)


# Per-dataset full class set. Class names match MOABB's event_id keys
# (e.g. BNCI2014_001 uses 'left_hand','right_hand','feet','tongue').
IV2A_CLASSES = ("left_hand", "right_hand", "feet", "tongue")
SCHIRR_CLASSES = ("left_hand", "right_hand", "feet", "rest")
LR_CLASSES = ("left_hand", "right_hand")

# Aliases preserved for backwards compatibility with internal call sites.
_IV2A_CLASSES = IV2A_CLASSES
_SCHIRR_CLASSES = SCHIRR_CLASSES
_LR_CLASSES = LR_CLASSES


_DATASET_CLASSES = {
    "iv2a": IV2A_CLASSES,
    "openbmi": LR_CLASSES,
    "cho2017": LR_CLASSES,
    "dreyer2023": LR_CLASSES,
    "schirrmeister2017": SCHIRR_CLASSES,
}


def dataset_full_classes(dataset_id: str) -> Tuple[str, ...]:
    """Full class set for a dataset, in the canonical order used to build
    the int<->name mapping. Same strings as MOABB's event_id keys.
    """
    dataset_id = dataset_id.lower()
    if dataset_id not in _DATASET_CLASSES:
        raise ValueError(
            f"Unknown dataset_id: {dataset_id!r}. Known: {tuple(_DATASET_CLASSES)}"
        )
    return _DATASET_CLASSES[dataset_id]


def resolve_classes(
    dataset_id: str,
    classes: Optional[Sequence[str]] = None,
) -> Tuple[str, ...]:
    """Validate and return the class set for this dataset.

    classes=None returns the full class set; otherwise validates that all
    entries are present in the dataset's class set and that there are >=2
    distinct labels.
    """
    full = dataset_full_classes(dataset_id)
    if classes is None:
        return full
    classes_t = tuple(classes)
    _validate_classes(classes_t, full)
    return classes_t


def _validate_classes(classes_t: Optional[Tuple[str, ...]], allowed: Tuple[str, ...]) -> None:
    if classes_t is None:
        return
    if len(classes_t) == 0:
        raise ValueError("classes=() is empty; pass None for default")
    unknown = [c for c in classes_t if c not in allowed]
    if unknown:
        raise ValueError(f"Unknown classes: {unknown}. Allowed: {allowed}")
    if len(set(classes_t)) < 2:
        raise ValueError(f"classes={classes_t} has fewer than 2 distinct labels")


def resolve_dataset(
    dataset_id: str,
    classes: Optional[Sequence[str]] = None,
):
    """Return (dataset, paradigm) for a short dataset_id.

    classes=None loads the dataset's full class set. For iv2a/schirrmeister,
    a non-None classes tuple is wired into MotorImagery(events=...). For
    LeftRightImagery datasets only ('left_hand', 'right_hand') is valid.

    MOABB workaround: when events is set, n_classes must equal len(events)
    or used_events crashes with TypeError. We always pass both.
    """
    dataset_id = dataset_id.lower()
    classes_t = tuple(classes) if classes is not None else None

    if dataset_id == "iv2a":
        from moabb.datasets import BNCI2014_001
        from moabb.paradigms import MotorImagery
        _validate_classes(classes_t, _IV2A_CLASSES)
        if classes_t is None:
            paradigm = MotorImagery(n_classes=4)
        else:
            paradigm = MotorImagery(events=list(classes_t), n_classes=len(classes_t))
        ds = BNCI2014_001()

    elif dataset_id == "openbmi":
        from moabb.paradigms import LeftRightImagery
        from refshift.compat import make_openbmi_dataset
        _validate_classes(classes_t, _LR_CLASSES)
        ds, paradigm = make_openbmi_dataset(), LeftRightImagery()

    elif dataset_id == "cho2017":
        from moabb.datasets import Cho2017
        from moabb.paradigms import LeftRightImagery
        _validate_classes(classes_t, _LR_CLASSES)
        ds, paradigm = Cho2017(), LeftRightImagery()

    elif dataset_id == "dreyer2023":
        from moabb.datasets import Dreyer2023
        from moabb.paradigms import LeftRightImagery
        _validate_classes(classes_t, _LR_CLASSES)
        ds, paradigm = Dreyer2023(), LeftRightImagery()

    elif dataset_id == "schirrmeister2017":
        from moabb.datasets import Schirrmeister2017
        from moabb.paradigms import MotorImagery
        _validate_classes(classes_t, _SCHIRR_CLASSES)
        # Resample to 250 Hz to match IV-2a's rate (paper bandpass is 8-32 Hz,
        # well below Nyquist for 250 Hz). Halves the per-trial sample count
        # (2000 -> 1000) and keeps Shallow's filter_time_length in the same
        # physical-time regime as on IV-2a (~100 ms).
        paradigm_kwargs = dict(
            channels=SCHIRRMEISTER_MOTOR_CHANNELS,
            resample=250.0,
        )
        ds = Schirrmeister2017()
        if classes_t is None:
            paradigm = MotorImagery(n_classes=4, **paradigm_kwargs)
        else:
            paradigm = MotorImagery(
                events=list(classes_t), n_classes=len(classes_t),
                **paradigm_kwargs,
            )

    else:
        raise ValueError(f"Unknown dataset_id: {dataset_id!r}. Known: {DATASET_IDS}")

    bad = _KNOWN_BAD_SUBJECTS.get(dataset_id, frozenset())
    if bad:
        ds.subject_list = [s for s in ds.subject_list if s not in bad]
    return ds, paradigm


def get_eeg_channel_names(dataset, subject=None, paradigm=None):
    """Return EEG channel names matching the X array's channel axis.

    When paradigm.channels is set (e.g. Schirrmeister motor subset), MOABB's
    RawToEpochs uses pick_channels(include=..., ordered=True), preserving the
    user-supplied order. This must be the order returned here so the
    neighbour graph aligns with the channel axis of the X paradigm produces.
    """
    if paradigm is not None and getattr(paradigm, "channels", None):
        return list(paradigm.channels)
    if subject is None:
        subject = dataset.subject_list[0]
    raws = dataset.get_data(subjects=[subject])
    raw = next(iter(next(iter(raws[subject].values())).values()))
    types = raw.get_channel_types()
    return [ch for ch, t in zip(raw.ch_names, types) if t == "eeg"]


def build_cache_config(path=None):
    """MOABB CacheConfig that saves and reads the final ndarray output."""
    from moabb.datasets.base import CacheConfig
    return CacheConfig(save_array=True, use=True, path=path)
