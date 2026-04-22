"""Thin helpers around MOABB dataset loading.

The goal here is to avoid parallel code paths: everything EEG-related goes
through MOABB's dataset classes and paradigm pipeline. We only add:

    get_eeg_channel_names(dataset)    peek at one subject to learn the EEG
                                      channel set (needed for graph build)
    load_paradigm_data(...)           paradigm.get_data with sensible defaults
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def get_eeg_channel_names(dataset, subject: Optional[int] = None) -> List[str]:
    """Return the list of EEG channel names for a MOABB dataset.

    Loads a single subject (the first by default) to inspect ``raw.info``
    and filter to EEG-typed channels, preserving the order that MOABB will
    use downstream in its paradigm pipeline.

    Parameters
    ----------
    dataset : moabb.datasets.base.BaseDataset
    subject : int or None
        Which subject to use as the reference. Defaults to
        ``dataset.subject_list[0]``.

    Returns
    -------
    list of str
        EEG channel names in MOABB's native order for this dataset.
    """
    if subject is None:
        subject = dataset.subject_list[0]
    raws_nested = dataset.get_data(subjects=[subject])
    # data = {subject: {session: {run: Raw}}}
    subject_data = raws_nested[subject]
    session_data = next(iter(subject_data.values()))
    raw = next(iter(session_data.values()))
    # Filter to EEG-typed channels, preserving order.
    types = raw.get_channel_types()
    return [ch for ch, t in zip(raw.ch_names, types) if t == "eeg"]


def load_paradigm_data(
    paradigm,
    dataset,
    subject: int,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load epoched data for one subject via MOABB's paradigm.

    Runs MOABB's full pipeline (SetRawAnnotations -> bandpass on Raw ->
    epoching -> resample -> get_data -> scaling) and returns arrays.

    Returns
    -------
    X : np.ndarray, shape (n_trials, n_channels, n_times), float
    y : np.ndarray, shape (n_trials,)
        String labels as produced by MOABB.
    metadata : pandas.DataFrame
        Rows aligned with X. Contains at least ``subject``, ``session``,
        ``run`` columns; use ``metadata.session`` to implement session-split
        train/test for IV-2a and OpenBMI.
    """
    X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[subject])
    return X, y, metadata


def split_train_test(
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    *,
    strategy: str = "auto",
    test_size: float = 0.2,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Produce (X_tr, y_tr, X_te, y_te) for a single subject.

    strategy :
        'auto'     : cross-session if >1 session, else stratified shuffle split
        'session'  : first session -> train, remaining sessions -> test
        'stratify' : 80/20 stratified shuffle split regardless of session count

    For IV-2a and OpenBMI 'auto' yields the native session split. For Cho2017
    and Dreyer2023 'auto' yields a stratified split seeded by ``seed``.
    """
    sessions = sorted(metadata["session"].unique())
    effective = strategy
    if strategy == "auto":
        effective = "session" if len(sessions) > 1 else "stratify"

    if effective == "session":
        train_session = sessions[0]
        train_mask = (metadata["session"] == train_session).to_numpy()
        test_mask = ~train_mask
        return X[train_mask], y[train_mask], X[test_mask], y[test_mask]

    if effective == "stratify":
        from sklearn.model_selection import StratifiedShuffleSplit
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=seed,
        )
        tr_idx, te_idx = next(splitter.split(X, y))
        return X[tr_idx], y[tr_idx], X[te_idx], y[te_idx]

    raise ValueError(f"Unknown split strategy: {strategy!r}")


def encode_labels(y: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """Encode string labels to contiguous ints [0, n_classes). Returns
    (y_encoded, class_names) where class_names[i] is the original string
    label for integer i.
    """
    classes = sorted(np.unique(y).tolist())
    to_int = {c: i for i, c in enumerate(classes)}
    return np.asarray([to_int[v] for v in y], dtype=np.int64), classes
