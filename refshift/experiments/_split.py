"""Train/test split for one subject.

Strategy resolution under 'auto':
    Schirrmeister2017 -> run-based: '0train' run -> train, '1test' -> test.
    >1 session in metadata -> cross-session: first session -> train.
    Otherwise -> stratified 80/20 within the single session.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from refshift.experiments._datasets import RUN_SPLIT_DATASETS


def encode_labels(y: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """String labels -> contiguous ints [0, n_classes)."""
    classes = sorted(np.unique(y).tolist())
    to_int = {c: i for i, c in enumerate(classes)}
    return np.asarray([to_int[v] for v in y], dtype=np.int64), classes


def split_train_test(
    X: np.ndarray, y: np.ndarray, metadata: pd.DataFrame,
    *, strategy: str = "auto", test_size: float = 0.2, seed: int = 0,
    dataset_id: Optional[str] = None,
):
    """Return (X_tr, y_tr, X_te, y_te) for one subject."""
    sessions = sorted(metadata["session"].unique())
    if strategy == "auto":
        if dataset_id is not None and dataset_id in RUN_SPLIT_DATASETS:
            effective = "run"
        elif len(sessions) > 1:
            effective = "session"
        else:
            effective = "stratify"
    else:
        effective = strategy

    if effective == "session":
        train_mask = (metadata["session"] == sessions[0]).to_numpy()
        return X[train_mask], y[train_mask], X[~train_mask], y[~train_mask]

    if effective == "run":
        if "run" not in metadata.columns:
            raise ValueError("split strategy 'run' requires a 'run' column")
        runs = sorted(metadata["run"].unique())
        if len(runs) < 2:
            raise ValueError(f"split strategy 'run' needs >=2 runs; got {runs}")
        # First run alphabetically -> train. Schirrmeister: '0train' -> train.
        train_mask = (metadata["run"] == runs[0]).to_numpy()
        return X[train_mask], y[train_mask], X[~train_mask], y[~train_mask]

    if effective == "stratify":
        from sklearn.model_selection import StratifiedShuffleSplit
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=seed,
        )
        tr, te = next(splitter.split(X, y))
        return X[tr], y[tr], X[te], y[te]

    raise ValueError(f"Unknown split strategy: {strategy!r}")
