"""Dataset registry and the train/test split (config only; no downloads)."""

import numpy as np
import pandas as pd
import pytest

from refshift.datasets import DATASET_IDS, classes_for, spec
from refshift.preprocess import split_train_test
from refshift.references import reference_modes_for_dataset


def test_registry():
    assert set(DATASET_IDS) == {
        "iv2a", "openbmi", "schirrmeister2017", "cho2017", "dreyer2023"}
    assert len(classes_for("iv2a")) == 4
    assert len(classes_for("schirrmeister2017")) == 4
    assert classes_for("openbmi") == ("left_hand", "right_hand")
    with pytest.raises(ValueError, match="Unknown dataset_id"):
        spec("not_a_dataset")


def test_schirrmeister_drops_cz_ref():
    # Schirrmeister recorded against Cz, so it has no Cz channel.
    assert "cz_ref" not in reference_modes_for_dataset("schirrmeister2017")
    assert "cz_ref" in reference_modes_for_dataset("iv2a")


def _frame(sessions, runs):
    n = len(sessions)
    X = np.zeros((n, 2, 4), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)
    return X, y, pd.DataFrame({"session": sessions, "run": runs})


def test_split_two_sessions():
    X, y, meta = _frame(["0train"] * 3 + ["1test"] * 2, ["0"] * 5)
    _, y_tr, _, y_te = split_train_test(X, y, meta, "iv2a")
    assert (len(y_tr), len(y_te)) == (3, 2)


def test_split_schirrmeister_by_run():
    X, y, meta = _frame(["0"] * 5, ["0train"] * 3 + ["1test"] * 2)
    _, y_tr, _, y_te = split_train_test(X, y, meta, "schirrmeister2017")
    assert (len(y_tr), len(y_te)) == (3, 2)


def test_split_dreyer_trains_on_acquisition_runs():
    # One session, 2 calibration runs then 4 online runs.
    runs = ["0R1acquisition", "1R2acquisition",
            "2R3online", "3R4online", "4R5online", "5R6online"]
    X, y, meta = _frame(["0"] * 6, runs)
    _, y_tr, _, y_te = split_train_test(X, y, meta, "dreyer2023")
    assert (len(y_tr), len(y_te)) == (2, 4)


def test_split_cho2017_fails_loudly():
    # Single session, single run: there is no held-out block.
    X, y, meta = _frame(["0"] * 4, ["0"] * 4)
    with pytest.raises(RuntimeError, match="needs two sessions"):
        split_train_test(X, y, meta, "cho2017")
