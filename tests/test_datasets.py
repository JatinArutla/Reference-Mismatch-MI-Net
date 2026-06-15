"""Tests for the dataset registry (config table only; no MOABB download)."""

import pytest

from refshift.datasets import (
    DATASET_IDS,
    classes_for,
    excludes_cz,
    spec,
    split_strategy_for,
)
from refshift.references import reference_modes_for_dataset


def test_all_five_datasets_present():
    assert set(DATASET_IDS) == {
        "iv2a", "openbmi", "cho2017", "dreyer2023", "schirrmeister2017",
    }


def test_spec_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown dataset_id"):
        spec("not_a_dataset")


def test_class_counts():
    assert len(classes_for("iv2a")) == 4
    assert len(classes_for("schirrmeister2017")) == 4
    for binary in ("openbmi", "cho2017", "dreyer2023"):
        assert classes_for(binary) == ("left_hand", "right_hand")


def test_split_strategies():
    assert split_strategy_for("schirrmeister2017") == "run"
    for session_ds in ("iv2a", "openbmi", "cho2017", "dreyer2023"):
        assert split_strategy_for(session_ds) == "session"


def test_schirrmeister_excludes_cz_everywhere():
    assert excludes_cz("schirrmeister2017") is True
    assert not excludes_cz("iv2a")
    # And the reference set must drop cz_ref for Schirrmeister.
    assert "cz_ref" not in reference_modes_for_dataset("schirrmeister2017")
    assert "cz_ref" in reference_modes_for_dataset("iv2a")


def test_openbmi_has_bad_subject():
    assert 29 in spec("openbmi").bad_subjects
