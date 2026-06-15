"""Tests for the loro/lofo sweep orchestration.

These mock out run_mismatch_jitter so they run without MOABB/torch: we only
check that the sweeps call it with the right train/test modes and tag rows
correctly.
"""

import pandas as pd
import pytest

from refshift import experiments
from refshift.references import REFERENCE_MODES


def _fake_jitter_factory(calls):
    def fake_jitter(dataset_id, *, model, condition, holdout_ref="cz_ref",
                    reference_modes=None, test_reference_modes=None,
                    seeds=(0,), subjects=None, progress=True, **kwargs):
        train = tuple(reference_modes)
        test = tuple(test_reference_modes) if test_reference_modes is not None else train
        calls.append({"dataset_id": dataset_id, "condition": condition,
                      "holdout_ref": holdout_ref, "train": train, "test": test})
        return pd.DataFrame([
            {"subject": 1, "seed": 0, "condition": condition,
             "holdout_ref": holdout_ref if condition == "loro" else "",
             "train_modes": ",".join(train), "test_ref": tr,
             "accuracy": 0.5, "kappa": 0.0, "n_train": 1, "n_test": 1}
            for tr in test
        ])
    return fake_jitter


def test_loro_sweeps_each_reference(monkeypatch):
    calls = []
    monkeypatch.setattr(experiments, "run_mismatch_jitter", _fake_jitter_factory(calls))
    out = experiments.run_loro_matrix("iv2a", model="shallow", seeds=[0], progress=False)
    # One jitter call per reference, each 'loro' with that holdout. (The actual
    # train = universe \ {holdout} removal happens inside run_mismatch_jitter,
    # which is mocked here and tested directly elsewhere.)
    assert len(calls) == len(REFERENCE_MODES)
    assert all(c["condition"] == "loro" for c in calls)
    assert sorted(c["holdout_ref"] for c in calls) == sorted(REFERENCE_MODES)


def test_lofo_holds_out_whole_family(monkeypatch):
    calls = []
    monkeypatch.setattr(experiments, "run_mismatch_jitter", _fake_jitter_factory(calls))
    out = experiments.run_lofo_matrix("iv2a", model="shallow", seeds=[0], progress=False)
    assert len(calls) == 3
    assert all(c["condition"] == "full" for c in calls)
    single = next(c for c in calls if "cz_ref" not in c["train"])
    assert "cz_ref" in single["test"]
    assert set(out["holdout_family"]) == {"global", "single", "spatial"}
    assert set(out["test_family"]) == {"global", "single", "spatial"}
    assert (out["condition"] == "lofo").all()


def test_lofo_rejects_overlapping_families(monkeypatch):
    monkeypatch.setattr(experiments, "run_mismatch_jitter", _fake_jitter_factory([]))
    with pytest.raises(ValueError, match="at most one family"):
        experiments.run_lofo_matrix(
            "iv2a", model="shallow",
            families={"a": ["car"], "b": ["car", "median"]},
            progress=False,
        )


def test_loro_rejects_unknown_holdout(monkeypatch):
    monkeypatch.setattr(experiments, "run_mismatch_jitter", _fake_jitter_factory([]))
    with pytest.raises(ValueError):
        experiments.run_loro_matrix(
            "iv2a", model="shallow", holdout_modes=("not_a_mode",), progress=False,
        )


def test_lofo_drops_cz_ref_for_schirrmeister(monkeypatch):
    # Schirrmeister has no Cz, so the 'single' family (cz_ref) becomes empty
    # and is skipped; only global and spatial are swept.
    calls = []
    monkeypatch.setattr(experiments, "run_mismatch_jitter", _fake_jitter_factory(calls))
    out = experiments.run_lofo_matrix(
        "schirrmeister2017", model="shallow", seeds=[0], progress=False,
    )
    assert set(out["holdout_family"]) == {"global", "spatial"}
    # cz_ref must never appear as a trained or tested mode.
    for c in calls:
        assert "cz_ref" not in c["train"]
        assert "cz_ref" not in c["test"]
