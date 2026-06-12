"""Cross-subject LOSO runner tests with MOABB mocked (synthetic data)."""

import numpy as np
import pandas as pd
import pytest

import refshift.experiments.cross_subject as cs


class _FakeParadigm:
    def __init__(self, store):
        self.store = store
    def get_data(self, dataset=None, subjects=None, **kw):
        s = subjects[0]
        X, y = self.store[s]
        meta = pd.DataFrame({"session": ["0"] * len(y)})  # single session -> stratify
        return X, np.asarray(y), meta


class _FakeDataset:
    code = "FAKE"
    def __init__(self, subs):
        self.subject_list = list(subs)


def _make_store(n_subj=4, per_class=40, c=22, t=200, seed=0):
    rng = np.random.default_rng(seed)
    store = {}
    for s in range(1, n_subj + 1):
        # subject-specific channel mixing so EA actually has work to do
        mix = rng.standard_normal((c, c))
        z = rng.standard_normal((2 * per_class, c, t))
        X = np.einsum("ij,njt->nit", mix, z).astype(np.float32)
        y = np.array(["left_hand", "right_hand"] * per_class)
        store[s] = (X, y)
    return store


@pytest.fixture
def patched(monkeypatch):
    store = _make_store()
    ds = _FakeDataset(store.keys())
    par = _FakeParadigm(store)
    monkeypatch.setattr(cs, "resolve_dataset", lambda *a, **k: (ds, par))
    monkeypatch.setattr(cs, "get_eeg_channel_names",
                        lambda *a, **k: [f"C{i}" for i in range(22)])
    monkeypatch.setattr(cs, "build_cache_config", lambda *a, **k: {})
    # no graph modes -> skip build_graph; use only native+car
    return store


def _run(regime, **kw):
    return cs.run_cross_subject_mismatch(
        "iv2a", ea_regime=regime,
        reference_modes={"native", "car"},
        seeds=[0], cache=False, progress=False, **kw,
    )


def test_source_regime_runs_and_has_expected_shape(patched):
    df = _run("source")
    # 4 targets x 1 seed x 2 train_ref x 2 test_ref = 16 rows
    assert len(df) == 16
    assert set(df["ea_regime"]) == {"source"}
    # source pool = 3 other subjects x train split (80% of 80 = 64) = 192
    assert df["n_source_train"].iloc[0] == 192
    assert df["accuracy"].between(0, 1).all()


def test_within_regime_uses_target_data(patched):
    df = _run("within")
    assert set(df["ea_regime"]) == {"within"}
    # within scores the full target test split (20% of 80 = 16)
    assert df["n_target_test"].iloc[0] == 16


def test_target_k_excludes_calibration_from_scoring(patched):
    df = _run("target_k", ea_target_k=3)
    # target test = 16 trials; 3/class * 2 = 6 calibration -> 10 scored
    assert df["n_target_test"].iloc[0] == 10


def test_regimes_give_different_numbers(patched):
    a = _run("within").set_index(["target_subject","train_ref","test_ref"])["accuracy"]
    b = _run("source").set_index(["target_subject","train_ref","test_ref"])["accuracy"]
    # different R_bar source -> generally different scores (not identical)
    assert not np.allclose(a.values, b.values)


def test_bad_regime_raises(patched):
    with pytest.raises(ValueError):
        _run("bogus")


def test_single_subject_raises(patched):
    with pytest.raises(ValueError):
        cs.run_cross_subject_mismatch(
            "iv2a", ea_regime="source", subjects=[1],
            reference_modes={"native"}, progress=False, cache=False,
        )
