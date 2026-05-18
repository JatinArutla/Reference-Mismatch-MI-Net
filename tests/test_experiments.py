"""Unit tests for refshift.experiments helpers (synthetic-only, no network)."""

from __future__ import annotations

import pytest

pytest.importorskip("moabb")


def test_resolve_dataset_excludes_openbmi_subject_29():
    """OpenBMI subject 29 has a corrupt .mat in the GigaDB release; the
    default subject_list returned by ``resolve_dataset`` must omit it so
    users don't accidentally hit it via ``subjects=None``."""
    from refshift.experiments._datasets import resolve_dataset
    ds, _ = resolve_dataset("openbmi")
    assert 29 not in ds.subject_list
    assert len(ds.subject_list) == 53
    assert 1 in ds.subject_list
    assert 54 in ds.subject_list


def test_resolve_dataset_iv2a_unfiltered():
    """No known-bad subjects on IV-2a; full 1-9 list returned."""
    from refshift.experiments._datasets import resolve_dataset
    ds, _ = resolve_dataset("iv2a")
    assert sorted(ds.subject_list) == list(range(1, 10))


def test_resolve_dataset_openbmi_uses_compat_shim():
    """OpenBMI dataset must come back configured with the session-filter
    bypass (200 trials/subject — calibration runs from both sessions).

    See refshift.compat.make_openbmi_dataset for the rationale; this test
    verifies the wiring through resolve_dataset.
    """
    from refshift.experiments._datasets import resolve_dataset
    ds, _ = resolve_dataset("openbmi")
    assert ds.train_run is True
    assert ds.test_run is False  # MOABB benchmark protocol; no test phase
    assert ds._selected_sessions is None


def test_resolve_dataset_unknown_id_raises():
    from refshift.experiments._datasets import resolve_dataset
    with pytest.raises(ValueError, match="Unknown dataset_id"):
        resolve_dataset("not_a_dataset")


def test_resolve_dataset_schirrmeister2017():
    """Schirrmeister2017: 4-class MI, ~14 subjects, single session with
    natural train/test run split. No compatibility shim required."""
    from refshift.experiments._datasets import resolve_dataset
    from moabb.datasets import Schirrmeister2017
    ds, paradigm = resolve_dataset("schirrmeister2017")
    assert isinstance(ds, Schirrmeister2017)
    # 4 events: left_hand, right_hand, feet, rest
    assert paradigm.n_classes == 4


def test_schirrmeister_motor_channels_subset_used():
    """Schirrmeister paradigm restricted to motor-cortex subset (~44 channels,
    matching Schirrmeister 2017 Section 2.7.1) instead of full 128."""
    from refshift.experiments._datasets import (
        resolve_dataset,
        SCHIRRMEISTER_MOTOR_CHANNELS,
    )
    _, paradigm = resolve_dataset("schirrmeister2017")
    assert paradigm.channels is not None
    assert len(paradigm.channels) == 44
    assert len(paradigm.channels) == len(SCHIRRMEISTER_MOTOR_CHANNELS)
    # Schirrmeister 2017 Section 2.7.1 specifies exactly 44 motor channels
    # (Cz excluded as recording reference); subset must be much smaller
    # than the full 128.
    assert "Cz" not in paradigm.channels
    # Sanity: includes the canonical motor channels
    for required in ("C3", "C4", "FC3", "FC4", "CP3", "CP4"):
        assert required in paradigm.channels


def test_schirrmeister_resamples_to_250_hz():
    """Schirrmeister paradigm resamples to 250 Hz to match IV-2a's rate
    and the canonical HGD pipeline (Schirrmeister 2017 example.py)."""
    from refshift.experiments._datasets import resolve_dataset
    _, paradigm = resolve_dataset("schirrmeister2017")
    assert paradigm.resample == 250.0


def test_get_eeg_channel_names_respects_paradigm_channels():
    """When paradigm.channels is set, get_eeg_channel_names returns that
    subset in *paradigm-supplied* order. MOABB picks with ordered=True
    so the X array has channels in paradigm-supplied order, not raw
    order — the graph must match.

    Without a paradigm, all EEG channels are returned in raw-channel
    order.
    """
    from unittest.mock import MagicMock
    from refshift.experiments._datasets import get_eeg_channel_names

    fake_raw = MagicMock()
    fake_raw.ch_names = ["C3", "Cz", "C4", "Fp1", "Fp2"]
    fake_raw.get_channel_types.return_value = ["eeg"] * 5
    fake_dataset = MagicMock()
    fake_dataset.subject_list = [1]
    fake_dataset.get_data.return_value = {1: {"0": {"0train": fake_raw}}}

    # Without paradigm: all 5 EEG channels in raw order
    assert get_eeg_channel_names(fake_dataset) == ["C3", "Cz", "C4", "Fp1", "Fp2"]

    # With paradigm.channels = subset: returned in paradigm-supplied order
    # (matching MOABB's mne.pick_channels(include=..., ordered=True))
    fake_paradigm = MagicMock()
    fake_paradigm.channels = ["C4", "C3"]
    assert get_eeg_channel_names(fake_dataset, paradigm=fake_paradigm) == ["C4", "C3"]

    # With paradigm but channels=None: behaves like no paradigm
    fake_paradigm.channels = None
    assert get_eeg_channel_names(fake_dataset, paradigm=fake_paradigm) == [
        "C3", "Cz", "C4", "Fp1", "Fp2"
    ]


def test_split_train_test_run_strategy():
    """Run-based split: '0train' rows -> train, '1test' rows -> test."""
    import numpy as np
    import pandas as pd
    from refshift.experiments._split import split_train_test

    # Synthetic: 6 trials, 4 channels, 100 samples; runs '0train' and '1test'.
    rng = np.random.default_rng(0)
    X = rng.standard_normal((6, 4, 100)).astype(np.float32)
    y = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
    metadata = pd.DataFrame({
        "session": ["0"] * 6,
        "run": ["0train", "0train", "0train", "1test", "1test", "1test"],
    })

    Xtr, ytr, Xte, yte = split_train_test(
        X, y, metadata, strategy="auto", dataset_id="schirrmeister2017",
    )
    assert Xtr.shape == (3, 4, 100)
    assert Xte.shape == (3, 4, 100)
    np.testing.assert_array_equal(ytr, [0, 1, 0])
    np.testing.assert_array_equal(yte, [1, 0, 1])


def test_split_train_test_run_strategy_explicit():
    """Explicit strategy='run' works for any dataset, not just registered ones."""
    import numpy as np
    import pandas as pd
    from refshift.experiments._split import split_train_test

    X = np.zeros((4, 2, 10), dtype=np.float32)
    y = np.array([0, 1, 0, 1])
    metadata = pd.DataFrame({
        "session": ["0"] * 4,
        "run": ["A", "A", "B", "B"],
    })
    Xtr, ytr, Xte, yte = split_train_test(X, y, metadata, strategy="run")
    # 'A' sorts before 'B' -> A is train, B is test
    assert Xtr.shape == (2, 2, 10)
    np.testing.assert_array_equal(ytr, [0, 1])
    np.testing.assert_array_equal(yte, [0, 1])


def test_split_train_test_session_strategy_unchanged():
    """Sanity: pre-existing session-split behaviour is preserved."""
    import numpy as np
    import pandas as pd
    from refshift.experiments._split import split_train_test

    X = np.zeros((4, 2, 10), dtype=np.float32)
    y = np.array([0, 1, 0, 1])
    metadata = pd.DataFrame({
        "session": ["0", "0", "1", "1"],
        "run": ["r"] * 4,
    })
    # Without dataset_id, defaults to session split when 2+ sessions
    Xtr, ytr, Xte, yte = split_train_test(X, y, metadata, strategy="auto")
    assert Xtr.shape == (2, 2, 10)
    np.testing.assert_array_equal(ytr, [0, 1])  # session '0'
    np.testing.assert_array_equal(yte, [0, 1])  # session '1'


# ---------------------------------------------------------------------------
# classes argument: binary-reduction ablation support (v0.13.1)
# ---------------------------------------------------------------------------

def test_resolve_dataset_default_classes_unchanged_iv2a():
    """When classes is None (default), iv2a uses MotorImagery(n_classes=4).
    This is a regression guard: the binary-reduction PR must not
    silently change the default 4-class behaviour."""
    from refshift.experiments._datasets import resolve_dataset
    _, paradigm = resolve_dataset("iv2a")
    # MotorImagery records its events list in self.events. With n_classes=4
    # and no explicit events list, events is None and n_classes drives the
    # selection.
    assert getattr(paradigm, "n_classes", None) == 4
    # If MOABB ever adds an events attribute that's set under n_classes=4,
    # this would catch it. For now we just assert n_classes.


def test_resolve_dataset_classes_binary_iv2a():
    """Passing classes=('left_hand', 'right_hand') to iv2a builds a
    paradigm with explicit events instead of n_classes=4.

    Also verifies n_classes is set to match events length: MOABB's
    ``MotorImagery.used_events`` compares ``len(out) < self.n_classes``
    and crashes with TypeError if n_classes is None when events is set.
    Always passing both is the workaround.
    """
    pytest.importorskip("moabb")
    from refshift.experiments._datasets import resolve_dataset
    _, paradigm = resolve_dataset(
        "iv2a", classes=("left_hand", "right_hand"),
    )
    assert paradigm.events == ["left_hand", "right_hand"]
    assert paradigm.n_classes == 2  # MOABB workaround


def test_resolve_dataset_classes_binary_schirrmeister():
    """Same for schirrmeister2017: classes=('left_hand','right_hand')
    produces a 2-class paradigm while preserving channel and resample
    settings, and n_classes matches len(events) (MOABB workaround)."""
    pytest.importorskip("moabb")
    from refshift.experiments._datasets import resolve_dataset, SCHIRRMEISTER_MOTOR_CHANNELS
    _, paradigm = resolve_dataset(
        "schirrmeister2017", classes=("left_hand", "right_hand"),
    )
    assert paradigm.events == ["left_hand", "right_hand"]
    assert paradigm.n_classes == 2  # MOABB workaround
    # Critical: channel selection and resample must survive the classes branch.
    assert tuple(paradigm.channels) == tuple(SCHIRRMEISTER_MOTOR_CHANNELS)
    assert paradigm.resample == 250.0


def test_resolve_dataset_classes_unknown_label_raises():
    """Passing a label not in the dataset's class set raises ValueError
    with a clear message."""
    from refshift.experiments._datasets import resolve_dataset
    with pytest.raises(ValueError, match="Unknown classes"):
        resolve_dataset("iv2a", classes=("left_hand", "not_a_real_class"))


def test_resolve_dataset_classes_singleton_raises():
    """A single-class subset isn't a classification task; raise."""
    from refshift.experiments._datasets import resolve_dataset
    with pytest.raises(ValueError, match="fewer than 2"):
        resolve_dataset("iv2a", classes=("left_hand",))


def test_resolve_dataset_classes_empty_raises():
    """An empty class subset is rejected with a different message
    pointing at the right fix (pass None for default)."""
    from refshift.experiments._datasets import resolve_dataset
    with pytest.raises(ValueError, match="empty"):
        resolve_dataset("iv2a", classes=())


def test_resolve_dataset_classes_rejects_invalid_for_lr_paradigm():
    """LeftRightImagery datasets only contain left_hand and right_hand;
    asking for 'feet' on cho2017 must raise rather than silently produce
    an empty paradigm."""
    pytest.importorskip("moabb")
    from refshift.experiments._datasets import resolve_dataset
    with pytest.raises(ValueError, match="Unknown classes"):
        resolve_dataset("cho2017", classes=("left_hand", "feet"))


def test_resolve_dataset_classes_lr_default_pair_is_noop():
    """Passing the LeftRightImagery datasets' own class set is a no-op,
    not an error. This lets users write portable binary-reduction code
    without dataset-specific branching."""
    pytest.importorskip("moabb")
    from refshift.experiments._datasets import resolve_dataset
    # Should not raise:
    _, paradigm = resolve_dataset(
        "cho2017", classes=("left_hand", "right_hand"),
    )
    # LeftRightImagery's class set is fixed; we just verify the call worked.
    assert paradigm is not None


def test_load_dl_data_classes_param_in_cache_key():
    """classes must be part of the cache key so 2-class and 4-class entries
    don't collide for the same (subject, preprocess params)."""
    from refshift.data import _CACHE_KEY_PARAMS
    assert "classes" in _CACHE_KEY_PARAMS


def test_resolve_classes_full_default():
    """resolve_classes(dataset_id, None) returns the full class set."""
    from refshift.experiments._datasets import (
        IV2A_CLASSES, LR_CLASSES, SCHIRR_CLASSES, resolve_classes,
    )
    assert resolve_classes("iv2a", None) == IV2A_CLASSES
    assert resolve_classes("schirrmeister2017", None) == SCHIRR_CLASSES
    assert resolve_classes("cho2017", None) == LR_CLASSES


def test_resolve_classes_binary_subset_on_multiclass():
    """For multiclass datasets, ('left_hand','right_hand') is valid."""
    from refshift.experiments._datasets import resolve_classes
    out = resolve_classes("iv2a", ("left_hand", "right_hand"))
    assert out == ("left_hand", "right_hand")
    out = resolve_classes("schirrmeister2017", ("left_hand", "right_hand"))
    assert out == ("left_hand", "right_hand")


def test_resolve_classes_rejects_unknown_class():
    from refshift.experiments._datasets import resolve_classes
    with pytest.raises(ValueError, match="Unknown classes"):
        resolve_classes("iv2a", ("left_hand", "not_a_real_class"))


def test_load_dl_data_classes_changes_cache_key():
    """4-class and 2-class with same other params produce distinct cache paths."""
    import tempfile
    from refshift.data import _cache_path

    base = dict(
        dataset_id="iv2a", subject=1, resample=250.0,
        l_freq=8.0, h_freq=32.0, ems_factor_new=1e-3, ems_init_block_size=1000,
        trial_start_offset_s=0.0, trial_stop_offset_s=0.0,
        pre_ems_reference=None, pre_ems_laplacian_k=4, pre_ems_montage="standard_1005",
    )
    p_4class = dict(base, classes="left_hand,right_hand,feet,tongue")
    p_2class = dict(base, classes="left_hand,right_hand")

    with tempfile.TemporaryDirectory() as tmp:
        path_4 = _cache_path(tmp, p_4class)
        path_2 = _cache_path(tmp, p_2class)
        assert path_4 != path_2, "2-class and 4-class share cache entry (BUG)"


# ---------------------------------------------------------------------------
# Default split strategy + public API surface
# ---------------------------------------------------------------------------

def test_run_pre_ems_diagonal_default_split_strategy_is_auto():
    """Default split_strategy must be 'auto' (matching run_mismatch).
    None would crash split_train_test."""
    import inspect
    from refshift import experiments
    sig = inspect.signature(experiments.run_pre_ems_diagonal)
    assert sig.parameters["split_strategy"].default == "auto"


def test_public_api_exposes_runners():
    """All runners are importable from the package root."""
    import refshift
    for name in (
        "run_mismatch",
        "run_mismatch_jitter",
        "run_lofo_matrix",
        "run_bandpass_mismatch",
        "run_pre_ems_diagonal",
        "calibrate_csp_lda",
        "mismatch_matrix",
    ):
        assert hasattr(refshift, name), f"refshift.{name} missing"
        assert name in refshift.__all__, f"{name} not in refshift.__all__"


# ---------------------------------------------------------------------------
# run_lofo_matrix: smoke + iteration over holdouts
# ---------------------------------------------------------------------------

def test_run_lofo_matrix_iterates_over_holdouts(monkeypatch):
    """run_lofo_matrix calls run_mismatch_jitter once per holdout (condition='lofo')
    and concatenates the resulting frames.
    """
    import pandas as pd
    from refshift.experiments import jitter as jitter_mod
    from refshift.reference import REFERENCE_MODES

    calls: list = []

    def fake_jitter(dataset_id, *, model, condition, holdout_ref,
                    seeds=(0,), subjects=None, progress=True, **kwargs):
        calls.append({
            "dataset_id": dataset_id, "model": model,
            "condition": condition, "holdout_ref": holdout_ref,
        })
        return pd.DataFrame([{
            "dataset": dataset_id, "subject": 1, "seed": 0,
            "condition": condition, "holdout_ref": holdout_ref,
            "train_modes": "stub", "test_ref": holdout_ref,
            "accuracy": 0.5, "kappa": 0.0,
            "n_train": 1, "n_test": 1,
        }])

    monkeypatch.setattr(jitter_mod, "run_mismatch_jitter", fake_jitter)

    out = jitter_mod.run_lofo_matrix(
        "iv2a", model="shallow", seeds=[0], progress=False,
    )
    assert len(calls) == len(REFERENCE_MODES)
    seen_holdouts = sorted(c["holdout_ref"] for c in calls)
    assert seen_holdouts == sorted(REFERENCE_MODES)
    assert all(c["condition"] == "lofo" for c in calls)
    assert len(out) == len(REFERENCE_MODES)


def test_run_lofo_matrix_rejects_unknown_holdout():
    from refshift import experiments
    with pytest.raises(ValueError, match="not in universe"):
        experiments.run_lofo_matrix(
            "iv2a", model="shallow",
            holdout_modes=("not_a_real_mode",),
            progress=False,
        )


# ---------------------------------------------------------------------------
# run_bandpass_mismatch: API shape
# ---------------------------------------------------------------------------

def test_run_bandpass_mismatch_rejects_csp_lda():
    from refshift import experiments
    with pytest.raises(ValueError, match="DL-only"):
        experiments.run_bandpass_mismatch("iv2a", model="csp_lda")


def test_run_bandpass_mismatch_rejects_unknown_reference():
    from refshift import experiments
    with pytest.raises(ValueError, match="REFERENCE_MODES"):
        experiments.run_bandpass_mismatch(
            "iv2a", model="shallow", reference_mode="not_a_mode",
        )


# ---------------------------------------------------------------------------
# DL-runner scaffolding (setup_dl_run, iter_per_subject_dl_jobs)
# ---------------------------------------------------------------------------

def test_setup_dl_run_skips_graph_when_no_spatial_modes():
    """No graph is built if all declared modes are non-spatial (native, car, median)."""
    from unittest.mock import MagicMock, patch
    from refshift.experiments._dl_runner import setup_dl_run

    fake_ds = MagicMock()
    fake_ds.code = "FAKE"
    fake_ds.subject_list = [1, 2]
    fake_paradigm = MagicMock()

    with patch(
        "refshift.experiments._dl_runner.resolve_dataset",
        return_value=(fake_ds, fake_paradigm),
    ):
        ctx = setup_dl_run(
            "iv2a", subjects=None, seeds=[0],
            reference_modes_for_graph=("native", "car", "median"),
            progress=False,
        )
    assert ctx.graph is None
    assert ctx.subjects == [1, 2]
    assert ctx.seeds == [0]
    assert ctx.dataset_code == "FAKE"


def test_setup_dl_run_includes_rest_only_when_requested():
    """include_rest / include_csd in build_graph is True iff 'rest' / 'csd'
    is in the declared modes. Legacy 'laplacian' resolves to lap_small."""
    from unittest.mock import MagicMock, patch
    from refshift.experiments._dl_runner import setup_dl_run

    fake_ds = MagicMock()
    fake_ds.code = "FAKE"
    fake_ds.subject_list = [1]
    fake_paradigm = MagicMock()
    fake_paradigm.channels = [
        "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz",
        "C2", "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz",
        "P2", "POz",
    ]

    with patch(
        "refshift.experiments._dl_runner.resolve_dataset",
        return_value=(fake_ds, fake_paradigm),
    ):
        # Legacy alias 'laplacian' still works; cz_ref triggers graph build.
        ctx = setup_dl_run(
            "iv2a", subjects=None, seeds=[0],
            reference_modes_for_graph=("laplacian", "cz_ref"),
            progress=False,
        )
        assert ctx.graph is not None
        assert ctx.graph.rest_matrix is None
        assert ctx.graph.csd_matrix is None

        ctx2 = setup_dl_run(
            "iv2a", subjects=None, seeds=[0],
            reference_modes_for_graph=("car", "rest"),
            progress=False,
        )
        assert ctx2.graph is not None
        assert ctx2.graph.rest_matrix is not None
        assert ctx2.graph.csd_matrix is None

        # csd in declared modes -> include_csd=True
        ctx3 = setup_dl_run(
            "iv2a", subjects=None, seeds=[0],
            reference_modes_for_graph=("car", "csd"),
            progress=False,
        )
        assert ctx3.graph is not None
        assert ctx3.graph.csd_matrix is not None
        assert ctx3.graph.rest_matrix is None


def test_iter_per_subject_dl_jobs_loads_once_per_subject(monkeypatch):
    """Reload underlying data only when subject changes; subsequent seeds
    reuse the in-memory tensor.
    """
    import numpy as np
    import pandas as pd
    from refshift.experiments._dl_runner import (
        DLRunContext,
        iter_per_subject_dl_jobs,
    )

    load_calls: list = []

    def fake_load_dl_data(dataset_id, subject, **kwargs):
        load_calls.append(int(subject))
        n = 8
        X = np.random.RandomState(int(subject)).standard_normal(
            (n, 22, 100)
        ).astype(np.float32)
        y = np.array([0, 1] * (n // 2), dtype=np.int64)
        meta = pd.DataFrame({
            "session": ["0"] * n,
            "run": ["0"] * n,
            "subject": [int(subject)] * n,
        })
        ch_names = [
            "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz",
            "C2", "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz",
            "P2", "POz",
        ]
        return X, y, meta, 250.0, ch_names

    # iter_per_subject_dl_jobs imports load_dl_data at call time from refshift.data
    import refshift.data as data_mod
    monkeypatch.setattr(data_mod, "load_dl_data", fake_load_dl_data)

    ctx = DLRunContext(
        dataset_id="iv2a",
        dataset_code="FAKE",
        subjects=[1, 2],
        seeds=[0, 1, 2],
        graph=None,
    )
    yielded = list(iter_per_subject_dl_jobs(
        ctx, split_strategy="stratify", progress=False,
    ))
    assert len(yielded) == 6
    assert load_calls == [1, 2], f"expected [1, 2], got {load_calls}"
    for subject, seed, X_tr, y_tr, X_te, y_te, sfreq in yielded:
        assert X_tr.ndim == 3 and X_te.ndim == 3
        assert sfreq == 250.0


# ---------------------------------------------------------------------------
# Schirrmeister cz_ref handling (v0.14 fix)
# ---------------------------------------------------------------------------

def test_reference_modes_for_dataset_drops_cz_ref_for_schirrmeister():
    from refshift.reference import REFERENCE_MODES, reference_modes_for_dataset
    out = reference_modes_for_dataset("schirrmeister2017")
    assert "cz_ref" not in out
    # Length must drop by exactly one
    assert len(out) == len(REFERENCE_MODES) - 1
    # All other modes preserved in order
    expected = tuple(m for m in REFERENCE_MODES if m != "cz_ref")
    assert out == expected


def test_reference_modes_for_dataset_keeps_cz_ref_for_others():
    from refshift.reference import REFERENCE_MODES, reference_modes_for_dataset
    for ds in ("iv2a", "openbmi", "cho2017", "dreyer2023"):
        out = reference_modes_for_dataset(ds)
        assert "cz_ref" in out, f"cz_ref should remain for {ds}"
        assert out == REFERENCE_MODES


def test_validate_reference_modes_catches_cz_ref_no_cz():
    """Early validation must fire on cz_ref + cz_idx=None, with a clear pointer
    to reference_modes_for_dataset."""
    from refshift.reference import (
        build_graph,
        validate_reference_modes,
    )
    chs_no_cz = ["Fz", "C3", "C4", "Pz"]
    g = build_graph(chs_no_cz, k=2, include_rest=False)
    assert g.cz_idx is None
    with pytest.raises(ValueError, match="cz_ref"):
        validate_reference_modes(("native", "cz_ref"), g, dataset_id="schirrmeister2017")


def test_validate_reference_modes_catches_rest_without_include_rest():
    from refshift.reference import build_graph, validate_reference_modes
    chs = ["Fz", "C3", "Cz", "C4"]
    g = build_graph(chs, k=2, include_rest=False)
    with pytest.raises(ValueError, match="include_rest=True"):
        validate_reference_modes(("native", "rest"), g)


def test_run_mismatch_jitter_full_drops_cz_ref_on_schirrmeister(monkeypatch):
    """run_mismatch_jitter on Schirrmeister with default reference_modes must
    NOT include cz_ref in the train-time sampler universe (would crash at
    transform construction)."""
    import numpy as np
    import pandas as pd
    from unittest.mock import MagicMock

    from refshift.experiments import jitter as jitter_mod
    from refshift.experiments._dl_runner import DLRunContext
    from refshift.reference import build_graph
    from refshift.experiments._datasets import SCHIRRMEISTER_MOTOR_CHANNELS

    real_graph = build_graph(
        list(SCHIRRMEISTER_MOTOR_CHANNELS), k=4, include_rest=True,
    )
    assert real_graph.cz_idx is None  # sanity

    captured_train_modes = {}

    def fake_setup_dl_run(*args, **kwargs):
        return DLRunContext(
            dataset_id="schirrmeister2017",
            dataset_code="HGD",
            subjects=[1],
            seeds=[0],
            graph=real_graph,
        )

    def fake_iter(ctx, *args, **kwargs):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((4, 44, 200)).astype(np.float32)
        y = np.tile(np.array([0, 1], dtype=np.int64), 10)
        yield 1, 0, X, y, X, y, 250.0

    fake_pipe = MagicMock()
    fake_pipe.predict = MagicMock(return_value=np.tile(np.array([0, 1], dtype=np.int64), 10))

    def fake_make_dl_model(*args, **kwargs):
        # Capture the transforms list to verify cz_ref isn't in it
        captured_train_modes["transforms"] = kwargs.get("transforms", [])
        return fake_pipe

    monkeypatch.setattr(jitter_mod, "setup_dl_run", fake_setup_dl_run)
    monkeypatch.setattr(jitter_mod, "iter_per_subject_dl_jobs", fake_iter)

    import refshift.model as model_mod
    monkeypatch.setattr(model_mod, "make_dl_model", fake_make_dl_model)

    df = jitter_mod.run_mismatch_jitter(
        "schirrmeister2017", model="shallow",
        condition="full", progress=False,
    )
    # Must have completed without raising and produced rows
    assert len(df) > 0
    assert "cz_ref" not in df["test_ref"].unique()
    assert "cz_ref" not in df["train_modes"].iloc[0]


def test_run_pre_ems_diagonal_passes_laplacian_k_to_load_dl_data(monkeypatch):
    """Regression for the 'silently ignored args' bug: laplacian_k and montage
    must reach load_dl_data via pre_ems_laplacian_k / pre_ems_montage."""
    import numpy as np
    import pandas as pd
    from unittest.mock import MagicMock

    from refshift.experiments import ems_control as ems_mod

    captured = []

    def fake_load(dataset_id, subject, **kwargs):
        captured.append(kwargs)
        n = 20
        X = np.random.randn(n, 22, 200).astype(np.float32)
        y = np.tile(np.array([0, 1], dtype=np.int64), 10)
        meta = pd.DataFrame({"session": ["0"]*n, "run": ["0"]*n, "subject": [int(subject)]*n})
        return X, y, meta, 250.0, ["Fz","FC3","FC1","FCz","FC2","FC4","C5","C3","C1","Cz","C2","C4","C6","CP3","CP1","CPz","CP2","CP4","P1","Pz","P2","POz"]

    fake_net = MagicMock()
    fake_net.fit = MagicMock(return_value=None)
    fake_net.predict = lambda X: np.zeros(len(X), dtype=np.int64)

    def fake_make_dl(*args, **kwargs):
        return fake_net

    fake_ds = MagicMock()
    fake_ds.code = "FAKE"
    fake_ds.subject_list = [1]

    monkeypatch.setattr(ems_mod, "resolve_dataset", lambda *a, **k: (fake_ds, MagicMock()))
    import refshift.data as data_mod
    monkeypatch.setattr(data_mod, "load_dl_data", fake_load)
    import refshift.model as model_mod
    monkeypatch.setattr(model_mod, "make_dl_model", fake_make_dl)

    ems_mod.run_pre_ems_diagonal(
        "iv2a", model="shallow", subjects=[1], seeds=[0],
        reference_modes=["native"],
        laplacian_k=8, montage="standard_1020",
        progress=False,
    )
    assert len(captured) > 0
    assert captured[0]["pre_ems_laplacian_k"] == 8
    assert captured[0]["pre_ems_montage"] == "standard_1020"


def test_run_pre_ems_diagonal_includes_dataset_column(monkeypatch):
    """Regression for the missing 'dataset' column in run_pre_ems_diagonal output."""
    import numpy as np
    import pandas as pd
    from unittest.mock import MagicMock

    from refshift.experiments import ems_control as ems_mod

    def fake_load(dataset_id, subject, **kwargs):
        n = 20
        X = np.random.randn(n, 22, 200).astype(np.float32)
        y = np.tile(np.array([0, 1], dtype=np.int64), 10)
        meta = pd.DataFrame({"session": ["0"]*n, "run": ["0"]*n, "subject": [int(subject)]*n})
        return X, y, meta, 250.0, ["Fz","FC3","FC1","FCz","FC2","FC4","C5","C3","C1","Cz","C2","C4","C6","CP3","CP1","CPz","CP2","CP4","P1","Pz","P2","POz"]

    fake_net = MagicMock()
    fake_net.fit = MagicMock(return_value=None)
    fake_net.predict = lambda X: np.zeros(len(X), dtype=np.int64)

    fake_ds = MagicMock()
    fake_ds.code = "BNCI2014_001"
    fake_ds.subject_list = [1]

    monkeypatch.setattr(ems_mod, "resolve_dataset", lambda *a, **k: (fake_ds, MagicMock()))
    import refshift.data as data_mod
    monkeypatch.setattr(data_mod, "load_dl_data", fake_load)
    import refshift.model as model_mod
    monkeypatch.setattr(model_mod, "make_dl_model", lambda *a, **k: fake_net)

    df = ems_mod.run_pre_ems_diagonal(
        "iv2a", model="shallow", subjects=[1], seeds=[0],
        reference_modes=["native"], progress=False,
    )
    assert "dataset" in df.columns
    assert (df["dataset"] == "BNCI2014_001").all()
