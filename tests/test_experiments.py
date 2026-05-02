"""Unit tests for refshift.experiments helpers (synthetic-only, no network)."""

from __future__ import annotations

import pytest

pytest.importorskip("moabb")


def test_resolve_dataset_excludes_openbmi_subject_29():
    """OpenBMI subject 29 has a corrupt .mat in the GigaDB release; the
    default subject_list returned by ``_resolve_dataset`` must omit it so
    users don't accidentally hit it via ``subjects=None``."""
    from refshift.experiments import _resolve_dataset
    ds, _ = _resolve_dataset("openbmi")
    assert 29 not in ds.subject_list
    assert len(ds.subject_list) == 53
    assert 1 in ds.subject_list
    assert 54 in ds.subject_list


def test_resolve_dataset_iv2a_unfiltered():
    """No known-bad subjects on IV-2a; full 1-9 list returned."""
    from refshift.experiments import _resolve_dataset
    ds, _ = _resolve_dataset("iv2a")
    assert sorted(ds.subject_list) == list(range(1, 10))


def test_resolve_dataset_openbmi_uses_compat_shim():
    """OpenBMI dataset must come back configured with the session-filter
    bypass (200 trials/subject — calibration runs from both sessions).

    See refshift.compat.make_openbmi_dataset for the rationale; this test
    verifies the wiring through _resolve_dataset.
    """
    from refshift.experiments import _resolve_dataset
    ds, _ = _resolve_dataset("openbmi")
    assert ds.train_run is True
    assert ds.test_run is False  # MOABB benchmark protocol; no test phase
    assert ds._selected_sessions is None


def test_resolve_dataset_unknown_id_raises():
    from refshift.experiments import _resolve_dataset
    with pytest.raises(ValueError, match="Unknown dataset_id"):
        _resolve_dataset("not_a_dataset")


def test_resolve_dataset_schirrmeister2017():
    """Schirrmeister2017: 4-class MI, ~14 subjects, single session with
    natural train/test run split. No compatibility shim required."""
    from refshift.experiments import _resolve_dataset
    from moabb.datasets import Schirrmeister2017
    ds, paradigm = _resolve_dataset("schirrmeister2017")
    assert isinstance(ds, Schirrmeister2017)
    # 4 events: left_hand, right_hand, feet, rest
    assert paradigm.n_classes == 4


def test_schirrmeister_motor_channels_subset_used():
    """Schirrmeister paradigm restricted to motor-cortex subset (~44 channels,
    matching Schirrmeister 2017 Section 2.7.1) instead of full 128."""
    from refshift.experiments import (
        _resolve_dataset,
        _SCHIRRMEISTER_MOTOR_CHANNELS,
    )
    _, paradigm = _resolve_dataset("schirrmeister2017")
    assert paradigm.channels is not None
    assert len(paradigm.channels) == 44
    assert len(paradigm.channels) == len(_SCHIRRMEISTER_MOTOR_CHANNELS)
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
    from refshift.experiments import _resolve_dataset
    _, paradigm = _resolve_dataset("schirrmeister2017")
    assert paradigm.resample == 250.0


def test_get_eeg_channel_names_respects_paradigm_channels():
    """When paradigm.channels is set, _get_eeg_channel_names returns that
    subset in *paradigm-supplied* order. MOABB picks with ordered=True
    so the X array has channels in paradigm-supplied order, not raw
    order — the graph must match.

    Without a paradigm, all EEG channels are returned in raw-channel
    order.
    """
    from unittest.mock import MagicMock
    from refshift.experiments import _get_eeg_channel_names

    fake_raw = MagicMock()
    fake_raw.ch_names = ["C3", "Cz", "C4", "Fp1", "Fp2"]
    fake_raw.get_channel_types.return_value = ["eeg"] * 5
    fake_dataset = MagicMock()
    fake_dataset.subject_list = [1]
    fake_dataset.get_data.return_value = {1: {"0": {"0train": fake_raw}}}

    # Without paradigm: all 5 EEG channels in raw order
    assert _get_eeg_channel_names(fake_dataset) == ["C3", "Cz", "C4", "Fp1", "Fp2"]

    # With paradigm.channels = subset: returned in paradigm-supplied order
    # (matching MOABB's mne.pick_channels(include=..., ordered=True))
    fake_paradigm = MagicMock()
    fake_paradigm.channels = ["C4", "C3"]
    assert _get_eeg_channel_names(fake_dataset, paradigm=fake_paradigm) == ["C4", "C3"]

    # With paradigm but channels=None: behaves like no paradigm
    fake_paradigm.channels = None
    assert _get_eeg_channel_names(fake_dataset, paradigm=fake_paradigm) == [
        "C3", "Cz", "C4", "Fp1", "Fp2"
    ]


def test_split_train_test_run_strategy():
    """Run-based split: '0train' rows -> train, '1test' rows -> test."""
    import numpy as np
    import pandas as pd
    from refshift.experiments import _split_train_test

    # Synthetic: 6 trials, 4 channels, 100 samples; runs '0train' and '1test'.
    rng = np.random.default_rng(0)
    X = rng.standard_normal((6, 4, 100)).astype(np.float32)
    y = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
    metadata = pd.DataFrame({
        "session": ["0"] * 6,
        "run": ["0train", "0train", "0train", "1test", "1test", "1test"],
    })

    Xtr, ytr, Xte, yte = _split_train_test(
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
    from refshift.experiments import _split_train_test

    X = np.zeros((4, 2, 10), dtype=np.float32)
    y = np.array([0, 1, 0, 1])
    metadata = pd.DataFrame({
        "session": ["0"] * 4,
        "run": ["A", "A", "B", "B"],
    })
    Xtr, ytr, Xte, yte = _split_train_test(X, y, metadata, strategy="run")
    # 'A' sorts before 'B' -> A is train, B is test
    assert Xtr.shape == (2, 2, 10)
    np.testing.assert_array_equal(ytr, [0, 1])
    np.testing.assert_array_equal(yte, [0, 1])


def test_split_train_test_session_strategy_unchanged():
    """Sanity: pre-existing session-split behaviour is preserved."""
    import numpy as np
    import pandas as pd
    from refshift.experiments import _split_train_test

    X = np.zeros((4, 2, 10), dtype=np.float32)
    y = np.array([0, 1, 0, 1])
    metadata = pd.DataFrame({
        "session": ["0", "0", "1", "1"],
        "run": ["r"] * 4,
    })
    # Without dataset_id, defaults to session split when 2+ sessions
    Xtr, ytr, Xte, yte = _split_train_test(X, y, metadata, strategy="auto")
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
    from refshift.experiments import _resolve_dataset
    _, paradigm = _resolve_dataset("iv2a")
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
    from refshift.experiments import _resolve_dataset
    _, paradigm = _resolve_dataset(
        "iv2a", classes=("left_hand", "right_hand"),
    )
    assert paradigm.events == ["left_hand", "right_hand"]
    assert paradigm.n_classes == 2  # MOABB workaround


def test_resolve_dataset_classes_binary_schirrmeister():
    """Same for schirrmeister2017: classes=('left_hand','right_hand')
    produces a 2-class paradigm while preserving channel and resample
    settings, and n_classes matches len(events) (MOABB workaround)."""
    pytest.importorskip("moabb")
    from refshift.experiments import _resolve_dataset, _SCHIRRMEISTER_MOTOR_CHANNELS
    _, paradigm = _resolve_dataset(
        "schirrmeister2017", classes=("left_hand", "right_hand"),
    )
    assert paradigm.events == ["left_hand", "right_hand"]
    assert paradigm.n_classes == 2  # MOABB workaround
    # Critical: channel selection and resample must survive the classes branch.
    assert tuple(paradigm.channels) == tuple(_SCHIRRMEISTER_MOTOR_CHANNELS)
    assert paradigm.resample == 250.0


def test_resolve_dataset_classes_unknown_label_raises():
    """Passing a label not in the dataset's class set raises ValueError
    with a clear message."""
    from refshift.experiments import _resolve_dataset
    with pytest.raises(ValueError, match="Unknown classes"):
        _resolve_dataset("iv2a", classes=("left_hand", "not_a_real_class"))


def test_resolve_dataset_classes_singleton_raises():
    """A single-class subset isn't a classification task; raise."""
    from refshift.experiments import _resolve_dataset
    with pytest.raises(ValueError, match="fewer than 2"):
        _resolve_dataset("iv2a", classes=("left_hand",))


def test_resolve_dataset_classes_empty_raises():
    """An empty class subset is rejected with a different message
    pointing at the right fix (pass None for default)."""
    from refshift.experiments import _resolve_dataset
    with pytest.raises(ValueError, match="empty"):
        _resolve_dataset("iv2a", classes=())


def test_resolve_dataset_classes_rejects_invalid_for_lr_paradigm():
    """LeftRightImagery datasets only contain left_hand and right_hand;
    asking for 'feet' on cho2017 must raise rather than silently produce
    an empty paradigm."""
    pytest.importorskip("moabb")
    from refshift.experiments import _resolve_dataset
    with pytest.raises(ValueError, match="Unknown classes"):
        _resolve_dataset("cho2017", classes=("left_hand", "feet"))


def test_resolve_dataset_classes_lr_default_pair_is_noop():
    """Passing the LeftRightImagery datasets' own class set is a no-op,
    not an error. This lets users write portable binary-reduction code
    without dataset-specific branching."""
    pytest.importorskip("moabb")
    from refshift.experiments import _resolve_dataset
    # Should not raise:
    _, paradigm = _resolve_dataset(
        "cho2017", classes=("left_hand", "right_hand"),
    )
    # LeftRightImagery's class set is fixed; we just verify the call worked.
    assert paradigm is not None


def test_run_mismatch_classes_rejected_for_dl_path():
    """The DL path doesn't yet support class subsetting; calling it with
    classes= must raise NotImplementedError (not a silent fallback to
    full 4-class data) so users notice the gap."""
    pytest.importorskip("moabb")
    from refshift import run_mismatch
    with pytest.raises(NotImplementedError, match="csp_lda"):
        run_mismatch(
            "iv2a", model="shallow",
            classes=("left_hand", "right_hand"),
            seeds=[0],
        )
