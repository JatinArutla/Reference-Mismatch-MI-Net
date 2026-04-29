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
