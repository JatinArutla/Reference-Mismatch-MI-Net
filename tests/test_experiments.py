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
