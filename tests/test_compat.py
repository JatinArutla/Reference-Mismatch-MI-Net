"""Unit tests for refshift.compat (MOABB / braindecode workarounds).

Synthetic-only — no MOABB / network. The tests check the *configuration*
of the returned dataset object (which knobs are set), not the loading
behavior, since loading requires real .mat files.
"""

from __future__ import annotations

import pytest

pytest.importorskip("moabb")


def test_make_openbmi_dataset_excludes_test_run():
    """test_run defaults to False for MI per MOABB's Lee2019.__init__:
    `self.test_run = paradigm == "p300" if test_run is None else test_run`.
    The test phase trials don't have reliable labels (real-time feedback
    contamination) and are excluded by the MOABB benchmark paper.
    """
    from refshift.compat import make_openbmi_dataset
    ds = make_openbmi_dataset()
    assert ds.train_run is True
    assert ds.test_run is False


def test_make_openbmi_dataset_disables_session_filter():
    """_selected_sessions=None bypasses the buggy MOABB session filter."""
    from refshift.compat import make_openbmi_dataset
    ds = make_openbmi_dataset()
    assert ds._selected_sessions is None
    # Sanity: the underlying sessions tuple is still set
    assert ds.sessions == (1, 2)


def test_make_openbmi_dataset_returns_lee2019_mi():
    """Type sanity — defends against accidental swap to a different class."""
    from refshift.compat import make_openbmi_dataset
    from moabb.datasets import Lee2019_MI
    ds = make_openbmi_dataset()
    assert isinstance(ds, Lee2019_MI)


def test_make_braindecode_dataset_unknown_dataset_id_raises():
    """Helper validates dataset_id; doesn't silently mis-route."""
    pytest.importorskip("braindecode")
    from refshift.compat import make_braindecode_dataset
    with pytest.raises(ValueError, match="Unknown dataset_id"):
        make_braindecode_dataset("not_a_dataset", subject=1)
