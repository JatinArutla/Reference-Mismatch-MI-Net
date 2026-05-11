"""MOABB / braindecode workarounds, kept in one module so the rest of the
codebase stays library-faithful.
"""

from __future__ import annotations


def make_openbmi_dataset():
    """Lee2019_MI configured to return 200 trials/subject (calibration runs,
    both sessions) instead of MOABB's default 100.

    MOABB 1.5.0's Lee2019.__init__ stores sessions=(1, 2) as _selected_sessions
    for a filter in BaseDataset.get_data, but _get_single_subject_data writes
    session keys as zero-indexed strings ('0', '1'). The filter then drops every
    key not in {'1', '2'}, silently throwing away session '0'. Setting
    _selected_sessions=None bypasses the filter.

    test_run=False follows the MOABB benchmark paper protocol: the test phase
    trials use cued direction during real-time feedback rather than subject
    intent, and aren't reliable for classification.
    """
    from moabb.datasets import Lee2019_MI

    ds = Lee2019_MI()
    ds._selected_sessions = None
    return ds


def make_braindecode_dataset(dataset_id: str, subject: int):
    """braindecode dataset for one subject. OpenBMI takes the assemble-by-hand
    path because MOABBDataset.__init__ constructs the underlying MOABB dataset
    internally with no way to inject the _selected_sessions=None fix.
    """
    from braindecode.datasets import MOABBDataset

    from refshift.data import _moabb_code
    moabb_code = _moabb_code(dataset_id)

    if dataset_id != "openbmi":
        return MOABBDataset(dataset_name=moabb_code, subject_ids=[int(subject)])

    from braindecode.datasets.base import BaseConcatDataset, RawDataset
    from braindecode.datasets.moabb import fetch_data_with_moabb

    moabb_dataset = make_openbmi_dataset()
    raws, description = fetch_data_with_moabb(
        moabb_dataset, subject_ids=[int(subject)],
    )
    return BaseConcatDataset([
        RawDataset(raw, row)
        for raw, (_, row) in zip(raws, description.iterrows())
    ])
