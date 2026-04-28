"""MOABB / braindecode compatibility shims.

Functions here exist to work around upstream issues in MOABB or braindecode
at the versions we pin. They poke library internals; keeping all such
pokes in one module lets the rest of the codebase stay library-faithful.

When upstream releases fix the underlying issues, the functions here can
be removed and the call sites simplified. Each function below documents
what to look for upstream.
"""

from __future__ import annotations


def make_openbmi_dataset():
    """Build a ``Lee2019_MI`` MOABB dataset configured to match the MOABB
    benchmark protocol (200 motor-imagery trials per subject, both sessions,
    calibration runs only).

    Two issues with the default ``Lee2019_MI()`` need addressing in MOABB
    1.5.0:

    1. **Session-label filter inconsistency.** ``Lee2019.__init__`` stores
       the user-facing ``sessions=(1, 2)`` as ``_selected_sessions`` for
       a filter in ``BaseDataset.get_data``, but
       ``_get_single_subject_data`` writes session keys as zero-indexed
       strings (``'0'``, ``'1'``). The filter then drops every key not
       in ``{'1', '2'}``, silently throwing away session ``'0'`` on
       every call. We bypass the filter by setting
       ``_selected_sessions = None`` post-construction.

       Without this fix, ``Lee2019_MI()`` returns 100 trials per subject
       (one session, calibration run only). With it, we get 200 trials
       per subject, which matches the MOABB benchmark paper's protocol
       (Table 1 of Chevallier et al. 2024:
       https://arxiv.org/abs/2404.15319 — "Lee2019_MI: 50 trials/class,
       2 classes, 2 sessions" = 200/subject).

    2. **Test-phase runs are excluded.** MOABB's own docstring warns:
       *"test_run for MI and SSVEP do not have labels associated with
       trials: these runs could not be used in classification tasks."*
       Even though the labels technically exist in the .mat files, they
       reflect the cued direction during real-time classifier feedback,
       not the subject's intent. The MOABB benchmark paper deliberately
       excludes the test phase. We follow that protocol by leaving
       ``test_run=False`` (the default for MI).

    Tracking: remove ``_selected_sessions = None`` once MOABB fixes the
    upstream session-label inconsistency in Lee2019.
    """
    from moabb.datasets import Lee2019_MI

    ds = Lee2019_MI()  # train_run=True, test_run=None (-> False for MI)
    ds._selected_sessions = None
    return ds


def make_braindecode_dataset(dataset_id: str, subject: int):
    """Construct a braindecode dataset for one subject.

    For most datasets this is a direct ``MOABBDataset(name, subject_ids=[s])``.
    For OpenBMI we have to bypass that path and assemble the braindecode
    object ourselves because ``MOABBDataset.__init__`` constructs the
    underlying MOABB dataset internally with default args (no way to
    inject our ``_selected_sessions=None`` fix through its public API).
    """
    from braindecode.datasets import MOABBDataset

    from refshift.dl import _moabb_code  # canonical id -> MOABB class name
    moabb_code = _moabb_code(dataset_id)

    if dataset_id != "openbmi":
        return MOABBDataset(dataset_name=moabb_code, subject_ids=[int(subject)])

    # OpenBMI: build the configured MOABB dataset, then assemble the same
    # braindecode object that MOABBDataset would have constructed.
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
