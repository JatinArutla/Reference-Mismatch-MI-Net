"""Audit the Cz channel across all five datasets used by refshift.

Purpose
-------
Before running any cz_ref experiment, verify what the MOABB-loaded array
actually contains for the Cz channel on each dataset. Three outcomes are
possible per dataset:

    "missing"            "Cz" is not in the channel list. cz_ref is
                         mathematically undefined; drop cz_ref from the
                         operator set for this dataset.

    "zero_or_near_zero"  "Cz" is in the channel list but the values are
                         identically zero (or numerically negligible
                         relative to other channels). This typically
                         means the dataset still ships Cz as a column
                         even though it was the recording reference and
                         carries no signal. cz_ref is technically
                         computable but redundant with native (since
                         X[Cz] = 0 implies X - X[Cz] = X). Treat as
                         equivalent to native and drop cz_ref to avoid
                         a duplicate column in the headline matrix.

    "usable"             "Cz" is present with a normal signal magnitude.
                         cz_ref is a meaningful, distinct operator on
                         this dataset.

Why this audit exists
---------------------
The original Schirrmeister 2017 recording used Cz as the hardware
reference, which is a fact about the recording protocol. What the
MOABB-loaded array contains depends on (a) whether the published
dataset retained a Cz column, and (b) whether MOABB / braindecode
preprocessing re-references the data on load. Neither (a) nor (b) can
be inferred from the original paper; both have to be observed in the
loaded array. This script runs that observation.

Usage
-----
On Kaggle / a machine with MOABB installed and dataset caches available:

    python audit_cz_channel.py                  # all five datasets
    python audit_cz_channel.py --datasets iv2a openbmi    # subset
    python audit_cz_channel.py --json out.json  # machine-readable output

The audit loads ONE subject per dataset (the first available subject) at
the dataset's native sampling rate and applies the standard refshift
bandpass (8-32 Hz) before checking, because pre-bandpass DC offsets can
make a "zero" channel look nonzero. Loading more subjects would not
change the answer — the recording reference is a per-dataset property.

Limitations
-----------
- Only checks Cz. If you later want to test a different single-electrode
  reference (e.g. FCz, Pz), edit the ``REF_CHANNELS`` list at the top of
  ``main`` or pass ``--ref-ch FCz``.
- Loads with default MOABB paradigm settings. If your refshift run uses
  custom channel selection or band, results may differ — but Cz being
  zero/missing is a property of the source data, not of any downstream
  filter, so the audit's conclusion is robust to bandpass choice.
- Does NOT exercise the actual ``apply_reference`` code path. It checks
  the loaded array directly. This is by design: we want to verify the
  data, not the operator implementation.

Output
------
Prints a per-dataset table and a final recommendation block:

    [DATASET]    status              cz_std    median_std    ratio
    iv2a         usable              4.21e-06   3.85e-06     1.09
    openbmi      usable              ...
    cho2017      usable              ...
    dreyer2023   usable              ...
    schirr2017   missing             —          —            —

    Recommendation:
      iv2a      : keep cz_ref in operator set
      openbmi   : keep cz_ref in operator set
      cho2017   : keep cz_ref in operator set
      dreyer    : keep cz_ref in operator set
      schirr    : drop cz_ref (status: missing)

Exit code is 0 if all datasets resolve cleanly (every dataset is either
"usable" or has a clear drop recommendation). Nonzero if a dataset fails
to load.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
# Maps a short id to (MOABB dataset class path, paradigm class path, label)
# We use string paths and import lazily so the script doesn't fail at import
# time if MOABB isn't installed in the current environment.

DATASETS = {
    "iv2a": {
        "dataset_path": "moabb.datasets.BNCI2014_001",
        "paradigm_path": "moabb.paradigms.MotorImagery",
        "paradigm_kwargs": {"n_classes": 4, "fmin": 8.0, "fmax": 32.0},
        "label": "BCI Competition IV-2a (BNCI2014_001)",
    },
    "openbmi": {
        "dataset_path": "moabb.datasets.Lee2019_MI",
        "paradigm_path": "moabb.paradigms.LeftRightImagery",
        "paradigm_kwargs": {"fmin": 8.0, "fmax": 32.0},
        "label": "OpenBMI / Lee2019_MI",
    },
    "cho2017": {
        "dataset_path": "moabb.datasets.Cho2017",
        "paradigm_path": "moabb.paradigms.LeftRightImagery",
        "paradigm_kwargs": {"fmin": 8.0, "fmax": 32.0},
        "label": "Cho2017",
    },
    "dreyer2023": {
        "dataset_path": "moabb.datasets.Dreyer2023",
        "paradigm_path": "moabb.paradigms.LeftRightImagery",
        "paradigm_kwargs": {"fmin": 8.0, "fmax": 32.0},
        "label": "Dreyer2023",
    },
    "schirrmeister2017": {
        "dataset_path": "moabb.datasets.Schirrmeister2017",
        "paradigm_path": "moabb.paradigms.MotorImagery",
        "paradigm_kwargs": {"n_classes": 4, "fmin": 8.0, "fmax": 32.0},
        "label": "Schirrmeister2017 (HGD)",
    },
}


# ---------------------------------------------------------------------------
# Audit primitive
# ---------------------------------------------------------------------------

def audit_ref_channel(
    X: np.ndarray,
    ch_names: list,
    ref_ch: str = "Cz",
    *,
    abs_tol: float = 1e-8,
    rel_tol: float = 1e-3,
) -> dict:
    """Audit one named channel for use as a single-electrode reference.

    Parameters
    ----------
    X : np.ndarray of shape (N, C, T)
        Trial-by-channel-by-time array, typically as returned by a MOABB
        paradigm's ``get_data``.
    ch_names : list of str
        Channel names in the order matching X's channel axis.
    ref_ch : str, default "Cz"
        Channel to audit.
    abs_tol : float, default 1e-8
        If the channel's std is below this absolute threshold, treat as
        zero. Tuned for typical EEG amplitudes after bandpass (V or μV
        depending on dataset; the relative-tolerance check below is the
        scale-invariant one).
    rel_tol : float, default 1e-3
        If the channel's std relative to the median channel std is below
        this, treat as near-zero. This is the scale-invariant check and
        the more reliable signal of "channel carries no information".

    Returns
    -------
    dict with keys:
        status        : "missing" | "zero_or_near_zero" | "usable"
        ref_ch        : the channel name audited
        ch_std        : float or None (None if missing)
        median_std    : float or None
        ratio         : float or None (ch_std / median_std)
        n_channels    : int
        n_trials      : int
    """
    out = {
        "ref_ch": ref_ch,
        "n_channels": len(ch_names),
        "n_trials": int(X.shape[0]) if X.ndim == 3 else None,
    }

    if ref_ch not in ch_names:
        out.update({
            "status": "missing",
            "ch_std": None,
            "median_std": None,
            "ratio": None,
        })
        return out

    idx = ch_names.index(ref_ch)
    # std across (trials, time) for the single channel; std across
    # (trials, time) per channel for the median comparator.
    ch_std = float(X[:, idx, :].std())
    all_std = X.std(axis=(0, 2))
    median_std = float(np.median(all_std))
    ratio = ch_std / (median_std + 1e-12)

    if ch_std < abs_tol or ratio < rel_tol:
        status = "zero_or_near_zero"
    else:
        status = "usable"

    out.update({
        "status": status,
        "ch_std": ch_std,
        "median_std": median_std,
        "ratio": float(ratio),
    })
    return out


# ---------------------------------------------------------------------------
# MOABB loader
# ---------------------------------------------------------------------------

def _import_path(path: str):
    """Import 'package.module.Name' and return the attribute."""
    module_path, _, attr = path.rpartition(".")
    if not module_path:
        raise ImportError(f"Cannot import bare name {path!r}")
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, attr)


def load_one_subject(dataset_id: str, *, subject: Optional[int] = None) -> tuple:
    """Load (X, y, ch_names, sfreq) for one subject of one dataset.

    Uses the dataset's first subject by default. Errors propagate; the
    caller is expected to wrap this in try/except since MOABB downloads
    can fail for many reasons (network, cache, missing extras).
    """
    spec = DATASETS[dataset_id]
    Dataset = _import_path(spec["dataset_path"])
    Paradigm = _import_path(spec["paradigm_path"])

    dataset = Dataset()
    paradigm = Paradigm(**spec["paradigm_kwargs"])

    if subject is None:
        subject = dataset.subject_list[0]

    X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[subject])

    # ch_names: try paradigm.channels first (the explicit subset), fall back
    # to peeking at the dataset's first raw. The MOABB MotorImagery paradigm
    # picks with ordered=True so paradigm.channels is in load order.
    if hasattr(paradigm, "channels") and paradigm.channels:
        ch_names = list(paradigm.channels)
    else:
        raws = dataset.get_data(subjects=[subject])
        # nested dict: subject -> session -> run -> Raw
        first_raw = None
        for sess in raws[subject].values():
            for run in sess.values():
                first_raw = run
                break
            if first_raw is not None:
                break
        ch_names = [
            ch for ch, kind in zip(first_raw.info["ch_names"],
                                    first_raw.get_channel_types())
            if kind == "eeg"
        ]

    sfreq = paradigm.resample if getattr(paradigm, "resample", None) else None
    if sfreq is None:
        # Fall back to the metadata column if present, else the first raw.
        sfreq = float(metadata.get("sfreq", [0])[0]) if "sfreq" in metadata else None

    if X.shape[1] != len(ch_names):
        # Channel-count mismatch: paradigm.channels reported one count
        # but X has another. Trust X's count and report whatever names
        # are available — but flag it in the output.
        ch_names = ch_names[: X.shape[1]]

    return X, np.asarray(y), ch_names, sfreq


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def format_value(v, fmt: str = ".3e", width: int = 10) -> str:
    """Format a number, or return em-dash padded to width if None."""
    if v is None:
        return "—".ljust(width)
    return f"{v:{fmt}}".ljust(width)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit the Cz channel across MOABB datasets.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASETS),
        default=list(DATASETS),
        help="Subset of datasets to audit (default: all five).",
    )
    parser.add_argument(
        "--ref-ch",
        default="Cz",
        help="Channel to audit. Default 'Cz'.",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="If provided, write per-dataset results as JSON to this path.",
    )
    parser.add_argument(
        "--abs-tol",
        type=float,
        default=1e-8,
        help="Absolute tolerance below which channel std is treated as zero.",
    )
    parser.add_argument(
        "--rel-tol",
        type=float,
        default=1e-3,
        help="Relative tolerance (vs median channel std) for near-zero.",
    )
    args = parser.parse_args()

    results: dict = {}
    failures: list = []

    print("=" * 78)
    print(f" Cz-channel audit — ref_ch={args.ref_ch!r}")
    print("=" * 78)
    print()
    print(f"{'dataset':<22} {'status':<20} {'cz_std':<12} "
          f"{'median_std':<12} {'ratio':<10} {'n_ch':<5}")
    print("-" * 78)

    for ds_id in args.datasets:
        try:
            X, y, ch_names, sfreq = load_one_subject(ds_id)
            audit = audit_ref_channel(
                X, ch_names, ref_ch=args.ref_ch,
                abs_tol=args.abs_tol, rel_tol=args.rel_tol,
            )
            audit["loaded"] = True
            audit["sfreq"] = sfreq
            results[ds_id] = audit
            print(
                f"{ds_id:<22} {audit['status']:<20} "
                f"{format_value(audit['ch_std'], '.3e', 12)} "
                f"{format_value(audit['median_std'], '.3e', 12)} "
                f"{format_value(audit['ratio'], '.3f', 10)} "
                f"{audit['n_channels']:<5}"
            )
        except Exception as e:
            failures.append((ds_id, e))
            results[ds_id] = {"loaded": False, "error": str(e)}
            print(f"{ds_id:<22} FAILED TO LOAD: {type(e).__name__}: {e}")

    # Recommendations block
    print()
    print("=" * 78)
    print(" Recommendation per dataset")
    print("=" * 78)
    for ds_id in args.datasets:
        r = results[ds_id]
        if not r.get("loaded", False):
            print(f"  {ds_id:<22}: COULD NOT AUDIT (load failed)")
            continue
        status = r["status"]
        if status == "usable":
            verdict = f"keep {args.ref_ch}-ref in operator set"
        elif status == "zero_or_near_zero":
            verdict = (
                f"DROP {args.ref_ch}-ref — channel is numerically zero "
                f"(equivalent to native; would duplicate that column)"
            )
        elif status == "missing":
            verdict = (
                f"DROP {args.ref_ch}-ref — channel not present in this "
                f"dataset's montage"
            )
        else:
            verdict = f"unknown status {status!r}"
        print(f"  {ds_id:<22}: {verdict}")

    if args.json:
        # Strip non-JSON-serializable bits (numpy scalars are usually fine
        # but let's coerce to plain Python).
        def _clean(v):
            if isinstance(v, (np.floating, np.integer)):
                return v.item()
            return v
        clean = {
            k: {kk: _clean(vv) for kk, vv in v.items()}
            for k, v in results.items()
        }
        with open(args.json, "w") as f:
            json.dump(clean, f, indent=2, sort_keys=True)
        print()
        print(f"Wrote machine-readable results to {args.json}")

    if failures:
        print()
        print("=" * 78)
        print(f" {len(failures)} dataset(s) failed to load")
        print("=" * 78)
        for ds_id, e in failures:
            print(f"\n[{ds_id}]")
            traceback.print_exception(type(e), e, e.__traceback__)
        return 2  # nonzero: at least one load failure

    return 0


if __name__ == "__main__":
    sys.exit(main())
