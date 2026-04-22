"""Kaggle / local environment setup.

Two things:

1. ``setup_kaggle_env()``: sets MNE_DATA, MOABB_RESULTS, thread caps, creates
   cache directories. Idempotent — safe to call multiple times.

2. ``setup_moabb_symlinks()``: symlinks Kaggle input datasets into MOABB's
   expected cache layout so MOABB doesn't re-download. Ported from v2's
   ``_ensure_moabb_cache_symlinks``. Paths are overridable via env vars.

MOABB's own cache layout per dataset:
    $MNE_DATA / MNE-{sign.lower()}-data / ...

Known sign values (from moabb/datasets/{bnci,gigadb,Lee2019,dreyer2023}.py):
    BNCI2014_001  -> sign="BNCI"         (via bnci.utils.bnci_data_path)
    Cho2017       -> sign="Cho2017"      (via Cho2017.data_path)
    Lee2019_MI    -> sign="Lee2019-MI"   (via Lee2019.data_path)
    Dreyer2023    -> sign="Dreyer2023"   (via dreyer2023.data_path)

Cho2017 path drift risk: v2 used "MNE-gigadb-data"; current MOABB may use
"MNE-cho2017-data". If the symlink path is wrong on your MOABB version,
the fallback is just re-download (slow, not broken).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional


# Kaggle input paths used in v2 that are known to contain the raw dataset
# files. Each maps to a MOABB cache destination relative to $MNE_DATA.
# Overridable via env vars for non-Kaggle or non-default setups.
_KAGGLE_SOURCES: Dict[str, Dict[str, str]] = {
    "iv2a": {
        "env_var": "REFSHIFT_IV2A_ROOT",
        "default": "/kaggle/input/datasets/delhialli/four-class-motor-imagery-bnci-001-2014",
        "pattern": "*.mat",
        "moabb_dest": "MNE-bnci-data/~bci/database/001-2014",
    },
    "cho2017": {
        "env_var": "REFSHIFT_CHO2017_ROOT",
        "default": "/kaggle/input/datasets/delhialli/cho2017",
        "pattern": "*.mat",
        # Cho2017 uses sign="GIGADB" in MOABB, so pooch puts files at the URL's
        # path tail under MNE-gigadb-data/. Full destination derived from
        # GIGA_URL = .../gigadb-datasets/live/pub/10.5524/100001_101000/100295/mat_data/
        "moabb_dest": "MNE-gigadb-data/gigadb-datasets/live/pub/10.5524/100001_101000/100295/mat_data",
    },
    "openbmi": {
        "env_var": "REFSHIFT_OPENBMI_ROOT",
        "default": "/kaggle/input/datasets/imaginer369/openbmi-dataset",
        # Lee2019_MI files are nested by session/subject under
        # MNE-lee2019-mi-data/gigadb-datasets/live/pub/10.5524/100001_101000/100542/sessionN/sM/...
        # The Kaggle dataset's internal layout may or may not match this;
        # marked untested. If MOABB re-downloads, that's the fallback.
        "pattern": None,  # link whole subtree
        "moabb_dest": "MNE-lee2019-mi-data",
    },
    "dreyer2023": {
        "env_var": "REFSHIFT_DREYER_ROOT",
        "default": "/kaggle/input/datasets/delhialli/dreyer2023/MNE-Dreyer2023-data",
        # Dreyer is already in MOABB's layout; we link the whole dir tree.
        "pattern": None,
        "moabb_dest": "MNE-Dreyer2023-data",
    },
}


def _link(src: Path, dst: Path) -> bool:
    """Create a symlink from dst -> src. Returns True if a new link was made."""
    if dst.exists() or dst.is_symlink():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(src, dst)
        return True
    except (FileExistsError, OSError):
        return False


def setup_moabb_symlinks(
    datasets: Optional[list] = None,
    *,
    mne_data: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, int]:
    """Symlink Kaggle input files into MOABB's expected cache layout.

    Parameters
    ----------
    datasets : list of str or None
        Which entries from _KAGGLE_SOURCES to process. If None, tries all
        four. Missing source paths are silently skipped (so this is safe to
        call on non-Kaggle machines).
    mne_data : str or None
        Target cache root. Defaults to $MNE_DATA or /kaggle/working/mne_data.
    verbose : bool

    Returns
    -------
    dict of {dataset_id: n_new_symlinks}
        Count of new symlinks created per dataset. 0 means either the
        source wasn't present or the destination already had all files.
    """
    mne_data_path = Path(
        mne_data or os.environ.get("MNE_DATA") or "/kaggle/working/mne_data"
    )
    mne_data_path.mkdir(parents=True, exist_ok=True)
    os.environ["MNE_DATA"] = str(mne_data_path)

    if datasets is None:
        datasets = list(_KAGGLE_SOURCES)

    counts: Dict[str, int] = {}
    for ds_id in datasets:
        if ds_id not in _KAGGLE_SOURCES:
            if verbose:
                print(f"  {ds_id}: unknown dataset id, skipping")
            continue
        entry = _KAGGLE_SOURCES[ds_id]
        src_root = Path(os.environ.get(entry["env_var"], entry["default"]))
        dst_root = mne_data_path / entry["moabb_dest"]

        if not src_root.exists():
            counts[ds_id] = 0
            if verbose:
                print(f"  {ds_id}: source not found ({src_root}); skipping")
            continue

        # Dir-level symlink for tree-layout datasets (Dreyer).
        if entry["pattern"] is None:
            # Symlink the whole dir if destination doesn't exist.
            if dst_root.exists() or dst_root.is_symlink():
                counts[ds_id] = 0
            else:
                dst_root.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.symlink(src_root, dst_root)
                    counts[ds_id] = 1
                except (FileExistsError, OSError):
                    counts[ds_id] = 0
            if verbose:
                status = "linked" if counts[ds_id] else "already linked"
                print(f"  {ds_id}: {status} {src_root} -> {dst_root}")
            continue

        # File-by-file symlinks for flat-layout datasets (IV-2a, Cho2017, OpenBMI).
        n_new = 0
        for f in src_root.glob(entry["pattern"]):
            if _link(f, dst_root / f.name):
                n_new += 1
        counts[ds_id] = n_new
        if verbose:
            n_existing = sum(1 for _ in dst_root.glob(entry["pattern"])) if dst_root.exists() else 0
            print(f"  {ds_id}: +{n_new} new links ({n_existing} total at {dst_root})")

    return counts


def setup_kaggle_env(
    *,
    mne_data: str = "/kaggle/working/mne_data",
    moabb_results: str = "/kaggle/working/moabb_results",
    symlink_datasets: Optional[list] = None,
    thread_cap: int = 1,
    verbose: bool = True,
) -> None:
    """One-call setup for Kaggle notebooks.

    Sets the env vars that MOABB/MNE/BLAS read, creates cache directories,
    and symlinks any Kaggle input datasets into MOABB's expected layout.
    Idempotent; safe to call more than once.

    Parameters
    ----------
    mne_data, moabb_results : str
        Where MOABB/MNE should write cached data and evaluation results.
        Defaults target /kaggle/working which is writable.
    symlink_datasets : list of str or None
        Datasets to symlink from Kaggle input. None -> try all four known.
    thread_cap : int
        Value for OMP_NUM_THREADS and MKL_NUM_THREADS. 1 matches the old
        v2 setup (deterministic, avoids thread-oversubscription on Kaggle's
        shared CPUs).
    verbose : bool

    Example
    -------
    >>> from refshift import setup_kaggle_env
    >>> setup_kaggle_env()
    """
    os.environ["MNE_DATA"] = mne_data
    os.environ["MOABB_RESULTS"] = moabb_results
    os.environ["OMP_NUM_THREADS"] = str(thread_cap)
    os.environ["MKL_NUM_THREADS"] = str(thread_cap)
    os.environ["PYTHONHASHSEED"] = "0"
    Path(mne_data).mkdir(parents=True, exist_ok=True)
    Path(moabb_results).mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"MNE_DATA       = {mne_data}")
        print(f"MOABB_RESULTS  = {moabb_results}")
        print(f"thread cap     = {thread_cap}")
        print("Symlinking Kaggle input datasets into MOABB cache layout:")

    setup_moabb_symlinks(
        datasets=symlink_datasets,
        mne_data=mne_data,
        verbose=verbose,
    )
