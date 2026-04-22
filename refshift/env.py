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
import re


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
    # openbmi is handled by _setup_openbmi_symlinks (per-file path rewriting;
    # source files are flat sess{SS}_subj{NN}_EEG_MI.mat but MOABB expects
    # them nested under session{S}/s{N}/).
    # dreyer2023 is handled by _setup_dreyer_symlinks (mirror-tree: real
    # writable directories with leaf files symlinked to the read-only Kaggle
    # source; necessary because MOABB's loader writes tempfiles inside the
    # cache dir via pooch, which fails on a symlink pointing into /kaggle/input).
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

    # Clean up stale whole-directory symlinks from older refshift versions.
    # These pointed at /kaggle/input/... (read-only), which breaks MOABB as
    # soon as it tries to write sidecar files or tempfiles inside the cache dir.
    for stale_name in ("MNE-lee2019-mi-data", "MNE-Dreyer2023-data"):
        stale = mne_data_path / stale_name
        if stale.is_symlink():
            try:
                stale.unlink()
                if verbose:
                    print(f"  cleaned up stale symlink: {stale}")
            except OSError:
                pass

    if datasets is None:
        datasets = list(_KAGGLE_SOURCES) + ["openbmi", "dreyer2023"]

    counts: Dict[str, int] = {}
    for ds_id in datasets:
        if ds_id == "openbmi":
            counts[ds_id] = _setup_openbmi_symlinks(mne_data_path, verbose=verbose)
            continue
        if ds_id == "dreyer2023":
            counts[ds_id] = _setup_dreyer_symlinks(mne_data_path, verbose=verbose)
            continue
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

        # File-by-file symlinks for flat-layout datasets (IV-2a, Cho2017).
        n_new = 0
        for f in src_root.glob(entry["pattern"]):
            if _link(f, dst_root / f.name):
                n_new += 1
        counts[ds_id] = n_new
        if verbose:
            n_existing = sum(1 for _ in dst_root.glob(entry["pattern"])) if dst_root.exists() else 0
            print(f"  {ds_id}: +{n_new} new links ({n_existing} total at {dst_root})")

    return counts


# OpenBMI needs per-file path rewriting: flat source filenames
# (sess{SS}_subj{NN}_EEG_MI.mat) -> MOABB's nested layout
# (session{S}/s{N}/sess{SS}_subj{NN}_EEG_MI.mat).
_OPENBMI_DEST_SUBDIR = (
    "MNE-lee2019-mi-data/gigadb-datasets/live/pub/"
    "10.5524/100001_101000/100542"
)
_OPENBMI_FNAME_RE = re.compile(r"^sess(\d{2})_subj(\d{2})_EEG_MI\.mat$")


def _setup_openbmi_symlinks(mne_data_path: Path, verbose: bool) -> int:
    """Per-file symlinks for OpenBMI (Lee2019_MI).

    Rewrites the flat Kaggle layout into MOABB's nested layout. Returns the
    total count of files present at the expected MOABB paths after linking.
    """
    src_root = Path(os.environ.get(
        "REFSHIFT_OPENBMI_ROOT",
        "/kaggle/input/datasets/imaginer369/openbmi-dataset",
    ))
    dest_root = mne_data_path / _OPENBMI_DEST_SUBDIR

    if not src_root.exists():
        if verbose:
            print(f"  openbmi: source not found ({src_root}); skipping")
        return 0

    n_new = 0
    n_total = 0
    n_skipped = 0
    for f in src_root.iterdir():
        m = _OPENBMI_FNAME_RE.match(f.name)
        if not m:
            n_skipped += 1
            continue
        sess, subj = int(m.group(1)), int(m.group(2))
        dest = dest_root / f"session{sess}" / f"s{subj}" / f.name
        if dest.exists() or dest.is_symlink():
            n_total += 1
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.symlink(f, dest)
            n_new += 1
            n_total += 1
        except OSError:
            pass

    if verbose:
        print(
            f"  openbmi: +{n_new} new links ({n_total}/108 total under "
            f"{mne_data_path}/MNE-lee2019-mi-data/)"
        )
        if n_skipped:
            print(f"    ({n_skipped} source files did not match expected pattern)")
    return n_total


def _setup_dreyer_symlinks(mne_data_path: Path, verbose: bool) -> int:
    """Mirror-tree symlinks for Dreyer2023 + zip file placeholders +
    monkey-patch MOABB's download_by_subject to skip its unzip loop.

    The Kaggle dataset has:
      - dreyer2023_manifest.tsv at the root
      - 87 unpacked sub-NN/ BIDS trees with the full EDF + JSON + events.tsv set
      - no sub-NN.zip archives (and we don't need them at runtime)

    Why the monkey-patch: MOABB's download_by_subject has an unzip loop that
    iterates every .zip entry in the manifest and tries to unpack any whose
    target directory doesn't exist. The manifest can contain non-subject
    zip entries (code.zip, sourcedata.zip, etc.) whose target dirs aren't
    in the Kaggle dataset; previous attempts to pre-create placeholder
    directories for these proved fragile across manifest variations. Since
    all subject EEG data is already unpacked via the mirror-tree symlinks,
    the unzip loop is never needed — patching it out is more robust than
    trying to predict every manifest entry.

    The monkey-patch preserves the download phase, so any top-level BIDS
    metadata files (README, dataset_description.json, participants.tsv)
    that aren't in the Kaggle dataset still get downloaded to the writable
    cache on first call (~15 KB total, one-time).
    """
    src_root = Path(os.environ.get(
        "REFSHIFT_DREYER_ROOT",
        "/kaggle/input/datasets/delhialli/dreyer2023/MNE-Dreyer2023-data",
    ))
    dst_root = mne_data_path / "MNE-Dreyer2023-data"

    if not src_root.exists():
        if verbose:
            print(f"  dreyer2023: source not found ({src_root}); skipping")
        return 0

    # Mirror-tree: real writable directories, symlinked leaf files.
    n_new = 0
    n_total = 0
    for root, _dirs, files in os.walk(src_root):
        rel = Path(root).relative_to(src_root)
        dst_subdir = dst_root / rel
        dst_subdir.mkdir(parents=True, exist_ok=True)
        for f in files:
            src_f = Path(root) / f
            dst_f = dst_subdir / f
            if dst_f.exists() or dst_f.is_symlink():
                n_total += 1
                continue
            try:
                os.symlink(src_f, dst_f)
                n_new += 1
                n_total += 1
            except OSError:
                pass

    # Zero-byte .zip placeholders so MOABB's download phase skips them.
    # (Without these, download_if_missing would attempt real downloads of
    # hundreds-of-MB OSF zip archives for every subject.)
    n_placeholders = 0
    manifest_path = dst_root / "dreyer2023_manifest.tsv"
    if manifest_path.exists():
        try:
            import pandas as pd
            manifest = pd.read_csv(manifest_path, sep="\t")
            fname_col = "filename" if "filename" in manifest.columns else manifest.columns[0]
            for fname in manifest[fname_col]:
                if not isinstance(fname, str) or not fname.endswith(".zip"):
                    continue
                placeholder = dst_root / fname
                placeholder.parent.mkdir(parents=True, exist_ok=True)
                if not placeholder.exists():
                    placeholder.touch()
                    n_placeholders += 1
        except Exception as e:
            if verbose:
                print(f"    dreyer2023: manifest parse failed ({e})")

    # Apply the monkey-patch so MOABB skips its unzip loop.
    _patch_moabb_dreyer_no_unzip(verbose=verbose)

    if verbose:
        print(
            f"  dreyer2023: +{n_new} new file links ({n_total} total), "
            f"+{n_placeholders} zip placeholders under {dst_root}"
        )
    return n_total


def _patch_moabb_dreyer_no_unzip(verbose: bool = True) -> None:
    """Replace ``Dreyer2023.download_by_subject`` with a variant that skips
    the unzip loop. The download loop (for BIDS metadata files listed in
    the manifest) is preserved. Idempotent — safe to call multiple times.
    """
    try:
        from moabb.datasets import dreyer2023 as _dr
        from moabb.datasets import download as _dl
    except ImportError:
        return  # MOABB not installed; nothing to patch

    # If we've already patched, the function will have our sentinel attribute.
    if getattr(_dr.Dreyer2023.download_by_subject, "_refshift_patched", False):
        return

    def download_by_subject(self, subject, path=None):
        """refshift-patched: downloads manifest entries, skips the unzip loop."""
        from pathlib import Path as _Path
        import pandas as _pd
        from tqdm import tqdm as _tqdm

        path = _Path(_dl.get_dataset_path(self.code, path)) / f"MNE-{self.code}-data"

        # Ensure the manifest is present (either symlinked or freshly downloaded).
        _dl.download_if_missing(
            path / "dreyer2023_manifest.tsv", _dr._manifest_link
        )
        manifest = _pd.read_csv(path / "dreyer2023_manifest.tsv", sep="\t")
        subject_index = manifest["filename"] == f"sub-{subject:02d}.zip"
        dataset_index = ~manifest["filename"].str.contains("sub")
        manifest_subject = manifest[subject_index | dataset_index]

        # Download phase — same as MOABB's original.
        for _, row in _tqdm(manifest_subject.iterrows()):
            download_url = _dr._api_base_url + row["url"].replace(
                "https://osf.io/download/", ""
            ).replace("/", "")
            _dl.download_if_missing(
                path / row["filename"], download_url, warn_missing=False
            )

        # Unzip loop deliberately skipped — all subject data is already
        # unpacked via refshift's mirror-tree symlinks.
        return path

    download_by_subject._refshift_patched = True
    _dr.Dreyer2023.download_by_subject = download_by_subject

    if verbose:
        print("    dreyer2023: monkey-patched download_by_subject (skip unzip)")


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
