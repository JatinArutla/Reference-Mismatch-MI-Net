"""Phase 2 DL pipelines via braindecode + skorch.

Two architectures are supported: ``eegnet`` (EEGNetv4 with F1=8, D=2 — the
EEGNet_8_2 configuration benchmarked in MOABB) and ``shallow``
(ShallowFBCSPNet from Schirrmeister et al. 2017, the canonical braindecode
model for motor-imagery decoding).

Preprocessing follows braindecode's canonical MOABB example
(``examples/model_building/plot_bcic_iv_2a_moabb_trial.py``) with one
difference: bandpass is 8–32 Hz (MOABB's MI paradigm default, consistent
with Phase 1 CSP+LDA) rather than 4–38 Hz. The pipeline is::

    MOABBDataset -> Preprocessor(pick_types EEG)
                 -> Preprocessor(V->uV scale)
                 -> Preprocessor(filter l_freq..h_freq)
                 -> Preprocessor(exponential_moving_standardize)
                 -> create_windows_from_events
                 -> numpy (N, C, T) tensor

The reference operator is applied to the windowed (N, C, T) tensor by
``run_mismatch`` after this module returns. For the linear reference
operators in ``refshift.reference`` this is numerically equivalent to
applying them in raw-space, so the EMS-before-reference order is
preserved by construction (EMS runs during preprocess).

The model factory returns a ``braindecode.EEGClassifier`` (subclass of
``skorch.NeuralNetClassifier``) so it drops into sklearn-style
``fit(X, y) / predict(X) / score(X, y)`` calls against numpy arrays.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


SUPPORTED_DL_MODELS = ("eegnet", "shallow")


_DATASET_ID_TO_MOABB = {
    "iv2a": "BNCI2014_001",
    "openbmi": "Lee2019_MI",
    "cho2017": "Cho2017",
    "dreyer2023": "Dreyer2023",
    "schirrmeister2017": "Schirrmeister2017",
}


def _scale_volts_to_microvolts(data):
    """Scale a raw ndarray from Volts to microvolts.

    Module-level (not a lambda) so braindecode's Preprocessor can pickle it,
    which (a) silences the "lambda cannot be saved" and "apply_on_array
    auto-correcting" warnings, and (b) allows braindecode's preprocessing
    cache to be reused across runs if the dataset is persisted.

    Multiplying by 1e6 is numerically a no-op against the downstream
    exponential_moving_standardize step (which renormalizes per-channel to
    zero mean, unit variance), but we keep it for consistency with MOABB's
    paradigm path, which applies dataset.unit_factor (= 1e6 for all four
    datasets in this study) before any downstream processing.
    """
    return data * 1e6


def _moabb_code(dataset_id: str) -> str:
    """Map refshift dataset_id -> braindecode MOABBDataset class name."""
    key = dataset_id.lower()
    if key not in _DATASET_ID_TO_MOABB:
        raise ValueError(
            f"Unknown dataset_id: {dataset_id!r}. "
            f"Known: {tuple(_DATASET_ID_TO_MOABB)}"
        )
    return _DATASET_ID_TO_MOABB[key]


# ---------------------------------------------------------------------------
# Disk cache for preprocessed (X, y, metadata, sfreq, ch_names) tensors
# ---------------------------------------------------------------------------
#
# A thin .npz cache keyed on a hash of the preprocessing parameters. The
# point is purely to avoid re-running braindecode's filter + EMS on the
# same Raws when we run multiple architectures / jitter conditions on the
# same dataset. Reference operators are applied AFTER this layer, so all
# 7 reference variants share a single cached entry.
#
# When future preprocessing-mismatch experiments add new knobs to
# load_dl_data (resample target, filter type, window length), add them to
# _CACHE_KEY_PARAMS so each variant gets its own slot.
#
# If a cache file is corrupt, the user deletes the directory and reruns.

_CACHE_KEY_PARAMS = (
    "dataset_id", "subject", "l_freq", "h_freq",
    "ems_factor_new", "ems_init_block_size",
    "trial_start_offset_s", "trial_stop_offset_s",
)


def _cache_path(cache_dir: str, params: dict) -> str:
    """Return ``<cache_dir>/<dataset_id>/sub-<NNN>/<hash>.npz``."""
    import hashlib
    import json
    import os

    relevant = {k: params[k] for k in _CACHE_KEY_PARAMS}
    key = hashlib.sha1(
        json.dumps(relevant, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    subdir = os.path.join(
        cache_dir, params["dataset_id"], f"sub-{int(params['subject']):03d}",
    )
    os.makedirs(subdir, exist_ok=True)
    return os.path.join(subdir, f"{key}.npz")


def load_dl_data(
    dataset_id: str,
    subject: int,
    *,
    l_freq: float = 8.0,
    h_freq: float = 32.0,
    ems_factor_new: float = 1e-3,
    ems_init_block_size: int = 1000,
    trial_start_offset_s: float = 0.0,
    trial_stop_offset_s: float = 0.0,
    preload: bool = True,
    n_jobs: int = 1,
    cache_dir: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, float, List[str]]:
    """Load one subject's MI data in the braindecode + MOABB canonical protocol.

    Steps:
      1. ``MOABBDataset(dataset_name=<resolved>, subject_ids=[subject])``
      2. ``preprocess`` with pick_types EEG, V->uV scale, bandpass,
         exponential_moving_standardize.
      3. ``create_windows_from_events`` with optional trial offsets.
      4. Concatenate windows across all runs/sessions into (X, y, metadata).

    Reference operators are NOT applied here — ``run_mismatch`` applies them
    to X after this returns.

    If ``cache_dir`` is provided, the preprocessed (X, y, metadata, sfreq,
    ch_names) tuple is read from / written to a per-subject ``.npz`` file
    keyed on a hash of all preprocessing parameters. Re-referencing happens
    after this function returns, so all 7 reference variants share one cache
    entry. Filter band, EMS settings, and trial offsets are part of the key,
    so future preprocessing-mismatch experiments (e.g. filter-band mismatch)
    automatically get separate cache slots without code changes.

    Cache misses (file missing, sidecar malformed, parameters disagree,
    format-version stale, or any disk error) silently fall back to the full
    preprocess pipeline and write a fresh entry.

    Parameters
    ----------
    dataset_id : {'iv2a', 'openbmi', 'cho2017', 'dreyer2023'}
    subject : int
        1-indexed (MOABB convention).
    l_freq, h_freq : float
        Bandpass cutoffs. Default 8–32 Hz to match Phase 1 / MOABB MI paradigm.
    ems_factor_new, ems_init_block_size : float, int
        Exponential moving standardization parameters. Defaults match
        braindecode's MOABB example.
    trial_start_offset_s, trial_stop_offset_s : float
        Offsets (seconds) applied to MOABB's event-defined trial window.
        Default 0.0 / 0.0 keeps MOABB's native trial interval.
    preload : bool
        Preload windows into memory. Default True. Not part of the cache key.
    n_jobs : int
        Parallel preprocess jobs. Default 1 for reproducibility. Not part of
        the cache key.
    cache_dir : str or None
        Root directory for the preprocessed-tensor cache. If None (default),
        no caching: every call runs the full preprocess pipeline. If a path
        is provided, it is created if missing and used for read/write.
        Each cache entry is ~5-200 MB depending on dataset.

    Returns
    -------
    X : (N, C, T) float32
    y : (N,) int64
    metadata : DataFrame with columns ['session', 'run', 'subject']
    sfreq : float
    ch_names : list of str (EEG channel order of X's axis=1)
    """
    # Cache parameters: anything in _CACHE_KEY_PARAMS gets hashed.
    params = {
        "dataset_id": str(dataset_id).lower(),
        "subject": int(subject),
        "l_freq": float(l_freq),
        "h_freq": float(h_freq),
        "ems_factor_new": float(ems_factor_new),
        "ems_init_block_size": int(ems_init_block_size),
        "trial_start_offset_s": float(trial_start_offset_s),
        "trial_stop_offset_s": float(trial_stop_offset_s),
    }

    # Cache lookup. If the file exists, load it; if anything goes wrong,
    # fall through to the full preprocess and overwrite.
    cache_path = _cache_path(cache_dir, params) if cache_dir is not None else None
    if cache_path is not None:
        import os
        if os.path.exists(cache_path):
            try:
                npz = np.load(cache_path, allow_pickle=True)
                metadata = pd.DataFrame({
                    "session": npz["metadata_session"],
                    "run": npz["metadata_run"],
                    "subject": npz["metadata_subject"],
                })
                return (npz["X"], npz["y"], metadata,
                        float(npz["sfreq"].item()), list(npz["ch_names"]))
            except Exception:
                # Corrupt or unreadable cache file (truncated download,
                # partial write, format mismatch). Fall through and overwrite.
                pass

    # Cache miss (or caching disabled): run the canonical preprocess.
    from refshift.compat import make_braindecode_dataset
    dataset = make_braindecode_dataset(dataset_id, int(subject))

    from braindecode.preprocessing import (
        Preprocessor,
        create_windows_from_events,
        exponential_moving_standardize,
        preprocess,
    )

    preprocessors = [
        Preprocessor("pick_types", eeg=True, meg=False, stim=False),
    ]

    # Schirrmeister2017: restrict to motor-cortex channel subset to match
    # the original paper's protocol (44 channels, Section 2.7.1) and keep
    # CSP/DL compute tractable. The full list lives in experiments.py to
    # stay co-located with _resolve_dataset's MOABB-side subsetting.
    if dataset_id == "schirrmeister2017":
        from refshift.experiments import _SCHIRRMEISTER_MOTOR_CHANNELS
        preprocessors.append(
            Preprocessor(
                "pick_channels",
                ch_names=list(_SCHIRRMEISTER_MOTOR_CHANNELS),
                ordered=False,
            )
        )

    preprocessors.extend([
        Preprocessor(_scale_volts_to_microvolts, apply_on_array=True),
        Preprocessor("filter", l_freq=l_freq, h_freq=h_freq),
        Preprocessor(
            exponential_moving_standardize,
            factor_new=ems_factor_new,
            init_block_size=ems_init_block_size,
        ),
    ])
    preprocess(dataset, preprocessors, n_jobs=n_jobs)

    # sfreq + channel names from first raw; enforce consistency across runs.
    sfreq = float(dataset.datasets[0].raw.info["sfreq"])
    for ds in dataset.datasets:
        if ds.raw.info["sfreq"] != sfreq:
            raise RuntimeError(
                f"Inconsistent sfreq across runs for subject {subject}: "
                f"first run {sfreq}, later run {ds.raw.info['sfreq']}."
            )
    ch_names = list(dataset.datasets[0].raw.info["ch_names"])

    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=int(round(sfreq * trial_start_offset_s)),
        trial_stop_offset_samples=int(round(sfreq * trial_stop_offset_s)),
        preload=preload,
    )

    # Concatenate windows + metadata across runs/sessions.
    Xs: List[np.ndarray] = []
    ys: List[int] = []
    rows: List[dict] = []
    for ds_wind in windows_dataset.datasets:
        desc = ds_wind.description
        sess = str(desc["session"]) if "session" in desc else "0"
        run = str(desc["run"]) if "run" in desc else "0"
        subj = int(desc["subject"]) if "subject" in desc else int(subject)
        for i in range(len(ds_wind)):
            x, y, _ind = ds_wind[i]
            Xs.append(np.asarray(x, dtype=np.float32))
            ys.append(int(y))
            rows.append({"session": sess, "run": run, "subject": subj})

    if not Xs:
        raise RuntimeError(
            f"No windows extracted for {dataset_id} subject {subject}. "
            "Check MOABB cache / Kaggle symlinks / trial offset values."
        )

    X = np.stack(Xs).astype(np.float32, copy=False)
    y = np.array(ys, dtype=np.int64)
    metadata = pd.DataFrame(rows)

    # Cache write. Don't let a disk error fail the otherwise-successful run.
    if cache_path is not None:
        try:
            np.savez(
                cache_path[:-len(".npz")],  # np.savez auto-appends .npz
                X=X, y=y, sfreq=np.float64(sfreq),
                metadata_session=metadata["session"].to_numpy(),
                metadata_run=metadata["run"].to_numpy(),
                metadata_subject=metadata["subject"].to_numpy(),
                ch_names=np.asarray(ch_names, dtype=object),
            )
        except OSError:
            pass

    return X, y, metadata, sfreq, ch_names


def make_dl_model(
    model: str,
    *,
    n_channels: int,
    n_classes: int,
    n_times: int,
    sfreq: float,
    seed: int = 0,
    max_epochs: int = 200,
    batch_size: int = 32,
    lr: Optional[float] = None,
    weight_decay: float = 0.0,
    drop_last: bool = False,
    device: Optional[str] = None,
    verbose: int = 0,
    transforms=None,
):
    """Construct a braindecode EEGClassifier for one supervised training run.

    Architecture-specific hyperparameters follow braindecode's MOABB example
    for ShallowFBCSPNet and the Lawhern-2018 defaults for EEGNet_8_2:

      - shallow: lr=6.25e-4 (= 0.0625 * 0.01, braindecode default)
      - eegnet:  lr=1e-3

    Both use AdamW, weight_decay=0, CosineAnnealingLR schedule,
    ``max_epochs`` epochs, ``batch_size`` batch size. No internal train/val
    split (``run_mismatch`` passes the full training set for the given
    train-reference).

    Parameters
    ----------
    model : {'eegnet', 'shallow'}
    n_channels, n_classes, n_times : int
    sfreq : float
        Passed for bookkeeping; not used by the model directly here (braindecode
        models can optionally infer from input_window_seconds + sfreq, but we
        pass n_times explicitly).
    seed : int
        Seed for torch + numpy + random + cudnn.
    max_epochs : int
        Default 200 (balances convergence and 7x7 runtime).
    batch_size : int
        Default 32 (small enough for ~150-250 train trials without losing
        the last partial batch too badly, when drop_last=False).
    lr : float or None
        Learning rate. None -> architecture default.
    weight_decay : float
        AdamW weight decay. Default 0.0.
    drop_last : bool
        If True, drop the incomplete last training batch. Default False
        (small MI datasets can't afford to lose 10-20% of trials).
    device : {'cuda', 'cpu', None}
        None -> auto-detect.
    verbose : int
        skorch verbosity. Default 0 (silent; tqdm in run_mismatch gives progress).
    transforms : list, Transform, or None
        If non-None, the train iterator is swapped to braindecode's
        ``AugmentedDataLoader`` and these transforms are applied per training
        batch. Test/predict is unaffected. Used by ``run_mismatch_jitter``
        for reference-jitter augmentation. None disables augmentation
        (default; matches the canonical Phase 2 mismatch-matrix runs).

    Returns
    -------
    braindecode.EEGClassifier
        A skorch NeuralNetClassifier subclass. Accepts numpy arrays in
        ``.fit(X, y)`` / ``.predict(X)`` / ``.score(X, y)``.
    """
    # Lazy imports so module imports cheaply without [dl] extras.
    import torch
    from braindecode import EEGClassifier
    from braindecode.models import EEGNetv4, ShallowFBCSPNet
    from braindecode.util import set_random_seeds
    from skorch.callbacks import LRScheduler

    model_lc = model.lower()
    if model_lc not in SUPPORTED_DL_MODELS:
        raise ValueError(
            f"Unknown DL model: {model!r}. "
            f"Supported: {SUPPORTED_DL_MODELS}"
        )

    cuda = torch.cuda.is_available()
    if device is None:
        device = "cuda" if cuda else "cpu"
    set_random_seeds(seed=int(seed), cuda=cuda)

    if model_lc == "shallow":
        if lr is None:
            lr = 6.25e-4  # braindecode MOABB example default for ShallowFBCSPNet
        module = ShallowFBCSPNet(
            n_chans=int(n_channels),
            n_outputs=int(n_classes),
            n_times=int(n_times),
            final_conv_length="auto",
        )
    else:  # eegnet
        if lr is None:
            lr = 1e-3  # Lawhern-2018 canonical
        module = EEGNetv4(
            n_chans=int(n_channels),
            n_outputs=int(n_classes),
            n_times=int(n_times),
            F1=8,
            D=2,
            final_conv_length="auto",
        )

    if device == "cuda":
        module = module.cuda()

    # Build EEGClassifier kwargs. When transforms is given, swap the train
    # iterator to AugmentedDataLoader and pass transforms through skorch's
    # ``iterator_train__<param>`` plumbing. Test/predict path is unaffected.
    classifier_kwargs = dict(
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.AdamW,
        optimizer__lr=float(lr),
        optimizer__weight_decay=float(weight_decay),
        batch_size=int(batch_size),
        max_epochs=int(max_epochs),
        train_split=None,
        iterator_train__shuffle=True,
        iterator_train__drop_last=bool(drop_last),
        callbacks=[
            ("lr_scheduler", LRScheduler(
                "CosineAnnealingLR", T_max=max(1, int(max_epochs) - 1),
            )),
        ],
        device=device,
        verbose=int(verbose),
    )
    if transforms is not None:
        from braindecode.augmentation import AugmentedDataLoader
        classifier_kwargs["iterator_train"] = AugmentedDataLoader
        classifier_kwargs["iterator_train__transforms"] = transforms

    clf = EEGClassifier(module, **classifier_kwargs)
    return clf
