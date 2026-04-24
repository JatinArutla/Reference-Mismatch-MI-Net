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
}


def _moabb_code(dataset_id: str) -> str:
    """Map refshift dataset_id -> braindecode MOABBDataset class name."""
    key = dataset_id.lower()
    if key not in _DATASET_ID_TO_MOABB:
        raise ValueError(
            f"Unknown dataset_id: {dataset_id!r}. "
            f"Known: {tuple(_DATASET_ID_TO_MOABB)}"
        )
    return _DATASET_ID_TO_MOABB[key]


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
        Preload windows into memory. Default True.
    n_jobs : int
        Parallel preprocess jobs. Default 1 for reproducibility.

    Returns
    -------
    X : (N, C, T) float32
    y : (N,) int64
    metadata : DataFrame with columns ['session', 'run', 'subject']
    sfreq : float
    ch_names : list of str (EEG channel order of X's axis=1)
    """
    # Lazy imports so that `import refshift` doesn't require `[dl]` extras.
    from braindecode.datasets import MOABBDataset
    from braindecode.preprocessing import (
        Preprocessor,
        create_windows_from_events,
        exponential_moving_standardize,
        preprocess,
    )

    moabb_code = _moabb_code(dataset_id)
    dataset = MOABBDataset(dataset_name=moabb_code, subject_ids=[int(subject)])

    preprocessors = [
        Preprocessor("pick_types", eeg=True, meg=False, stim=False),
        Preprocessor(lambda data: data * 1e6, apply_on_array=True),
        Preprocessor("filter", l_freq=l_freq, h_freq=h_freq),
        Preprocessor(
            exponential_moving_standardize,
            factor_new=ems_factor_new,
            init_block_size=ems_init_block_size,
        ),
    ]
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

    clf = EEGClassifier(
        module,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.AdamW,
        optimizer__lr=float(lr),
        optimizer__weight_decay=float(weight_decay),
        batch_size=int(batch_size),
        max_epochs=int(max_epochs),
        train_split=None,  # full training on given X, y
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
    return clf
