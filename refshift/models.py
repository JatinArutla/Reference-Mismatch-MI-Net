"""Model factories: one classical pipeline and three deep nets.

CSP+LDA (classical)
    Covariances(oas) -> CSP(6 filters) -> LDA. This matches MOABB's canonical
    CSP recipe, so the calibration check can confirm we reproduce the standard
    baseline. An optional reference operator can be prepended.

Deep nets (braindecode, skorch-wrapped so they take numpy arrays)
    shallow  ShallowFBCSPNet (Schirrmeister 2017), LR 6.25e-4 (braindecode's
             MOABB example value).
    eegnet   EEGNet (Lawhern 2018), LR 5e-4 (the small-data MI recommendation).
    atcnet   ATCNet (Altaheri 2022): conv + attention + TCN, LR 9e-4.

All three share the same constructor signature (n_chans, n_outputs, n_times,
sfreq) and the same optimiser/schedule (AdamW + cosine annealing), so the only
thing that differs between runs is the architecture.
"""

from __future__ import annotations

from typing import Optional

from sklearn.pipeline import Pipeline

from refshift.references import DatasetGraph, ReferenceTransformer

SUPPORTED_DL_MODELS: tuple = ("eegnet", "shallow", "atcnet")


def make_csp_lda_pipeline(
    reference_mode: Optional[str] = None,
    *,
    graph: Optional[DatasetGraph] = None,
    n_filters: int = 6,
) -> Pipeline:
    """CSP+LDA matching MOABB's CSP recipe; optional reference at the front.

    With reference_mode=None this is MOABB's bare pipeline. With
    reference_mode='native' it gains a no-op ReferenceTransformer; the
    calibration check verifies the two score identically within fp noise.

    The CSP+LDA path applies no per-channel standardisation: it is
    covariance-based and calibrated against MOABB directly. (Per-channel
    z-scoring is a deep-learning-only step, done in preprocess.py.)
    """
    from pyriemann.estimation import Covariances
    from pyriemann.spatialfilters import CSP
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    steps = []
    if reference_mode is not None:
        steps.append(("reference", ReferenceTransformer(reference_mode, graph=graph)))
    steps.extend([
        ("cov", Covariances(estimator="oas")),
        ("csp", CSP(nfilter=n_filters)),
        ("lda", LinearDiscriminantAnalysis(solver="svd")),
    ])
    return Pipeline(steps)


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
    device: Optional[str] = None,
    verbose: int = 0,
    transforms=None,
):
    """Build a skorch-wrapped braindecode classifier for one training run.

    ``transforms`` plugs an AugmentedDataLoader into the *train* iterator only
    (used by the jitter experiment to re-reference each sample on the fly); the
    predict path is unaffected.
    """
    import torch
    from braindecode import EEGClassifier
    from braindecode.models import ShallowFBCSPNet
    from braindecode.util import set_random_seeds
    from skorch.callbacks import LRScheduler

    # EEGNetv4 was renamed EEGNet in braindecode 1.12; accept either.
    try:
        from braindecode.models import EEGNet as _EEGNet
    except ImportError:
        from braindecode.models import EEGNetv4 as _EEGNet

    try:
        from braindecode.models import ATCNet as _ATCNet
    except ImportError:
        _ATCNet = None

    model_lc = model.lower()
    if model_lc not in SUPPORTED_DL_MODELS:
        raise ValueError(f"Unknown DL model {model!r}; supported: {SUPPORTED_DL_MODELS}")
    if model_lc == "atcnet" and _ATCNet is None:
        raise ImportError("ATCNet requires braindecode>=0.8.")

    cuda = torch.cuda.is_available()
    if device is None:
        device = "cuda" if cuda else "cpu"
    set_random_seeds(seed=int(seed), cuda=cuda)

    if model_lc == "shallow":
        if lr is None:
            lr = 6.25e-4
        module = ShallowFBCSPNet(
            n_chans=int(n_channels), n_outputs=int(n_classes),
            n_times=int(n_times), final_conv_length="auto",
        )
    elif model_lc == "eegnet":
        if lr is None:
            lr = 5e-4
        module = _EEGNet(
            n_chans=int(n_channels), n_outputs=int(n_classes),
            n_times=int(n_times), F1=8, D=2, final_conv_length="auto",
        )
    else:  # atcnet
        if lr is None:
            lr = 9e-4
        # ATCNet checks n_times == input_window_seconds * sfreq, so derive the
        # window length from the data rather than hardcoding it.
        module = _ATCNet(
            n_chans=int(n_channels), n_outputs=int(n_classes),
            n_times=int(n_times), sfreq=float(sfreq),
            input_window_seconds=float(n_times) / float(sfreq),
        )

    if device == "cuda":
        module = module.cuda()

    classifier_kwargs = dict(
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.AdamW,
        optimizer__lr=float(lr),
        optimizer__weight_decay=float(weight_decay),
        batch_size=int(batch_size),
        max_epochs=int(max_epochs),
        train_split=None,
        iterator_train__shuffle=True,
        iterator_train__drop_last=False,
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

    return EEGClassifier(module, **classifier_kwargs)
