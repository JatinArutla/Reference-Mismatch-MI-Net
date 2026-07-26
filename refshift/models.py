"""Model factories: one classical pipeline and three deep nets.

CSP+LDA     Covariances(oas) -> CSP(6 filters) -> LDA. MOABB's canonical CSP
            recipe, so calibrate_csp_lda can check we reproduce the standard
            baseline. Covariance-based, so no per-channel standardisation.

shallow     ShallowFBCSPNet (Schirrmeister 2017), lr 6.25e-4
eegnet      EEGNet (Lawhern 2018), lr 5e-4
atcnet      ATCNet (Altaheri 2022), lr 9e-4

All three deep nets share one optimiser and schedule (AdamW + cosine
annealing), so the architecture is the only thing that differs between runs.
"""

from __future__ import annotations

from typing import Optional

from sklearn.pipeline import Pipeline

SUPPORTED_DL_MODELS = ("shallow", "eegnet", "atcnet")
DEFAULT_LR = {"shallow": 6.25e-4, "eegnet": 5e-4, "atcnet": 9e-4}


def make_csp_lda_pipeline(n_filters: int = 6) -> Pipeline:
    from pyriemann.estimation import Covariances
    from pyriemann.spatialfilters import CSP
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    return Pipeline([
        ("cov", Covariances(estimator="oas")),
        ("csp", CSP(nfilter=n_filters)),
        ("lda", LinearDiscriminantAnalysis(solver="svd")),
    ])


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
    transforms=None,
):
    """Build a skorch-wrapped braindecode classifier for one training run.

    ``transforms`` plugs an AugmentedDataLoader into the *train* iterator only
    (the jitter experiment re-references each sample on the fly); predict is
    unaffected.
    """
    import torch
    from braindecode import EEGClassifier
    from braindecode.models import ATCNet, ShallowFBCSPNet
    from braindecode.util import set_random_seeds
    from skorch.callbacks import LRScheduler

    try:                                    # renamed in braindecode 1.12
        from braindecode.models import EEGNet
    except ImportError:
        from braindecode.models import EEGNetv4 as EEGNet

    model = model.lower()
    if model not in SUPPORTED_DL_MODELS:
        raise ValueError(f"Unknown model {model!r}; use one of {SUPPORTED_DL_MODELS}")

    cuda = torch.cuda.is_available()
    device = "cuda" if cuda else "cpu"
    set_random_seeds(seed=int(seed), cuda=cuda)   # before init, so weights are seeded

    if model == "shallow":
        module = ShallowFBCSPNet(n_chans=n_channels, n_outputs=n_classes,
                                 n_times=n_times, final_conv_length="auto")
    elif model == "eegnet":
        module = EEGNet(n_chans=n_channels, n_outputs=n_classes,
                        n_times=n_times, F1=8, D=2, final_conv_length="auto")
    else:
        # ATCNet checks n_times == input_window_seconds * sfreq, so derive the
        # window length from the data rather than hardcoding it.
        module = ATCNet(n_chans=n_channels, n_outputs=n_classes, n_times=n_times,
                        sfreq=sfreq, input_window_seconds=n_times / sfreq)

    kwargs = dict(
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.AdamW,
        optimizer__lr=DEFAULT_LR[model],
        batch_size=int(batch_size),
        max_epochs=int(max_epochs),
        train_split=None,
        iterator_train__shuffle=True,
        callbacks=[("lr_scheduler", LRScheduler(
            "CosineAnnealingLR", T_max=max(1, int(max_epochs) - 1)))],
        device=device,
        verbose=0,
    )
    if transforms is not None:
        from braindecode.augmentation import AugmentedDataLoader
        kwargs["iterator_train"] = AugmentedDataLoader
        kwargs["iterator_train__transforms"] = transforms

    return EEGClassifier(module.to(device), **kwargs)
