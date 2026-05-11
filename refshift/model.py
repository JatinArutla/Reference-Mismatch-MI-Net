"""Model factories for both pipelines.

CSP+LDA: matches MOABB's canonical CSP.yml (Covariances(oas) -> CSP(6) -> LDA).
DL: braindecode EEGNetv4 / ShallowFBCSPNet, skorch-wrapped so they expose
fit/predict against numpy arrays.

The DL factory keeps both architectures' canonical defaults: ShallowFBCSPNet
uses braindecode's MOABB-example LR (6.25e-4); EEGNet uses Lawhern et al.
2018's small-data MI recommendation (5e-4) uniformly across datasets.
"""

from __future__ import annotations

from typing import Optional

from sklearn.pipeline import Pipeline

from refshift.reference import DatasetGraph, ReferenceTransformer


SUPPORTED_DL_MODELS = ("eegnet", "shallow")


def make_csp_lda_pipeline(
    reference_mode: Optional[str] = None,
    *,
    graph: Optional[DatasetGraph] = None,
    n_filters: int = 6,
) -> Pipeline:
    """CSP+LDA matching MOABB CSP.yml; optional ReferenceTransformer prepended.

    With reference_mode=None this is identical to MOABB's bare pipeline. With
    reference_mode='native' it gains a no-op ReferenceTransformer at the front;
    calibration verifies the two produce identical scores within fp noise.
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
    drop_last: bool = False,
    device: Optional[str] = None,
    verbose: int = 0,
    transforms=None,
):
    """Build a skorch-wrapped braindecode classifier for one training run.

    LR defaults: shallow=6.25e-4 (braindecode MOABB example), eegnet=5e-4
    (Lawhern 2018 small-data MI; 1e-3 overshoots EEGNet's ~3000 params on
    Cho2017's ~80 train trials). transforms=[...] swaps in AugmentedDataLoader
    on the train iterator (used by run_mismatch_jitter); test/predict path is
    unaffected.
    """
    import torch
    from braindecode import EEGClassifier
    from braindecode.models import ShallowFBCSPNet
    from braindecode.util import set_random_seeds
    from skorch.callbacks import LRScheduler

    # EEGNetv4 was renamed to EEGNet in braindecode 1.12; the v4 alias is
    # scheduled to be removed in 1.14.
    try:
        from braindecode.models import EEGNet as _EEGNet
    except ImportError:
        from braindecode.models import EEGNetv4 as _EEGNet

    model_lc = model.lower()
    if model_lc not in SUPPORTED_DL_MODELS:
        raise ValueError(f"Unknown DL model {model!r}; supported: {SUPPORTED_DL_MODELS}")

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
    else:  # eegnet
        if lr is None:
            lr = 5e-4
        module = _EEGNet(
            n_chans=int(n_channels), n_outputs=int(n_classes),
            n_times=int(n_times), F1=8, D=2, final_conv_length="auto",
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

    return EEGClassifier(module, **classifier_kwargs)
