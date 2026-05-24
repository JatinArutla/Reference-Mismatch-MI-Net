"""Model factories for both pipelines.

CSP+LDA: matches MOABB's canonical CSP.yml (Covariances(oas) -> CSP(6) -> LDA).
DL: braindecode EEGNetv4 / ShallowFBCSPNet, skorch-wrapped so they expose
fit/predict against numpy arrays.

The DL factory keeps both architectures' canonical defaults: ShallowFBCSPNet
uses braindecode's MOABB-example LR (6.25e-4); EEGNet uses Lawhern et al.
2018's small-data MI recommendation (5e-4) uniformly across datasets.

v0.16: optional trace-normalisation step (per-trial cov / tr(cov)) inserted
between Covariances and CSP. Set trace_normalize=True to enable. This is the
methodological ablation for the CSD-scale confound: if CSD remains
operator-isolated in the mismatch matrix under trace-normalised covariance,
the cross-reference failure is genuinely about operator topology, not
about amplitude scale.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from refshift.reference import DatasetGraph, ReferenceTransformer


SUPPORTED_DL_MODELS = ("eegnet", "shallow", "atcnet")


class TraceNormalizer(BaseEstimator, TransformerMixin):
    """Per-trial trace normalisation of covariance matrices.

    For each trial i, replaces Sigma_i with Sigma_i / trace(Sigma_i). Standard
    technique for removing per-trial scaling effects before CSP / LDA; sits
    between Covariances(oas) and CSP in the pipeline. Stateless (no fit
    parameters), so safe under train/test split.

    Trace(Sigma_i) > 0 by construction since Sigma_i is SPD; an epsilon
    floor guards against degenerate near-zero traces (e.g. all-zero trial).
    """

    def __init__(self, eps: float = 1e-12):
        self.eps = float(eps)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # X is (N, C, C) from Covariances(); trace is along the last two axes.
        X = np.asarray(X)
        if X.ndim != 3:
            raise ValueError(
                f"TraceNormalizer expects (N, C, C); got shape {X.shape}"
            )
        traces = np.trace(X, axis1=-2, axis2=-1)  # (N,)
        # Broadcast (N,) -> (N, 1, 1)
        norm = np.maximum(traces, self.eps)[:, None, None]
        return X / norm


def make_csp_lda_pipeline(
    reference_mode: Optional[str] = None,
    *,
    graph: Optional[DatasetGraph] = None,
    n_filters: int = 6,
    trace_normalize: bool = False,
) -> Pipeline:
    """CSP+LDA matching MOABB CSP.yml; optional ReferenceTransformer prepended.

    With reference_mode=None this is identical to MOABB's bare pipeline. With
    reference_mode='native' it gains a no-op ReferenceTransformer at the front;
    calibration verifies the two produce identical scores within fp noise.

    trace_normalize=True inserts a TraceNormalizer between Covariances(oas)
    and CSP. This is the methodological ablation for the CSD-scale confound
    (see module docstring). All other behaviour is unchanged.
    """
    from pyriemann.estimation import Covariances
    from pyriemann.spatialfilters import CSP
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    steps = []
    if reference_mode is not None:
        steps.append(("reference", ReferenceTransformer(reference_mode, graph=graph)))
    steps.append(("cov", Covariances(estimator="oas")))
    if trace_normalize:
        steps.append(("trace_norm", TraceNormalizer()))
    steps.extend([
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

    LR defaults:
      shallow=6.25e-4 (braindecode MOABB example);
      eegnet=5e-4 (Lawhern 2018 small-data MI; 1e-3 overshoots EEGNet's
                   ~3000 params on Cho2017's ~80 train trials);
      atcnet=9e-4  (Altaheri 2022 used 1e-3 with Adam; 9e-4 mild downscale for AdamW).

    transforms=[...] swaps in AugmentedDataLoader on the train iterator
    (used by run_mismatch_jitter); test/predict path is unaffected.
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

    # ATCNet (Altaheri et al. 2022): convolution + self-attention + TCN.
    # Available in braindecode 0.8+; uses same n_chans / n_outputs / n_times
    # / sfreq constructor signature as Shallow and EEGNet.
    _ATCNet = None
    try:
        from braindecode.models import ATCNet as _ATCNet
    except ImportError:
        pass

    model_lc = model.lower()
    if model_lc not in SUPPORTED_DL_MODELS:
        raise ValueError(f"Unknown DL model {model!r}; supported: {SUPPORTED_DL_MODELS}")
    if model_lc == "atcnet" and _ATCNet is None:
        raise ImportError(
            "ATCNet requires braindecode>=0.8. Upgrade braindecode or pick a different model."
        )

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
            # Altaheri et al. 2022 used 1e-3 with Adam; 9e-4 with AdamW is a
            # mild downscale for AdamW's stronger decay. Matches the ranges
            # used in the Altaheri repo example for IV-2a.
            lr = 9e-4
        # ATCNet's __init__ validates n_times == input_window_seconds * sfreq.
        # Default input_window_seconds is 4.5 (Altaheri's IV-2a window after
        # cue trimming). Derive the right value from n_times and sfreq to
        # match whatever trial window the user actually configured.
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
