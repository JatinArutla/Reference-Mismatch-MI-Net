"""
refshift.training — fit and evaluate ATCNet and CSP+LDA.

ATCNet:
    fit_atcnet(X_tr, y_tr, ...) -> trained model
    evaluate_atcnet(model, X_te, y_te) -> dict of metrics

CSP+LDA:
    fit_csp_lda(X_tr, y_tr, ...) -> fitted pipeline
    evaluate_csp_lda(pipeline, X_te, y_te) -> dict of metrics

For the 6x6 benchmark, train once per train_ref and evaluate against all
six test_refs. The jitter variant is handled by passing a custom
`get_train_batch` callable to `fit_atcnet` (see make_jitter_batch_fn).

All inputs are float32 [N, C, T] arrays with int64 y. Device auto-detects
CUDA when available.
"""

from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional

import numpy as np


# ============================================================================
# Reproducibility
# ============================================================================

def set_seed(seed: int):
    """Seed random, numpy, and torch (CPU + CUDA) for reproducibility.

    Does not enable cudnn deterministic mode (that's a separate, performance-
    costly toggle the caller can set if needed).
    """
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device: str):
    """Resolve 'auto' / 'cuda' / 'cpu' to a torch.device.

    In 'auto' mode, probes CUDA to ensure kernels are available (guards
    against older GPUs like P100/sm_60 where the installed PyTorch may
    lack compatible kernel binaries).
    """
    import torch
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        return torch.device("cuda")
    # auto: check availability AND kernel compatibility
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        _probe = torch.tensor([1.0], device="cuda") * 2.0
        _probe = _probe.cpu()  # force execution
        return torch.device("cuda")
    except RuntimeError as e:
        import warnings
        warnings.warn(
            f"CUDA reported available but kernel launch failed: {e}. "
            f"Falling back to CPU. (If on Kaggle, switch accelerator to "
            f"'GPU T4 x2' for PyTorch-compatible CUDA.)"
        )
        return torch.device("cpu")


# ============================================================================
# ATCNet fit + evaluate
# ============================================================================

def fit_atcnet(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_classes: int,
    sfreq: float,
    *,
    n_epochs: int = 200,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    seed: int = 0,
    device: str = "auto",
    get_train_batch: Optional[Callable] = None,
    verbose: bool = False,
):
    """Train an ATCNet from scratch on preprocessed training data.

    Args:
        X_train:         [N, C, T] float32, already bandpassed + referenced +
                         standardized (whatever protocol the caller chose)
        y_train:         [N] int64 class labels
        n_classes:       number of output classes (4 for iv2a, 2 else)
        sfreq:           sampling rate in Hz
        n_epochs:        number of training epochs
        batch_size:      batch size
        lr:              Adam learning rate
        weight_decay:    AdamW weight decay
        seed:            global RNG seed (random, numpy, torch)
        device:          'auto', 'cuda', or 'cpu'
        get_train_batch: optional callable(np.ndarray batch_indices) -> np.ndarray
                         If given, produces the X batch from the chosen indices
                         (used for reference jitter). If None, defaults to
                         X_train[indices].
        verbose:         print per-epoch loss when True

    Returns:
        Trained torch.nn.Module on the resolved device (caller handles
        eval / state_dict saving).
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim

    from models import build_atcnet

    set_seed(seed)
    device = _resolve_device(device)

    n_chans = X_train.shape[1]
    n_samples = X_train.shape[2]
    input_window_seconds = n_samples / sfreq

    model = build_atcnet(
        n_chans=n_chans,
        n_outputs=n_classes,
        input_window_seconds=input_window_seconds,
        sfreq=sfreq,
    ).to(device)

    y_tensor = torch.from_numpy(y_train.astype(np.int64)).to(device)

    if get_train_batch is None:
        def get_train_batch(indices):
            return X_train[indices]

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    N = len(X_train)
    indices_all = np.arange(N)

    rng = np.random.RandomState(seed)
    model.train()
    for epoch in range(n_epochs):
        perm = rng.permutation(indices_all)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, N, batch_size):
            batch_idx = perm[start:start + batch_size]
            X_batch_np = get_train_batch(batch_idx)
            X_batch = torch.from_numpy(np.ascontiguousarray(X_batch_np, dtype=np.float32)).to(device)
            y_batch = y_tensor[batch_idx]

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            n_batches += 1
        if verbose and (epoch % max(1, n_epochs // 10) == 0 or epoch == n_epochs - 1):
            print(f"  epoch {epoch+1:3d}/{n_epochs}  loss={epoch_loss/n_batches:.4f}")

    return model


def evaluate_atcnet(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    batch_size: int = 256,
    device: str = "auto",
) -> Dict:
    """Run the model on X_test and return metrics."""
    import torch
    from sklearn.metrics import accuracy_score, cohen_kappa_score

    device = _resolve_device(device)
    model = model.to(device).eval()

    preds_all = []
    with torch.no_grad():
        N = len(X_test)
        for start in range(0, N, batch_size):
            X_batch = torch.from_numpy(
                np.ascontiguousarray(X_test[start:start + batch_size], dtype=np.float32)
            ).to(device)
            logits = model(X_batch)
            preds = logits.argmax(dim=-1).cpu().numpy()
            preds_all.append(preds)
    predictions = np.concatenate(preds_all, axis=0)

    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "kappa":    float(cohen_kappa_score(y_test, predictions)),
        "predictions": predictions,
        "n_test": int(len(y_test)),
    }


# ============================================================================
# CSP + LDA fit + evaluate
# ============================================================================

def fit_csp_lda(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    n_filters: int = 6,
):
    """Fit a CSP+LDA pipeline on [N, C, T] training data.

    No standardization is applied inside (CSP handles scaling via covariance
    trace normalization internally). Inputs should be bandpassed + referenced.
    """
    from models import build_csp_lda
    pipeline = build_csp_lda(n_filters=n_filters)
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_csp_lda(pipeline, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
    """Score a fitted CSP+LDA pipeline on test data."""
    from sklearn.metrics import accuracy_score, cohen_kappa_score
    predictions = pipeline.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "kappa":    float(cohen_kappa_score(y_test, predictions)),
        "predictions": predictions,
        "n_test": int(len(y_test)),
    }


# ============================================================================
# Jitter batch factory
# ============================================================================

def make_jitter_batch_fn(
    X_by_ref: Dict[str, np.ndarray],
    training_refs: List[str],
    seed: int,
) -> Callable:
    """Build a get_train_batch function that samples a reference per trial
    from `training_refs` on each call.

    All entries in X_by_ref must have the same [N, C, T] shape. On each
    batch, we pick a reference per trial uniformly at random and pull the
    corresponding trial from the pre-computed X_by_ref[ref] array.

    Args:
        X_by_ref:      dict mapping reference name -> [N, C, T] array that
                       has already been referenced + standardized
        training_refs: references to sample from (keys into X_by_ref)
        seed:          RNG seed for reproducible jitter sampling

    Returns:
        Callable(indices: np.ndarray) -> np.ndarray of shape [len(indices), C, T]
    """
    for ref in training_refs:
        if ref not in X_by_ref:
            raise ValueError(f"training_refs contains {ref!r} but X_by_ref has {list(X_by_ref)}")
    # Sanity: all reference arrays must be aligned
    shapes = {ref: X_by_ref[ref].shape for ref in training_refs}
    first_shape = next(iter(shapes.values()))
    for ref, shape in shapes.items():
        if shape != first_shape:
            raise ValueError(f"Shape mismatch: {ref}={shape}, first={first_shape}")

    rng = np.random.RandomState(seed)
    training_refs = list(training_refs)

    def get_batch(indices: np.ndarray) -> np.ndarray:
        refs = rng.choice(training_refs, size=len(indices))
        out = np.empty(
            (len(indices), first_shape[1], first_shape[2]),
            dtype=np.float32,
        )
        for k, (i, ref) in enumerate(zip(indices, refs)):
            out[k] = X_by_ref[ref][i]
        return out

    return get_batch
