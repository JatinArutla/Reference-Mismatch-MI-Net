
from __future__ import annotations
import numpy as np


def standardize_trial_instance(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mu = x.mean(axis=-1, keepdims=True)
    sd = x.std(axis=-1, keepdims=True)
    return ((x - mu) / np.maximum(sd, eps)).astype(np.float32)


def standardize_trials_instance(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mu = X.mean(axis=-1, keepdims=True)
    sd = X.std(axis=-1, keepdims=True)
    return ((X - mu) / np.maximum(sd, eps)).astype(np.float32)


def fit_train_standardizer(X: np.ndarray, eps: float = 1e-6):
    mu = X.mean(axis=(0,2), keepdims=True)
    sd = X.std(axis=(0,2), keepdims=True)
    sd = np.maximum(sd, eps)
    return mu.astype(np.float32), sd.astype(np.float32)


def apply_train_standardizer(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return ((X - mu) / sd).astype(np.float32)
