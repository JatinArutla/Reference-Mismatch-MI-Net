
from __future__ import annotations
from typing import Dict, List
import numpy as np


def _ensure(X: np.ndarray) -> np.ndarray:
    if X.ndim not in (2, 3):
        raise ValueError(f'Expected [C,T] or [N,C,T], got {X.shape}')
    return X.astype(np.float32, copy=False)


def native(X: np.ndarray) -> np.ndarray:
    return _ensure(X).copy()


def car(X: np.ndarray) -> np.ndarray:
    X = _ensure(X)
    axis = 1 if X.ndim == 3 else 0
    return (X - X.mean(axis=axis, keepdims=True)).astype(np.float32)


def median_ref(X: np.ndarray) -> np.ndarray:
    X = _ensure(X)
    axis = 1 if X.ndim == 3 else 0
    return (X - np.median(X, axis=axis, keepdims=True)).astype(np.float32)


def gram_schmidt_ref(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    X = _ensure(X)
    if X.ndim == 2:
        X = X[None, ...]
        squeeze = True
    else:
        squeeze = False
    N, C, T = X.shape
    s = X.sum(axis=1, keepdims=True)
    r = (s - X) / max(C - 1, 1)
    num = np.sum(X * r, axis=2, keepdims=True)
    den = np.sum(r * r, axis=2, keepdims=True) + eps
    Y = X - (num / den) * r
    return Y[0] if squeeze else Y


def laplacian_knn(X: np.ndarray, ch_names: List[str], neighbor_map: Dict[str, List[str]]) -> np.ndarray:
    X = _ensure(X)
    squeeze = False
    if X.ndim == 2:
        X = X[None, ...]
        squeeze = True
    idx = {ch: i for i, ch in enumerate(ch_names)}
    Y = np.empty_like(X)
    for ch in ch_names:
        i = idx[ch]
        neigh = [idx[n] for n in neighbor_map[ch]]
        Y[:, i, :] = X[:, i, :] - X[:, neigh, :].mean(axis=1)
    return Y[0] if squeeze else Y


def bipolar_nearest(X: np.ndarray, ch_names: List[str], partner_map: Dict[str, str]) -> np.ndarray:
    X = _ensure(X)
    squeeze = False
    if X.ndim == 2:
        X = X[None, ...]
        squeeze = True
    idx = {ch: i for i, ch in enumerate(ch_names)}
    Y = np.empty_like(X)
    for ch in ch_names:
        i = idx[ch]
        j = idx[partner_map[ch]]
        Y[:, i, :] = X[:, i, :] - X[:, j, :]
    return Y[0] if squeeze else Y


def apply_reference(X: np.ndarray, mode: str, ch_names: List[str] | None = None, neighbor_map: Dict[str, List[str]] | None = None, partner_map: Dict[str, str] | None = None) -> np.ndarray:
    mode = mode.lower()
    if mode == 'native':
        return native(X)
    if mode == 'car':
        return car(X)
    if mode == 'median':
        return median_ref(X)
    if mode == 'gs':
        return gram_schmidt_ref(X)
    if mode == 'laplacian':
        if ch_names is None or neighbor_map is None:
            raise ValueError('laplacian requires ch_names and neighbor_map')
        return laplacian_knn(X, ch_names, neighbor_map)
    if mode == 'bipolar':
        if ch_names is None or partner_map is None:
            raise ValueError('bipolar requires ch_names and partner_map')
        return bipolar_nearest(X, ch_names, partner_map)
    raise ValueError(f'Unknown reference mode: {mode}')
