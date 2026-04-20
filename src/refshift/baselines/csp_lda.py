
from __future__ import annotations
import numpy as np
from scipy.linalg import eigh
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from refshift.preprocessing.reference_ops import apply_reference
from refshift.preprocessing.standardization import standardize_trials_instance, fit_train_standardizer, apply_train_standardizer
from refshift.utils.metrics import classification_metrics


def _cov_normalized(X, eps=1e-10):
    C = X @ X.T
    tr = np.trace(C)
    return C / (tr + eps)

def _mean_cov(X):
    return np.mean(np.stack([_cov_normalized(x) for x in X], axis=0), axis=0)

def _csp_ovr_filters(X, y, m=2, eps=1e-8):
    classes = np.unique(y)
    C = X.shape[1]
    filters = []
    for k in classes:
        Xk = X[y == k]
        Xr = X[y != k]
        Rk = _mean_cov(Xk); Rr = _mean_cov(Xr); R = Rk + Rr
        w, v = eigh(Rk + eps*np.eye(C), R + eps*np.eye(C))
        idx = np.argsort(w)
        v = v[:, idx]
        filters.append(np.concatenate([v[:, :m], v[:, -m:]], axis=1))
    return np.concatenate(filters, axis=1).astype(np.float32)

def _features_logvar(X, W, eps=1e-10):
    Z = np.einsum('nct,cf->nft', X, W)
    var = np.var(Z, axis=2) + eps
    return np.log(var).astype(np.float32)

def run_csp_lda(Xtr_native, ytr, Xte_native, yte, train_mode, test_mode, ch_names, neighbor_map, partner_map, standardization='mechanistic', m=2):
    Xtr = apply_reference(Xtr_native, train_mode, ch_names=ch_names, neighbor_map=neighbor_map, partner_map=partner_map)
    Xte = apply_reference(Xte_native, test_mode, ch_names=ch_names, neighbor_map=neighbor_map, partner_map=partner_map)
    if standardization == 'mechanistic':
        Xtr = standardize_trials_instance(Xtr); Xte = standardize_trials_instance(Xte)
    else:
        mu, sd = fit_train_standardizer(Xtr)
        Xtr = apply_train_standardizer(Xtr, mu, sd); Xte = apply_train_standardizer(Xte, mu, sd)
    W = _csp_ovr_filters(Xtr, ytr, m=m)
    Ftr = _features_logvar(Xtr, W); Fte = _features_logvar(Xte, W)
    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    clf.fit(Ftr, ytr)
    pred = clf.predict(Fte)
    return classification_metrics(yte, pred), pred
