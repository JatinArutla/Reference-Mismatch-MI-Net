"""Tests for Euclidean Alignment (v0.18). Synthetic only; no MOABB."""

import numpy as np
import pytest

from refshift.reference import (
    REFERENCE_MODES,
    apply_reference,
    apply_reference_then_ea,
    build_graph,
    euclidean_alignment,
)


def _block(n=30, c=22, t=400, seed=0):
    rng = np.random.default_rng(seed)
    # give channels correlated structure so EA has something to whiten
    mix = rng.standard_normal((c, c))
    z = rng.standard_normal((n, c, t))
    return np.einsum("ij,njt->nit", mix, z).astype(np.float32)


def test_ea_yields_identity_mean_covariance():
    X = _block()
    Y = euclidean_alignment(X)
    cov = np.mean([np.cov(Y[i].astype(np.float64)) for i in range(len(Y))], axis=0)
    C = X.shape[1]
    # mean per-trial covariance should be ~ identity after EA
    assert np.allclose(np.diag(cov), 1.0, atol=1e-3)
    off = cov - np.diag(np.diag(cov))
    assert np.abs(off).max() < 1e-3


def test_ea_preserves_shape_and_dtype():
    X = _block()
    Y = euclidean_alignment(X)
    assert Y.shape == X.shape
    assert Y.dtype == np.float32


def test_ea_empty_block_is_safe():
    X = np.empty((0, 22, 400), dtype=np.float32)
    Y = euclidean_alignment(X)
    assert Y.shape == X.shape


def test_apply_reference_then_ea_no_ea_matches_apply_reference():
    X = _block()
    a = apply_reference_then_ea(X, "car", apply_ea=False)
    b = apply_reference(X, "car")
    assert np.allclose(a, b, atol=1e-5)


def test_apply_reference_then_ea_changes_output_when_ea_on():
    X = _block()
    no_ea = apply_reference_then_ea(X, "car", apply_ea=False)
    with_ea = apply_reference_then_ea(X, "car", apply_ea=True)
    # EA should materially change the CAR output
    assert not np.allclose(no_ea, with_ea, atol=1e-3)


def test_ea_survives_rank_deficient_cz_ref():
    # cz_ref zeroes the Cz channel -> rank-deficient covariance. EA's ridge
    # must keep fractional_matrix_power finite and real.
    ch = ["Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz",
          "C2", "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz",
          "P2", "POz"]
    g = build_graph(ch)
    X = _block(c=len(ch))
    Y = apply_reference_then_ea(X, "cz_ref", graph=g, apply_ea=True)
    assert np.isfinite(Y).all()
    assert Y.dtype == np.float32
