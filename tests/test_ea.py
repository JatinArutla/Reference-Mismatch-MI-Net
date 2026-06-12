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


# --- v0.19: fit/apply split + calibration EA ---

from refshift.reference import (
    _ea_fit, _ea_apply, stratified_calibration_index,
)


def test_fit_apply_equals_euclidean_alignment():
    """euclidean_alignment(X) must equal _ea_apply(X, _ea_fit(X)) exactly.

    This pins the refactor: the convenience function is just fit-then-apply
    on the same block. If this drifts, the k=full sweep endpoint stops
    matching the v0.18 EA numbers.
    """
    X = _block(seed=3)
    a = euclidean_alignment(X)
    b = _ea_apply(X, _ea_fit(X))
    assert np.allclose(a, b, atol=1e-6)


def test_ea_whitener_fit_on_subset_applies_to_others():
    X = _block(n=60, seed=4)
    whit = _ea_fit(X[:20])
    out = _ea_apply(X[20:], whit)
    assert out.shape == (40, X.shape[1], X.shape[2])
    assert np.isfinite(out).all()


def test_stratified_calibration_index_balanced_and_reproducible():
    y = np.array([0, 1, 2, 3] * 25)  # 100 trials, 25/class
    idx1 = stratified_calibration_index(y, 5, seed=0)
    idx2 = stratified_calibration_index(y, 5, seed=0)
    assert np.array_equal(idx1, idx2)            # reproducible
    assert len(idx1) == 20                        # 5 per class * 4 classes
    counts = np.bincount(y[idx1], minlength=4)
    assert (counts == 5).all()                    # balanced per class


def test_stratified_calibration_handles_class_shortfall():
    y = np.array([0, 0, 1, 1, 1, 1])  # class 0 has only 2 trials
    idx = stratified_calibration_index(y, 5, seed=0)
    # takes all of class 0 (2) and capped class 1 (min(5,4)=4) -> 6 total here
    assert set(y[idx].tolist()) == {0, 1}
    assert (y[idx] == 0).sum() == 2


def test_calib_ea_smoke_via_build_test_variants():
    """Calibration path runs, excludes calib trials, scores the rest."""
    from refshift.experiments.mismatch import _build_test_variants
    ch = ["Fz","FC3","FC1","FCz","FC2","FC4","C5","C3","C1","Cz","C2","C4",
          "C6","CP3","CP1","CPz","CP2","CP4","P1","Pz","P2","POz"]
    g = build_graph(ch)
    X = _block(n=80, c=len(ch), seed=5)
    y = np.array([0, 1] * 40)
    modes = ("native", "car", "cz_ref")
    Xby, y_sc, n_cal = _build_test_variants(
        X_te=X, y_te=y, modes=modes, graph=g,
        apply_ea=True, ea_calib_trials=10, ea_eps=1e-12, seed=0,
    )
    assert n_cal == 20                       # 10/class * 2 classes
    assert len(y_sc) == 60                    # 80 - 20 calibration
    for m in modes:
        assert Xby[m].shape[0] == 60          # only scored trials aligned
        assert np.isfinite(Xby[m]).all()


def test_calib_none_scores_all_trials():
    from refshift.experiments.mismatch import _build_test_variants
    X = _block(n=40, seed=6)
    y = np.array([0, 1] * 20)
    Xby, y_sc, n_cal = _build_test_variants(
        X_te=X, y_te=y, modes=("native", "car"), graph=None,
        apply_ea=True, ea_calib_trials=None, ea_eps=1e-12, seed=0,
    )
    assert len(y_sc) == 40
    assert n_cal == 40
