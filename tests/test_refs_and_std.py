"""
Tests for refshift.reference_ops and refshift.standardization.

Covers:
    - Mathematical identities of each reference operator
    - Graph construction correctness (shapes, index validity, symmetry)
    - Standardization correctness (zero mean, unit std)
    - Numerical stability (eps handling, zero variance)
    - Integration with real loader output (one subject per dataset)

Runtime on Kaggle: ~60 seconds (most spent loading four subjects).
"""

from __future__ import annotations

import sys
import traceback

import numpy as np

from loader import load_subject
from preprocessing import bandpass_subject_data
from reference_ops import (
    DatasetGraph,
    REFERENCE_MODES,
    apply_reference,
    bipolar_ref,
    build_graph,
    car_ref,
    get_channel_positions,
    gs_ref,
    laplacian_ref,
    median_ref,
    native_ref,
)
from standardization import (
    apply_standardizer,
    fit_standardizer,
    standardize_mechanistic,
)


# ============================================================================
# Reference operator identities
# ============================================================================

def test_native_is_copy_not_view():
    X = np.random.randn(5, 8, 100).astype(np.float32)
    Y = native_ref(X)
    assert Y.shape == X.shape
    assert Y.dtype == np.float32
    assert np.array_equal(Y, X)
    assert Y is not X
    Y[0, 0, 0] = 999.0
    assert X[0, 0, 0] != 999.0, "native_ref returned a view, not a copy"
    print("  native is a fresh copy: OK")


def test_car_zero_channel_mean():
    """After CAR, mean across channels is 0 at every (trial, time)."""
    X = np.random.randn(5, 8, 100).astype(np.float32) * 10 + 3.0  # nonzero mean
    Y = car_ref(X)
    max_abs_mean = np.abs(Y.mean(axis=1)).max()
    assert max_abs_mean < 1e-5, f"CAR residual channel mean = {max_abs_mean:.3g}"
    # 2D case
    X2 = np.random.randn(8, 100).astype(np.float32) + 5.0
    Y2 = car_ref(X2)
    assert np.abs(Y2.mean(axis=0)).max() < 1e-5
    print(f"  CAR residual channel mean: {max_abs_mean:.3g}")


def test_median_zero_channel_median():
    """After median ref, the median across channels is 0 at every (trial, time)."""
    X = np.random.randn(5, 8, 100).astype(np.float32) + 2.0
    Y = median_ref(X)
    max_abs_med = np.abs(np.median(Y, axis=1)).max()
    assert max_abs_med < 1e-5, f"Median-ref residual = {max_abs_med:.3g}"
    print(f"  Median-ref residual: {max_abs_med:.3g}")


def test_gs_orthogonal_to_loo_mean():
    """After GS, Y[n,c,:] has near-zero inner product with its LOO mean r[n,c,:]."""
    np.random.seed(0)
    X = np.random.randn(3, 10, 200).astype(np.float32)
    Y = gs_ref(X)
    N, C, T = X.shape
    s = X.sum(axis=1, keepdims=True)
    r = (s - X) / (C - 1)
    # Compute <Y, r> relative to <r, r> (so we see a ratio, not absolute magnitude)
    inner = np.sum(Y * r, axis=2)                 # [N, C]
    norm_r = np.sqrt(np.sum(r * r, axis=2))       # [N, C]
    norm_Y = np.sqrt(np.sum(Y * Y, axis=2))       # [N, C]
    cos_sim = inner / np.maximum(norm_r * norm_Y, 1e-10)
    max_abs_cos = np.abs(cos_sim).max()
    assert max_abs_cos < 1e-3, (
        f"GS: max |cos(<Y, r>)| = {max_abs_cos:.3g}, expected near 0"
    )
    print(f"  GS orthogonality (max |cos|): {max_abs_cos:.3g}")


def test_laplacian_manual():
    """Hand-computed 3-channel case."""
    X = np.array([[1.0, 2.0, 3.0, 4.0],
                  [5.0, 6.0, 7.0, 8.0],
                  [9.0, 10., 11., 12.]], dtype=np.float32)  # [C=3, T=4]
    # neighbor_idx such that c=0 uses {1,2}, c=1 uses {0,2}, c=2 uses {0,1}
    idx = np.array([[1, 2], [0, 2], [0, 1]], dtype=np.int64)
    Y = laplacian_ref(X, idx)
    # Expected: Y[c,t] = X[c,t] - mean(X[neighbors_of_c, t])
    expected = np.array([
        [1.0 - (5 + 9)/2,  2.0 - (6 + 10)/2, 3.0 - (7 + 11)/2, 4.0 - (8 + 12)/2],
        [5.0 - (1 + 9)/2,  6.0 - (2 + 10)/2, 7.0 - (3 + 11)/2, 8.0 - (4 + 12)/2],
        [9.0 - (1 + 5)/2, 10.0 - (2 + 6)/2, 11.0 - (3 + 7)/2, 12.0 - (4 + 8)/2],
    ], dtype=np.float32)
    assert np.allclose(Y, expected), f"Laplacian mismatch:\n got\n{Y}\n expected\n{expected}"
    print("  Laplacian manual (3-channel): OK")


def test_bipolar_manual():
    """Hand-computed bipolar case."""
    X = np.array([[1.0, 2.0, 3.0],
                  [10., 20., 30.],
                  [100., 200., 300.]], dtype=np.float32)  # [C=3, T=3]
    idx = np.array([1, 0, 1], dtype=np.int64)  # 0→1, 1→0, 2→1
    Y = bipolar_ref(X, idx)
    expected = np.array([
        [1.0 - 10., 2.0 - 20., 3.0 - 30.],
        [10. - 1.,  20. - 2.,  30. - 3.],
        [100. - 10, 200. - 20, 300. - 30],
    ], dtype=np.float32)
    assert np.allclose(Y, expected), f"Bipolar mismatch:\n got\n{Y}\n expected\n{expected}"
    print("  Bipolar manual (3-channel): OK")


def test_reference_ops_preserve_shape_and_dtype():
    X = np.random.randn(5, 8, 100).astype(np.float32)
    # Synthetic graph for 8 channels
    idx_lap = np.stack([np.roll(np.arange(8), -1),
                        np.roll(np.arange(8), 1),
                        np.roll(np.arange(8), 2),
                        np.roll(np.arange(8), -2)], axis=1).astype(np.int64)  # [8, 4]
    idx_bip = np.roll(np.arange(8), -1).astype(np.int64)                        # [8]
    graph = DatasetGraph(ch_names=[str(i) for i in range(8)],
                         laplacian_idx=idx_lap, bipolar_idx=idx_bip,
                         k=4, montage="standard_1005")

    for mode in REFERENCE_MODES:
        Y = apply_reference(X, mode, graph=graph)
        assert Y.shape == X.shape, f"{mode}: shape changed"
        assert Y.dtype == np.float32, f"{mode}: dtype changed"
        assert Y.flags["C_CONTIGUOUS"], f"{mode}: not contiguous"
        assert np.isfinite(Y).all(), f"{mode}: NaN/Inf introduced"
    print(f"  all modes preserve shape/dtype: {REFERENCE_MODES}")


# ============================================================================
# Graph construction
# ============================================================================

def test_positions_standard_1005():
    """standard_1005 contains every channel we care about."""
    ch = ["Cz", "C3", "C4", "Fz", "Pz", "FC1", "FC2", "CP1", "CP2"]
    xyz = get_channel_positions(ch)
    assert xyz.shape == (len(ch), 3)
    assert np.isfinite(xyz).all()
    # Sanity: Cz should have x ~ 0 (midline)
    cz_idx = ch.index("Cz")
    assert abs(xyz[cz_idx, 0]) < 0.005, f"Cz x = {xyz[cz_idx, 0]}, expected ~0"
    print(f"  position lookup (9 channels): Cz x = {xyz[cz_idx, 0]:.4f}")


def test_build_graph_iv2a_shape():
    """Graph for IV-2a's 22 channels has expected shapes and valid indices."""
    # Use the actual IV-2a channel list by loading one subject. (Small overhead
    # but guarantees we test realistic channel sets.)
    data = load_subject("iv2a", 1)
    graph = build_graph(data.ch_names, k=4)

    C = len(data.ch_names)
    assert graph.laplacian_idx.shape == (C, 4)
    assert graph.bipolar_idx.shape == (C,)
    assert graph.k == 4
    assert graph.laplacian_idx.dtype == np.int64
    assert graph.bipolar_idx.dtype == np.int64
    # All indices valid
    assert graph.laplacian_idx.min() >= 0 and graph.laplacian_idx.max() < C
    assert graph.bipolar_idx.min() >= 0 and graph.bipolar_idx.max() < C
    # Self-neighbor excluded
    for c in range(C):
        assert c not in graph.laplacian_idx[c], f"self in laplacian for channel {c}"
        assert graph.bipolar_idx[c] != c, f"self-partner for channel {c}"

    # Sanity: C3's single nearest neighbor should be a nearby central channel
    # (standard layouts have C1 directly adjacent to C3).
    c3 = data.ch_names.index("C3")
    bip_of_c3 = data.ch_names[graph.bipolar_idx[c3]]
    assert bip_of_c3 in ("C1", "C5", "CP3", "FC3"), (
        f"C3's nearest neighbor is {bip_of_c3}; expected C1/C5/CP3/FC3"
    )
    print(f"  IV-2a graph: laplacian[{C},4], bipolar[{C}]; C3→{bip_of_c3}")


# ============================================================================
# Standardization
# ============================================================================

def test_mechanistic_zero_mean_unit_std():
    X = np.random.randn(5, 8, 100).astype(np.float32) * 3.0 + 7.0  # nonzero mean, non-unit std
    Y = standardize_mechanistic(X)
    mu = Y.mean(axis=-1)
    sd = Y.std(axis=-1)
    assert np.abs(mu).max() < 1e-5, f"mechanistic residual mean = {np.abs(mu).max():.3g}"
    assert np.abs(sd - 1.0).max() < 1e-4, f"mechanistic residual std dev from 1 = {np.abs(sd - 1.0).max():.3g}"
    print(f"  mechanistic: max|mean|={np.abs(mu).max():.3g}, max|std-1|={np.abs(sd - 1.0).max():.3g}")


def test_mechanistic_shape_and_2d():
    X = np.random.randn(5, 8, 100).astype(np.float32)
    Y = standardize_mechanistic(X)
    assert Y.shape == X.shape
    assert Y.dtype == np.float32
    # 2D
    X2 = np.random.randn(8, 100).astype(np.float32)
    Y2 = standardize_mechanistic(X2)
    assert Y2.shape == X2.shape
    print("  mechanistic shape/2D: OK")


def test_mechanistic_zero_variance_channel():
    """A constant channel shouldn't produce NaN/Inf."""
    X = np.random.randn(3, 5, 50).astype(np.float32)
    X[:, 2, :] = 5.0  # channel 2 is constant across time in every trial
    Y = standardize_mechanistic(X)
    assert np.isfinite(Y).all()
    # The constant channel gets (x-x)/eps = 0 after standardization
    assert np.abs(Y[:, 2, :]).max() < 1e-3
    print("  mechanistic handles zero-variance channel without NaN")


def test_deployment_fit_apply_roundtrip():
    """After fit on train and apply to train, train should be ~ zero-mean unit-std per channel."""
    X_train = np.random.randn(10, 8, 100).astype(np.float32) * 2.0 + 1.5
    X_test  = np.random.randn(5,  8, 100).astype(np.float32) * 2.0 + 1.5

    mu, sd = fit_standardizer(X_train)
    assert mu.shape == (1, 8, 1), f"mu shape {mu.shape}"
    assert sd.shape == (1, 8, 1), f"sd shape {sd.shape}"

    Xtr_s = apply_standardizer(X_train, mu, sd)
    Xte_s = apply_standardizer(X_test,  mu, sd)

    # Train: per-channel global mean ~ 0, std ~ 1
    ch_mean = Xtr_s.mean(axis=(0, 2))
    ch_std  = Xtr_s.std(axis=(0, 2))
    assert np.abs(ch_mean).max() < 1e-4, f"train channel mean residual {np.abs(ch_mean).max():.3g}"
    assert np.abs(ch_std - 1.0).max() < 1e-3, f"train channel std residual {np.abs(ch_std - 1.0).max():.3g}"

    # Test: shape preserved, finite
    assert Xte_s.shape == X_test.shape
    assert np.isfinite(Xte_s).all()
    print(f"  deployment train post-stats: |mean|max={np.abs(ch_mean).max():.3g}, "
          f"|std-1|max={np.abs(ch_std - 1.0).max():.3g}")


def test_deployment_2d_apply():
    """apply_standardizer should work on 2D input too (single trial)."""
    X_train = np.random.randn(10, 8, 100).astype(np.float32)
    mu, sd = fit_standardizer(X_train)
    x = np.random.randn(8, 100).astype(np.float32)
    y = apply_standardizer(x, mu, sd)
    assert y.shape == x.shape
    assert np.isfinite(y).all()
    print("  deployment 2D apply: OK")


# ============================================================================
# Integration with real data (one subject each, post-bandpass)
# ============================================================================

def test_full_pipeline_iv2a():
    """Full loader -> bandpass -> each reference mode -> standardize, no NaN."""
    data = bandpass_subject_data(load_subject("iv2a", 1))
    graph = build_graph(data.ch_names, k=4)

    for mode in REFERENCE_MODES:
        X_ref = apply_reference(data.X_train, mode, graph=graph)
        assert X_ref.shape == data.X_train.shape
        assert np.isfinite(X_ref).all(), f"NaN after ref={mode}"
        # Mechanistic
        X_m = standardize_mechanistic(X_ref)
        assert np.isfinite(X_m).all(), f"NaN after mechanistic (ref={mode})"
        # Deployment
        mu, sd = fit_standardizer(X_ref)
        X_d = apply_standardizer(X_ref, mu, sd)
        assert np.isfinite(X_d).all(), f"NaN after deployment (ref={mode})"
    print(f"  IV-2a full pipeline: 6 modes × 2 standardizations, all finite")


def test_full_pipeline_openbmi():
    data = bandpass_subject_data(load_subject("openbmi", 1))
    graph = build_graph(data.ch_names, k=4)

    for mode in REFERENCE_MODES:
        X_ref = apply_reference(data.X_train, mode, graph=graph)
        assert X_ref.shape == data.X_train.shape
        assert np.isfinite(X_ref).all(), f"NaN after ref={mode}"
    print(f"  OpenBMI 6 modes: all finite")


def test_full_pipeline_cho2017():
    data = bandpass_subject_data(load_subject("cho2017", 1))
    graph = build_graph(data.ch_names, k=4)

    for mode in REFERENCE_MODES:
        X_ref = apply_reference(data.X_all, mode, graph=graph)
        assert X_ref.shape == data.X_all.shape
        assert np.isfinite(X_ref).all(), f"NaN after ref={mode}"
    print(f"  Cho2017 6 modes: all finite")


def test_full_pipeline_dreyer():
    data = bandpass_subject_data(load_subject("dreyer2023", 1))
    graph = build_graph(data.ch_names, k=4)

    for mode in REFERENCE_MODES:
        X_ref = apply_reference(data.X_all, mode, graph=graph)
        assert X_ref.shape == data.X_all.shape
        assert np.isfinite(X_ref).all(), f"NaN after ref={mode}"
    print(f"  Dreyer 6 modes: all finite")


# ============================================================================
# Main
# ============================================================================

TESTS = [
    ("native is copy",             test_native_is_copy_not_view),
    ("CAR zero channel mean",      test_car_zero_channel_mean),
    ("Median zero channel median", test_median_zero_channel_median),
    ("GS orthogonal to LOO mean",  test_gs_orthogonal_to_loo_mean),
    ("Laplacian manual",           test_laplacian_manual),
    ("Bipolar manual",             test_bipolar_manual),
    ("All modes shape/dtype",      test_reference_ops_preserve_shape_and_dtype),
    ("standard_1005 positions",    test_positions_standard_1005),
    ("build_graph iv2a",           test_build_graph_iv2a_shape),
    ("mechanistic zero/unit",      test_mechanistic_zero_mean_unit_std),
    ("mechanistic shape/2D",       test_mechanistic_shape_and_2d),
    ("mechanistic zero variance",  test_mechanistic_zero_variance_channel),
    ("deployment fit+apply",       test_deployment_fit_apply_roundtrip),
    ("deployment 2D apply",        test_deployment_2d_apply),
    ("full pipeline iv2a",         test_full_pipeline_iv2a),
    ("full pipeline openbmi",      test_full_pipeline_openbmi),
    ("full pipeline cho2017",      test_full_pipeline_cho2017),
    ("full pipeline dreyer",       test_full_pipeline_dreyer),
]


def run_all():
    passed, failed = [], []
    for name, fn in TESTS:
        print(f"\n===== {name} =====")
        try:
            fn()
            passed.append(name)
        except Exception as e:
            traceback.print_exc()
            failed.append((name, repr(e)))
    print("\n" + "=" * 72)
    print(f"PASSED ({len(passed)}):")
    for n in passed:
        print(f"  PASS  {n}")
    if failed:
        print(f"\nFAILED ({len(failed)}):")
        for n, err in failed:
            print(f"  FAIL  {n}: {err}")
    else:
        print("\nAll chunk-3 tests PASSED.")
    return failed


if __name__ == "__main__":
    failed = run_all()
    sys.exit(1 if failed else 0)
