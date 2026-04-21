"""
Tests for refshift.models and refshift.training.

Covers:
    - ATCNet forward pass: correct output shape
    - ATCNet trainable: one training step reduces loss on overfittable data
    - CSP+LDA pipeline: fit and predict on separable synthetic data
    - End-to-end smoke test: IV-2a subject 1, native reference, ~30 epochs,
      expect test accuracy well above 25% chance
    - Reproducibility: same seed gives same initial weights
    - Jitter batch factory: returns correct shapes, uses all refs over time

Runtime on Kaggle GPU: ~2-4 minutes (dominated by the IV-2a ATCNet smoke test).
"""

from __future__ import annotations

import sys
import traceback
from collections import Counter

import numpy as np

from loader import load_subject
from preprocessing import bandpass_subject_data
from reference_ops import apply_reference, build_graph
from standardization import standardize_mechanistic
from models import build_atcnet, build_csp_lda
from training import (
    evaluate_atcnet,
    evaluate_csp_lda,
    fit_atcnet,
    fit_csp_lda,
    make_jitter_batch_fn,
    set_seed,
)


# ============================================================================
# ATCNet structural
# ============================================================================

def test_atcnet_forward_shape():
    """ATCNet accepts [B, C, T] and outputs [B, n_classes]."""
    import torch
    model = build_atcnet(n_chans=22, n_outputs=4,
                         input_window_seconds=4.0, sfreq=250.0)
    X = torch.randn(8, 22, 1000)
    with torch.no_grad():
        out = model(X)
    assert out.shape == (8, 4), f"output shape {out.shape}, expected (8, 4)"
    assert not torch.isnan(out).any(), "NaN in ATCNet output"
    print(f"  ATCNet forward: input {tuple(X.shape)} -> output {tuple(out.shape)}")


def test_atcnet_seed_reproducibility():
    """Same seed -> same initial weights."""
    import torch
    set_seed(0)
    m1 = build_atcnet(n_chans=22, n_outputs=4, input_window_seconds=4.0, sfreq=250.0)
    p1 = next(m1.parameters()).detach().cpu().numpy().copy()

    set_seed(0)
    m2 = build_atcnet(n_chans=22, n_outputs=4, input_window_seconds=4.0, sfreq=250.0)
    p2 = next(m2.parameters()).detach().cpu().numpy().copy()

    assert np.allclose(p1, p2), "Same seed produced different initial weights"
    print("  ATCNet seed reproducibility: OK")


def test_atcnet_overfits_tiny_data():
    """Training should reduce loss on a tiny, noise-free, learnable dataset."""
    import torch
    import torch.nn as nn

    set_seed(0)
    # 2 classes, clearly separable: class 0 has positive bias, class 1 negative
    N = 20
    X = np.random.randn(N, 22, 500).astype(np.float32) * 0.1
    y = np.array([i % 2 for i in range(N)], dtype=np.int64)
    X[y == 0, 5:10, :] += 2.0  # class 0: positive bump on channels 5-9
    X[y == 1, 5:10, :] -= 2.0  # class 1: negative bump on channels 5-9

    model = fit_atcnet(
        X, y, n_classes=2, sfreq=250.0,
        n_epochs=30, batch_size=8, lr=1e-3, seed=0,
    )

    metrics = evaluate_atcnet(model, X, y)
    # On this easy problem, should reach very high train accuracy (we score on train as a smoke test)
    assert metrics["accuracy"] > 0.70, (
        f"ATCNet did not overfit tiny separable data: train acc = {metrics['accuracy']:.3f}"
    )
    print(f"  ATCNet overfits tiny data: train acc = {metrics['accuracy']:.3f}")


# ============================================================================
# CSP + LDA
# ============================================================================

def test_csp_lda_fits_and_predicts():
    """Fit + predict on random data produces valid predictions."""
    np.random.seed(0)
    X = np.random.randn(40, 22, 500).astype(np.float32)
    y = np.array([i % 2 for i in range(40)], dtype=np.int64)
    pipeline = fit_csp_lda(X, y)
    preds = pipeline.predict(X)
    assert preds.shape == (40,)
    assert set(preds.tolist()) <= {0, 1}
    print("  CSP+LDA fit + predict: OK")


def test_csp_lda_separable_data():
    """On obviously separable synthetic data, accuracy is near 100%."""
    np.random.seed(0)
    N = 80
    X = np.random.randn(N, 22, 500).astype(np.float32) * 0.3
    y = np.array([i % 2 for i in range(N)], dtype=np.int64)
    # Class 0: strong alpha-band sinusoid on left motor channels
    # Class 1: strong alpha-band sinusoid on right motor channels
    t = np.arange(500) / 250.0
    alpha = np.sin(2 * np.pi * 10 * t)
    X[y == 0, 0:5, :] += 2.0 * alpha
    X[y == 1, 5:10, :] += 2.0 * alpha

    pipeline = fit_csp_lda(X, y)
    # Score on a held-out set with the same class structure
    X_te = np.random.randn(40, 22, 500).astype(np.float32) * 0.3
    y_te = np.array([i % 2 for i in range(40)], dtype=np.int64)
    X_te[y_te == 0, 0:5, :] += 2.0 * alpha
    X_te[y_te == 1, 5:10, :] += 2.0 * alpha

    metrics = evaluate_csp_lda(pipeline, X_te, y_te)
    assert metrics["accuracy"] > 0.85, (
        f"CSP+LDA test acc on separable data = {metrics['accuracy']:.3f}"
    )
    print(f"  CSP+LDA separable data: acc = {metrics['accuracy']:.3f}")


# ============================================================================
# Jitter batch factory
# ============================================================================

def test_make_jitter_batch_fn_shapes():
    """Factory returns a callable that yields stacked batches."""
    N, C, T = 20, 22, 500
    X_by_ref = {
        "native": np.random.randn(N, C, T).astype(np.float32),
        "car":    np.random.randn(N, C, T).astype(np.float32),
        "median": np.random.randn(N, C, T).astype(np.float32),
    }
    fn = make_jitter_batch_fn(X_by_ref, training_refs=["native", "car", "median"], seed=0)
    indices = np.array([0, 3, 7, 19])
    batch = fn(indices)
    assert batch.shape == (4, C, T)
    assert batch.dtype == np.float32
    print("  jitter batch factory shapes: OK")


def test_make_jitter_batch_fn_uses_all_refs():
    """Over enough draws, all training refs are sampled at least once."""
    N, C, T = 5, 4, 10
    X_by_ref = {
        ref: np.full((N, C, T), fill_value=k, dtype=np.float32)
        for k, ref in enumerate(["native", "car", "median"])
    }
    fn = make_jitter_batch_fn(X_by_ref, training_refs=["native", "car", "median"], seed=0)
    # Sample one trial many times. Recover the ref from the first element's value.
    seen = set()
    for _ in range(200):
        batch = fn(np.array([0]))
        seen.add(int(batch[0, 0, 0]))
    assert seen == {0, 1, 2}, f"Jitter did not sample all refs; saw {seen}"
    print(f"  jitter uses all refs: saw {seen}")


def test_make_jitter_batch_fn_shape_mismatch_raises():
    """Differently-shaped reference arrays should be rejected."""
    X_by_ref = {
        "native": np.zeros((10, 4, 10), dtype=np.float32),
        "car":    np.zeros((10, 4, 12), dtype=np.float32),  # different T
    }
    try:
        make_jitter_batch_fn(X_by_ref, training_refs=["native", "car"], seed=0)
    except ValueError:
        print("  jitter shape mismatch raises: OK")
        return
    raise AssertionError("Expected ValueError for shape mismatch")


# ============================================================================
# End-to-end smoke test (the expensive one)
# ============================================================================

def test_iv2a_atcnet_end_to_end():
    """
    IV-2a subject 1, native reference, mechanistic standardization, ATCNet
    for 30 epochs. Smoke test: test accuracy should be comfortably above
    chance (0.25 for 4-class). A real run would use 200+ epochs.
    """
    data = load_subject("iv2a", 1)
    data_bp = bandpass_subject_data(data)
    graph = build_graph(data_bp.ch_names, k=4)

    X_tr = standardize_mechanistic(apply_reference(data_bp.X_train, "native", graph=graph))
    X_te = standardize_mechanistic(apply_reference(data_bp.X_test,  "native", graph=graph))

    model = fit_atcnet(
        X_tr, data_bp.y_train, n_classes=4, sfreq=data_bp.sfreq,
        n_epochs=30, batch_size=32, lr=1e-3, seed=0, verbose=True,
    )
    metrics = evaluate_atcnet(model, X_te, data_bp.y_test)
    assert metrics["accuracy"] > 0.35, (
        f"IV-2a ATCNet 30-epoch smoke test: test acc = {metrics['accuracy']:.3f}, "
        f"expected > 0.35 (chance = 0.25)"
    )
    print(f"  IV-2a ATCNet smoke: test acc = {metrics['accuracy']:.3f}, "
          f"kappa = {metrics['kappa']:.3f}, n_test = {metrics['n_test']}")


def test_iv2a_csp_lda_end_to_end():
    """
    IV-2a subject 1, native reference (no standardization), CSP+LDA.
    Expect accuracy > 0.35; published CSP+LDA on IV-2a gets 50-70% per subject.
    """
    data = load_subject("iv2a", 1)
    data_bp = bandpass_subject_data(data)
    graph = build_graph(data_bp.ch_names, k=4)

    X_tr = apply_reference(data_bp.X_train, "native", graph=graph)
    X_te = apply_reference(data_bp.X_test,  "native", graph=graph)

    pipeline = fit_csp_lda(X_tr, data_bp.y_train)
    metrics = evaluate_csp_lda(pipeline, X_te, data_bp.y_test)
    assert metrics["accuracy"] > 0.35, (
        f"IV-2a CSP+LDA smoke: test acc = {metrics['accuracy']:.3f}"
    )
    print(f"  IV-2a CSP+LDA smoke: test acc = {metrics['accuracy']:.3f}, "
          f"kappa = {metrics['kappa']:.3f}")


# ============================================================================
# Main
# ============================================================================

TESTS = [
    ("ATCNet forward shape",          test_atcnet_forward_shape),
    ("ATCNet seed reproducibility",   test_atcnet_seed_reproducibility),
    ("ATCNet overfits tiny data",     test_atcnet_overfits_tiny_data),
    ("CSP+LDA fit/predict",           test_csp_lda_fits_and_predicts),
    ("CSP+LDA separable data",        test_csp_lda_separable_data),
    ("jitter batch fn shapes",        test_make_jitter_batch_fn_shapes),
    ("jitter batch fn uses all refs", test_make_jitter_batch_fn_uses_all_refs),
    ("jitter batch fn rejects mismatch", test_make_jitter_batch_fn_shape_mismatch_raises),
    ("IV-2a ATCNet e2e smoke",        test_iv2a_atcnet_end_to_end),
    ("IV-2a CSP+LDA e2e smoke",       test_iv2a_csp_lda_end_to_end),
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
        print("\nAll chunk-4 tests PASSED.")
    return failed


if __name__ == "__main__":
    failed = run_all()
    sys.exit(1 if failed else 0)
