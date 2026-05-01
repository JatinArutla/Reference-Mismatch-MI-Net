"""Regression tests for v0.11 bug fixes and new entry points.

These tests are pure-synthetic / API-shape only — they don't require MOABB
downloads or the full braindecode preprocessing pipeline. They protect
against the specific issues fixed in v0.11:

  - run_pre_ems_diagonal called make_dl_model with wrong parameter names
    (would TypeError before any work was done).
  - run_mismatch_jitter referenced an undefined `paradigm` variable in
    the graph-construction branch (NameError on every default call).
  - load_dl_data did not include `pre_ems_reference` and `resample` in
    the cache-key tuple, leading to silent cache collisions.
  - operator_distance_correlation now returns bootstrap CI and
    permutation p-values; the result schema changed.
  - run_lofo_matrix and run_bandpass_mismatch are new public entry
    points and need basic API tests.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Bug 1: run_pre_ems_diagonal had a stale make_dl_model call site
# ---------------------------------------------------------------------------

def test_run_pre_ems_diagonal_uses_correct_make_dl_model_kwargs():
    """The keyword names in the make_dl_model call inside
    run_pre_ems_diagonal must match make_dl_model's actual signature.

    Earlier code used name=, n_chans=, input_window_samples= against a
    function whose params were model=, n_channels=, n_times=. That's a
    silent crash bug: the function never produced a result on first call
    and went undetected because there were no end-to-end tests.

    We don't actually run the function (it requires MOABB + GPU); we
    inspect its source to confirm the call site uses the right names.
    """
    from refshift import experiments

    src = inspect.getsource(experiments.run_pre_ems_diagonal)
    assert "model=model_lc" in src, (
        "run_pre_ems_diagonal must pass 'model=', not 'name='"
    )
    assert "n_channels=" in src, (
        "run_pre_ems_diagonal must pass 'n_channels=', not 'n_chans='"
    )
    assert "n_times=" in src, (
        "run_pre_ems_diagonal must pass 'n_times=', not 'input_window_samples='"
    )
    # Negative checks to catch regressions
    assert "name=model_lc" not in src
    assert "n_chans=" not in src
    assert "input_window_samples=" not in src


def test_run_pre_ems_diagonal_default_split_strategy_is_auto():
    """Default split_strategy must be 'auto' (matching run_mismatch);
    None would crash _split_train_test which expects a string strategy."""
    from refshift import experiments

    sig = inspect.signature(experiments.run_pre_ems_diagonal)
    assert sig.parameters["split_strategy"].default == "auto"


# ---------------------------------------------------------------------------
# Bug 2: run_mismatch_jitter referenced an undefined `paradigm` variable
# ---------------------------------------------------------------------------

def test_run_mismatch_jitter_does_not_reference_underscored_paradigm():
    """Earlier code did `dataset, _paradigm = _resolve_dataset(...)` and
    then later referenced bare `paradigm` (NameError). The fix renames
    the unpacked variable to `paradigm`.
    """
    from refshift import experiments

    src = inspect.getsource(experiments.run_mismatch_jitter)
    # Once we unpack as `paradigm`, every reference to a graph-build
    # paradigm should resolve. Verify the bug pattern is gone.
    assert "_paradigm" not in src, (
        "run_mismatch_jitter still uses the underscored binding; the "
        "graph-build code path then references bare `paradigm` -> NameError."
    )


# ---------------------------------------------------------------------------
# Bug 3: load_dl_data cache key
# ---------------------------------------------------------------------------

def test_dl_cache_key_includes_pre_ems_reference():
    """Earlier _CACHE_KEY_PARAMS did not include pre_ems_reference, so
    calls with pre_ems_reference=X silently returned cached data from
    pre_ems_reference=None.
    """
    from refshift.dl import _CACHE_KEY_PARAMS

    assert "pre_ems_reference" in _CACHE_KEY_PARAMS, (
        "pre_ems_reference must be part of the cache key, otherwise the "
        "EMS-control ablation reads stale cached data."
    )


def test_dl_cache_key_includes_resample():
    """v0.11 added a resample step to load_dl_data with default 250 Hz.
    The rate must be part of the cache key so different rates get
    different cache entries.
    """
    from refshift.dl import _CACHE_KEY_PARAMS

    assert "resample" in _CACHE_KEY_PARAMS, (
        "resample must be part of the cache key after v0.11 added the "
        "common-rate resampling step."
    )


def test_dl_cache_key_no_duplicates():
    from refshift.dl import _CACHE_KEY_PARAMS

    assert len(_CACHE_KEY_PARAMS) == len(set(_CACHE_KEY_PARAMS))


# ---------------------------------------------------------------------------
# Bug 4: Schirrmeister DL channel order
# ---------------------------------------------------------------------------

def test_schirrmeister_dl_pick_channels_uses_ordered_true():
    """The DL preprocess for Schirrmeister must use ordered=True so the
    X-axis-1 channel order matches the graph's ch_names (paradigm.channels
    order). Earlier code had ordered=False which would have caused a
    runtime channel-order mismatch.
    """
    from refshift import dl

    src = inspect.getsource(dl.load_dl_data)
    # The Schirrmeister preprocessor block should use ordered=True.
    # Look for the construction "ordered=True" in the schirrmeister branch.
    schirr_idx = src.find('"schirrmeister2017"')
    assert schirr_idx >= 0, "expected schirrmeister branch in load_dl_data"
    # The pick_channels call should follow soon after.
    schirr_block = src[schirr_idx:schirr_idx + 1500]
    assert "ordered=True" in schirr_block, (
        "Schirrmeister DL pick_channels must use ordered=True so the "
        "channel order matches paradigm.channels (and therefore the "
        "neighbour graph used by reference operators)."
    )
    assert "ordered=False" not in schirr_block


# ---------------------------------------------------------------------------
# Bug 5: dl.py module docstring stopped lying about EMS commutativity
# ---------------------------------------------------------------------------

def test_dl_module_docstring_acknowledges_ems_noncommutativity():
    """Earlier the dl.py module docstring claimed reference operators
    applied to EMS-standardized data were "numerically equivalent to
    applying them in raw-space". That's false. v0.11 corrects this.
    """
    from refshift import dl

    doc = (dl.__doc__ or "").lower()
    assert "does **not** commute" in doc or "do not commute" in doc or "not commute" in doc, (
        "dl.py module docstring should acknowledge EMS-reference "
        "non-commutativity. The previous 'numerically equivalent' claim "
        "was empirically false."
    )
    assert "numerically equivalent" not in doc, (
        "dl.py module docstring still contains the false 'numerically "
        "equivalent' claim about EMS-reference commutativity."
    )


# ---------------------------------------------------------------------------
# EEGNet learning rate uniform default
# ---------------------------------------------------------------------------

def test_eegnet_default_lr_is_uniform():
    """v0.11 set EEGNet default LR to 5e-4 (Lawhern 2018 small-data MI)
    instead of 1e-3. This removes the per-dataset Cho2017 override.
    """
    from refshift import dl

    src = inspect.getsource(dl.make_dl_model)
    # Find the eegnet branch and its lr fallback
    eeg_idx = src.find("else:  # eegnet")
    assert eeg_idx >= 0
    eeg_block = src[eeg_idx:eeg_idx + 800]
    assert "5e-4" in eeg_block, "EEGNet default LR should be 5e-4 in v0.11"
    # Make sure 1e-3 isn't still present as a default
    assert "lr = 1e-3" not in eeg_block


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------

def test_public_api_exposes_new_runners():
    """Public surface must expose the v0.11 additions.
    Importing them through the package root is the supported path.
    """
    import refshift

    for name in (
        "run_lofo_matrix",
        "run_bandpass_mismatch",
        "run_pre_ems_diagonal",
    ):
        assert hasattr(refshift, name), f"refshift.{name} missing"
        assert name in refshift.__all__, f"{name} not exported in __all__"


# ---------------------------------------------------------------------------
# run_lofo_matrix: smoke + iteration over holdouts
# ---------------------------------------------------------------------------

def test_run_lofo_matrix_iterates_over_holdouts(monkeypatch):
    """run_lofo_matrix should call run_mismatch_jitter once per holdout,
    each time with condition='lofo' and the appropriate holdout_ref,
    and concatenate the resulting frames.
    """
    import pandas as pd
    from refshift import experiments
    from refshift.reference import REFERENCE_MODES

    calls: list = []

    def fake_jitter(dataset_id, *, model, condition, holdout_ref,
                    seeds=(0,), subjects=None, progress=True, **kwargs):
        calls.append({
            "dataset_id": dataset_id, "model": model,
            "condition": condition, "holdout_ref": holdout_ref,
        })
        return pd.DataFrame([{
            "dataset": dataset_id, "subject": 1, "seed": 0,
            "condition": condition, "holdout_ref": holdout_ref,
            "train_modes": "stub", "test_ref": holdout_ref,
            "accuracy": 0.5, "kappa": 0.0,
            "n_train": 1, "n_test": 1,
        }])

    monkeypatch.setattr(experiments, "run_mismatch_jitter", fake_jitter)

    out = experiments.run_lofo_matrix(
        "iv2a", model="shallow", seeds=[0], progress=False,
    )

    # One call per holdout, all in lofo condition
    assert len(calls) == len(REFERENCE_MODES)
    seen_holdouts = sorted(c["holdout_ref"] for c in calls)
    assert seen_holdouts == sorted(REFERENCE_MODES)
    assert all(c["condition"] == "lofo" for c in calls)
    # Concatenated frame has one row per holdout in this stub
    assert len(out) == len(REFERENCE_MODES)


def test_run_lofo_matrix_rejects_unknown_holdout():
    from refshift import experiments

    with pytest.raises(ValueError, match="not in REFERENCE_MODES"):
        experiments.run_lofo_matrix(
            "iv2a", model="shallow",
            holdout_modes=("not_a_real_mode",),
            progress=False,
        )


# ---------------------------------------------------------------------------
# run_bandpass_mismatch: API shape
# ---------------------------------------------------------------------------

def test_run_bandpass_mismatch_rejects_csp_lda():
    from refshift import experiments

    with pytest.raises(ValueError, match="DL-only"):
        experiments.run_bandpass_mismatch(
            "iv2a", model="csp_lda",
        )


def test_run_bandpass_mismatch_rejects_unknown_reference():
    from refshift import experiments

    with pytest.raises(ValueError, match="REFERENCE_MODES"):
        experiments.run_bandpass_mismatch(
            "iv2a", model="shallow", reference_mode="not_a_mode",
        )


# ---------------------------------------------------------------------------
# operator_distance_correlation: bootstrap + permutation
# ---------------------------------------------------------------------------

def test_operator_distance_correlation_returns_ci_and_perm_p():
    """v0.11 added bootstrap CI and permutation p-values to the result;
    they must be present on the dataclass.
    """
    import pandas as pd
    from refshift.analysis import (
        OperatorDistanceResult,
        operator_distance_correlation,
    )

    # Synthetic 6x6 mean matrix (diagonal much higher than off-diagonal,
    # with mild noise so the gap is not constant — otherwise spearmanr
    # warns harmlessly on every bootstrap and permutation iteration).
    refs = ["native", "car", "median", "laplacian", "rest", "cz_ref"]
    rng = np.random.default_rng(0)
    M = 0.4 + 0.05 * rng.standard_normal((6, 6))
    np.fill_diagonal(M, 0.7)
    df = pd.DataFrame(M, index=refs, columns=refs)

    iv2a_chs = [
        "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz",
        "C2", "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz",
        "P2", "POz",
    ]

    res = operator_distance_correlation(
        df, iv2a_chs,
        n_probe_times=200, n_probes=2,
        n_permutations=100, n_bootstrap=100,
        seed=0,
    )
    assert isinstance(res, OperatorDistanceResult)
    # The new fields must be present
    assert hasattr(res, "ci95_spearman")
    assert hasattr(res, "ci95_pearson")
    assert hasattr(res, "perm_p_spearman")
    assert hasattr(res, "perm_p_pearson")
    # CIs are 2-tuples of floats with low <= high (when not nan)
    lo_s, hi_s = res.ci95_spearman
    if not (np.isnan(lo_s) or np.isnan(hi_s)):
        assert lo_s <= hi_s
    # Permutation p-values are in [0, 1] (Phipson-Smyth small-sample
    # corrected, so strictly > 0)
    assert 0.0 < res.perm_p_spearman <= 1.0
    assert 0.0 < res.perm_p_pearson <= 1.0


# ---------------------------------------------------------------------------
# DL-runner helper scaffolding
# ---------------------------------------------------------------------------

def test_setup_dl_run_skips_graph_when_no_spatial_modes():
    """If the run only uses non-spatial references (native, car, median),
    no neighbour graph should be built — saving the spherical-model
    construction cost when REST is not in the run, and avoiding a MOABB
    raw fetch entirely if no graph is needed at all."""
    from unittest.mock import MagicMock, patch
    from refshift.experiments import _setup_dl_run

    fake_ds = MagicMock()
    fake_ds.code = "FAKE"
    fake_ds.subject_list = [1, 2]
    fake_paradigm = MagicMock()

    with patch("refshift.experiments._resolve_dataset",
               return_value=(fake_ds, fake_paradigm)):
        ctx = _setup_dl_run(
            "iv2a", subjects=None, seeds=[0],
            reference_modes_for_graph=("native", "car", "median"),
            progress=False,
        )
    assert ctx.graph is None
    assert ctx.subjects == [1, 2]
    assert ctx.seeds == [0]
    assert ctx.dataset_code == "FAKE"


def test_setup_dl_run_includes_rest_only_when_requested():
    """include_rest in build_graph should be True iff 'rest' is in the
    declared reference modes; otherwise the spherical model is skipped."""
    from unittest.mock import MagicMock, patch
    from refshift.experiments import _setup_dl_run

    fake_ds = MagicMock()
    fake_ds.code = "FAKE"
    fake_ds.subject_list = [1]
    fake_paradigm = MagicMock()
    fake_paradigm.channels = [
        "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz",
        "C2", "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz",
        "P2", "POz",
    ]

    with patch("refshift.experiments._resolve_dataset",
               return_value=(fake_ds, fake_paradigm)):
        # laplacian + cz_ref but no rest -> graph built without rest matrix
        ctx = _setup_dl_run(
            "iv2a", subjects=None, seeds=[0],
            reference_modes_for_graph=("laplacian", "cz_ref"),
            progress=False,
        )
        assert ctx.graph is not None
        assert ctx.graph.rest_matrix is None

        # rest in the modes -> graph built WITH rest matrix
        ctx2 = _setup_dl_run(
            "iv2a", subjects=None, seeds=[0],
            reference_modes_for_graph=("car", "rest"),
            progress=False,
        )
        assert ctx2.graph is not None
        assert ctx2.graph.rest_matrix is not None


def test_iter_per_subject_dl_jobs_loads_once_per_subject(monkeypatch):
    """The generator must reload underlying data only when the subject
    changes; subsequent seeds reuse the in-memory tensor. This is the
    main reason the helper exists.
    """
    import numpy as np
    import pandas as pd
    from refshift.experiments import _iter_per_subject_dl_jobs, _DLRunContext

    load_calls: list = []

    def fake_load_dl_data(dataset_id, subject, **kwargs):
        load_calls.append(int(subject))
        n = 8
        X = np.random.RandomState(int(subject)).standard_normal((n, 22, 100)).astype(np.float32)
        y = np.array([0, 1] * (n // 2), dtype=np.int64)
        meta = pd.DataFrame({
            "session": ["0"] * n,
            "run": ["0"] * n,
            "subject": [int(subject)] * n,
        })
        ch_names = [
            "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz",
            "C2", "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz",
            "P2", "POz",
        ]
        return X, y, meta, 250.0, ch_names

    # Patch the symbol where _iter_per_subject_dl_jobs imports it
    import refshift.dl as dl_mod
    monkeypatch.setattr(dl_mod, "load_dl_data", fake_load_dl_data)

    ctx = _DLRunContext(
        dataset_id="iv2a",
        dataset_code="FAKE",
        subjects=[1, 2],
        seeds=[0, 1, 2],
        graph=None,
    )

    yielded = list(_iter_per_subject_dl_jobs(
        ctx, split_strategy="stratify", progress=False,
    ))

    # 2 subjects x 3 seeds = 6 jobs, but only 2 underlying loads
    assert len(yielded) == 6
    assert load_calls == [1, 2], f"expected [1, 2], got {load_calls}"

    # Sanity check the yielded shapes
    for subject, seed, X_tr, y_tr, X_te, y_te, sfreq in yielded:
        assert X_tr.ndim == 3 and X_te.ndim == 3
        assert sfreq == 250.0
