"""Tests for refshift.env module integrity.

These tests don't actually exercise Kaggle-specific filesystem setup
(the test environment doesn't have /kaggle/input mounted); they verify
that the module's helper functions are importable and have the right
shape. The goal is to catch dropped function headers, broken imports,
and similar silent breakage that would surface only when a user tries
to call setup_kaggle_env() on Kaggle.

Background: in v0.13.x an earlier edit accidentally deleted the
``def _patch_moabb_dreyer_no_unzip`` line from refshift.env, leaving
the function body as unreachable code. The function was still
*referenced* from ``_setup_dreyer_symlinks`` but no longer defined,
producing a NameError on Kaggle every time someone called
setup_kaggle_env() with the default datasets list (which includes
dreyer2023). Unit tests didn't catch it because nothing imported
the function by name.
"""
from __future__ import annotations

import inspect

import pytest


def test_patch_moabb_dreyer_no_unzip_is_defined():
    """Regression guard for v0.13.x: this function went missing in an
    edit and broke setup_kaggle_env() on Kaggle. It must remain defined
    and callable as long as _setup_dreyer_symlinks references it."""
    from refshift.env import _patch_moabb_dreyer_no_unzip
    assert callable(_patch_moabb_dreyer_no_unzip)
    sig = inspect.signature(_patch_moabb_dreyer_no_unzip)
    assert "verbose" in sig.parameters, (
        f"Expected 'verbose' parameter, got {list(sig.parameters)}"
    )


def test_setup_kaggle_env_importable():
    """The public entry point must import without error."""
    from refshift.env import setup_kaggle_env
    assert callable(setup_kaggle_env)


def test_setup_dreyer_symlinks_calls_patch_helper():
    """The Dreyer setup function references _patch_moabb_dreyer_no_unzip
    by name. If that reference is broken (helper missing or renamed),
    Dreyer setup will fail at runtime on Kaggle. We verify the reference
    resolves at import time by checking the source.
    """
    from refshift import env
    src = inspect.getsource(env._setup_dreyer_symlinks)
    assert "_patch_moabb_dreyer_no_unzip" in src
    # And that the named function exists in the same module:
    assert hasattr(env, "_patch_moabb_dreyer_no_unzip")


def test_all_dataset_setup_helpers_defined():
    """Each per-dataset setup helper referenced by setup_moabb_symlinks
    must exist as a callable in the env module. Catches any future
    function that goes missing the same way _patch_moabb_dreyer_no_unzip
    did."""
    from refshift import env
    expected_helpers = [
        "_setup_openbmi_symlinks",
        "_setup_dreyer_symlinks",
        "_setup_schirrmeister_symlinks",
        "_patch_moabb_dreyer_no_unzip",
    ]
    for name in expected_helpers:
        assert hasattr(env, name), f"Missing function: refshift.env.{name}"
        assert callable(getattr(env, name)), f"Not callable: refshift.env.{name}"
