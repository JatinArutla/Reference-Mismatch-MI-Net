"""The algebraic operator matrices must match the real operators."""

import numpy as np
import pytest

from refshift.inversion import contrast_recovery_report, operator_matrix
from refshift.references import apply_reference, build_graph

pytest.importorskip("mne")

IV2A = ["Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2",
        "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz"]


def test_matrices_match_the_operators():
    graph = build_graph(IV2A, include_rest=True)
    C = len(IV2A)
    X = np.random.default_rng(0).standard_normal((3, C, 40)).astype(np.float32)
    for mode in ("native", "car", "rest", "cz_ref", "lap_small", "lap_large"):
        M = operator_matrix(mode, graph, C)
        expected = np.einsum("ij,njt->nit", M, X.astype(np.float64))
        actual = apply_reference(X, mode, graph=graph).astype(np.float64)
        assert np.allclose(actual, expected, atol=1e-4), mode


def test_global_refs_preserve_contrasts_and_laplacians_do_not():
    df = contrast_recovery_report(IV2A).set_index("operator")
    for mode in ("native", "car", "rest", "cz_ref"):
        assert bool(df.loc[mode, "contrast_preserving"]), mode
    for mode in ("lap_small", "lap_large"):
        assert not bool(df.loc[mode, "contrast_preserving"]), mode
        # Not canonicalizable, but the contrasts are still linearly recoverable.
        assert bool(df.loc[mode, "contrasts_recoverable"]), mode
