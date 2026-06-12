"""Experiment runners for refshift.

Public entry points (re-exported at refshift package root):

    calibrate_csp_lda      MOABB calibration on CSP+LDA
    run_mismatch           NxN train-test reference mismatch matrix
    mismatch_matrix        long-form -> NxN pivot
    run_mismatch_jitter    DL with per-sample reference jitter (full or LOFO)
    run_lofo_matrix        sweep run_mismatch_jitter(condition='lofo') over holdouts
    run_pre_ems_diagonal   EMS-control diagonal (DL only)
    run_pre_ems_mismatch   EMS-control full NxN matrix (DL only; v0.16+)
    run_bandpass_mismatch  bandpass-mismatch control (DL only)
"""

from refshift.experiments.bandpass import run_bandpass_mismatch
from refshift.experiments.calibration import calibrate_csp_lda
from refshift.experiments.cross_subject import run_cross_subject_mismatch
from refshift.experiments.ems_control import run_pre_ems_diagonal
from refshift.experiments.jitter import run_lofo_matrix, run_mismatch_jitter
from refshift.experiments.mismatch import mismatch_matrix, run_mismatch
from refshift.experiments.pre_ems_mismatch import run_pre_ems_mismatch


__all__ = [
    "calibrate_csp_lda",
    "mismatch_matrix",
    "run_bandpass_mismatch",
    "run_cross_subject_mismatch",
    "run_lofo_matrix",
    "run_mismatch",
    "run_mismatch_jitter",
    "run_pre_ems_diagonal",
    "run_pre_ems_mismatch",
]
