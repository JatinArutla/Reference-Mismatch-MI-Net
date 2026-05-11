"""Experiment runners for refshift.

Public entry points (re-exported at refshift package root):

    calibrate_csp_lda      MOABB calibration on CSP+LDA
    run_mismatch           6x6 train-test reference mismatch matrix
    mismatch_matrix        long-form -> 6x6 pivot
    run_mismatch_jitter    DL with per-sample reference jitter (full or LOFO)
    run_lofo_matrix        sweep run_mismatch_jitter(condition='lofo') over holdouts
    run_pre_ems_diagonal   EMS-control diagonal (DL only)
    run_bandpass_mismatch  bandpass-mismatch control (DL only)
"""

from refshift.experiments.bandpass import run_bandpass_mismatch
from refshift.experiments.calibration import calibrate_csp_lda
from refshift.experiments.ems_control import run_pre_ems_diagonal
from refshift.experiments.jitter import run_lofo_matrix, run_mismatch_jitter
from refshift.experiments.mismatch import mismatch_matrix, run_mismatch


__all__ = [
    "calibrate_csp_lda",
    "mismatch_matrix",
    "run_bandpass_mismatch",
    "run_lofo_matrix",
    "run_mismatch",
    "run_mismatch_jitter",
    "run_pre_ems_diagonal",
]
