"""Operator invertibility control (data-free algebraic check).

Prints, per reference operator on a montage: whether it is canonicalizable by
re-referencing (contrast_preserving, H M = H), whether the native channel
contrasts are still linearly recoverable from its output (contrasts_recoverable),
and how well-conditioned that recovery is on the contrast subspace (cond_contrast).

This underpins the paper's global-vs-spatial claim: it shows global re-references
collapse to a common reference, while Laplacians change the coordinate system but
(on this montage) lose no contrast information and invert with a small condition
number -- so a fixed decoder's failure on them is a coordinate mismatch, not lost
information or an unstable inverse.

Usage:
    python scripts/inversion_control.py            # IV-2a 22-channel montage
    python scripts/inversion_control.py --schirr   # Schirrmeister 44-channel motor set
"""

import sys

from refshift.datasets import SCHIRRMEISTER_MOTOR_CHANNELS
from refshift.inversion import contrast_recovery_report

# IV-2a 22-channel montage (the primary dataset).
IV2A_CHANNELS = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2",
    "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz",
]


def main() -> None:
    if "--schirr" in sys.argv:
        ch_names, label = list(SCHIRRMEISTER_MOTOR_CHANNELS), "Schirrmeister (44 ch)"
    else:
        ch_names, label = IV2A_CHANNELS, "IV-2a (22 ch)"

    df = contrast_recovery_report(ch_names)
    print(f"Operator invertibility -- {label}")
    print(df.to_string(index=False))

    spatial = df[~df["contrast_preserving"]]
    print(
        "\nRead: contrast-preserving operators are canonicalizable to a common "
        "reference (car_after collapses them).\n"
        "Contrast-transforming operators are not, but if contrasts_recoverable is "
        "True their\ntask-relevant contrasts are recoverable by a linear inverse; "
        "cond_contrast says how hard\n(~1 trivial, large = ill-conditioned). "
        f"Spatial cond range here: "
        f"{spatial['cond_contrast'].min():.1f}-{spatial['cond_contrast'].max():.1f}."
    )


if __name__ == "__main__":
    main()
