"""CLI wrapper around refshift.calibrate_csp_lda.

For notebook usage prefer:
    from refshift import calibrate_csp_lda
    _, summary, passed = calibrate_csp_lda("iv2a")
"""

from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="iv2a",
                   choices=["iv2a", "openbmi", "cho2017", "dreyer2023"])
    p.add_argument("--subjects", type=int, nargs="+", default=None)
    p.add_argument("--random-state", type=int, default=42)
    args = p.parse_args(argv)

    from refshift import calibrate_csp_lda
    _, _, passed = calibrate_csp_lda(
        dataset_id=args.dataset,
        subjects=args.subjects,
        random_state=args.random_state,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
