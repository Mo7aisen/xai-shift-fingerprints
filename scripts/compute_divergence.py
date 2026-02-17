#!/usr/bin/env python
"""Compute shift metrics between stored fingerprints."""

from __future__ import annotations

import _path_setup  # noqa: F401 - ensures xfp is importable

import argparse
from pathlib import Path


from xfp.shift.divergence import compute_shift_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute divergence between two fingerprints.")
    parser.add_argument("--reference", required=True, help="Path to reference fingerprint file.")
    parser.add_argument("--target", required=True, help="Path to target fingerprint file.")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["emd", "kl_divergence", "graph_edit_distance"],
        help="Shift metrics to compute.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write JSON results. Prints to stdout if omitted.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference = Path(args.reference)
    target = Path(args.target)
    scores = compute_shift_scores(reference, target, metrics=args.metrics)
    if args.output:
        Path(args.output).write_text(scores.model_dump_json(indent=2))
    else:
        print(scores.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
