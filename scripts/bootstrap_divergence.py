#!/usr/bin/env python
"""Bootstrap divergence metrics to estimate confidence intervals."""

from __future__ import annotations

import _path_setup  # noqa: F401 - ensures xfp is importable


import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
from typing import Dict, List

import numpy as np
import pandas as pd


from xfp.shift.divergence import bootstrap_shift_scores, compute_shift_scores  # noqa: E402

FINGERPRINT_ROOT = PROJECT_ROOT / "data" / "fingerprints"

EXPERIMENTS: Dict[str, Dict[str, str]] = {
    "JSRT → Montgomery": {
        "folder": "jsrt_to_montgomery",
        "ref_key": "jsrt",
        "target_key": "montgomery",
    },
    "JSRT → Shenzhen": {
        "folder": "jsrt_to_shenzhen",
        "ref_key": "jsrt",
        "target_key": "shenzhen",
    },
    "JSRT → NIH": {
        "folder": "jsrt_to_nih",
        "ref_key": "jsrt",
        "target_key": "nih_chestxray14",
    },
    "Montgomery → JSRT": {
        "folder": "montgomery_to_jsrt",
        "ref_key": "montgomery",
        "target_key": "jsrt",
    },
    "Montgomery → Shenzhen": {
        "folder": "montgomery_to_shenzhen",
        "ref_key": "montgomery",
        "target_key": "shenzhen",
    },
    "Montgomery → NIH": {
        "folder": "montgomery_to_nih",
        "ref_key": "montgomery",
        "target_key": "nih_chestxray14",
    },
    "Shenzhen → NIH": {
        "folder": "shenzhen_to_nih",
        "ref_key": "shenzhen",
        "target_key": "nih_chestxray14",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap divergence metrics for confidence intervals.")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["kl_divergence", "emd", "graph_edit_distance"],
        help="Metrics to bootstrap.",
    )
    parser.add_argument(
        "--n-resamples",
        type=int,
        default=512,
        help="Number of bootstrap resamples per comparison.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "divergence" / "divergence_uncertainty.csv",
        help="Path to write the bootstrap summary table.",
    )
    parser.add_argument(
        "--save-samples",
        action="store_true",
        help="Persist raw bootstrap samples alongside the summary table.",
    )
    return parser.parse_args()


def _fingerprint_paths(folder: str, ref_key: str, target_key: str) -> tuple[Path, Path]:
    ref_path = FINGERPRINT_ROOT / folder / f"{ref_key}.parquet"
    tgt_path = FINGERPRINT_ROOT / folder / f"{target_key}.parquet"
    return ref_path, tgt_path


def _summarise_samples(samples: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    summary_rows = []
    for metric in metrics:
        values = samples[metric].to_numpy(dtype=float)
        summary_rows.append(
            {
                "Metric": metric,
                "Mean": float(values.mean()),
                "Std": float(values.std(ddof=1)),
                "CI 2.5%": float(np.quantile(values, 0.025)),
                "CI 97.5%": float(np.quantile(values, 0.975)),
            }
        )
    return pd.DataFrame(summary_rows)


def main() -> None:
    args = parse_args()
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summaries: List[pd.DataFrame] = []
    for comparison, config in EXPERIMENTS.items():
        ref_path, tgt_path = _fingerprint_paths(config["folder"], config["ref_key"], config["target_key"])
        samples = bootstrap_shift_scores(
            reference=ref_path,
            target=tgt_path,
            metrics=args.metrics,
            n_resamples=args.n_resamples,
            random_state=args.seed,
        )
        summary = _summarise_samples(samples, args.metrics)
        baseline = compute_shift_scores(ref_path, tgt_path, args.metrics)
        summary.insert(0, "Comparison", comparison)
        summary["Point Estimate"] = summary["Metric"].map(baseline.scores)
        summaries.append(summary)

        if args.save_samples:
            sample_path = output_path.parent / f"bootstrap_samples_{config['folder']}.parquet"
            samples.assign(resample=np.arange(len(samples))).to_parquet(sample_path, index=False)
            print(f"✓ Saved raw samples for {comparison} → {sample_path}")

    combined = pd.concat(summaries, ignore_index=True)
    combined.to_csv(output_path, index=False)
    print(f"✓ Saved bootstrap summary → {output_path}")

    print("\n=== BOOTSTRAP CONFIDENCE INTERVALS ===")
    print(combined.to_string(index=False, float_format=lambda v: f"{v:.6f}"))


if __name__ == "__main__":
    main()
