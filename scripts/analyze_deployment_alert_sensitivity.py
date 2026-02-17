#!/usr/bin/env python3
"""
Deployment Alert Sensitivity Analysis
=====================================

Purpose & Output:
-----------------
Estimates how often attribution-based alerts fire for Shenzhen monitoring
under different weekly sample sizes. Uses bootstrap resampling of target
cohorts (≥10,000 iterations) and compares against reference fingerprints.
Generates CSV/Markdown/LaTeX tables with Wilson confidence intervals.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import logging

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINGERPRINT_ROOT = PROJECT_ROOT / "data" / "fingerprints"
REPORTS_DIR = PROJECT_ROOT / "reports" / "deployment"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
LOGGER = logging.getLogger("deployment_alert_sensitivity")

ALERT_THRESHOLDS = {
    "attr_ratio_high": 1.25,
    "attr_ratio_low": 0.75,
    "dice_drop_pct": -5.0,  # negative indicates drop
}


@dataclass
class Scenario:
    label: str
    folder: str
    reference_key: str
    target_key: str


SCENARIOS = [
    Scenario("JSRT→Shenzhen", "jsrt_to_shenzhen", "jsrt", "shenzhen"),
    Scenario("Montgomery→Shenzhen", "montgomery_to_shenzhen", "montgomery", "shenzhen"),
]


def load_fingerprints(scenario: Scenario) -> Dict[str, pd.DataFrame]:
    """Load reference and target fingerprints for a scenario."""
    base = FINGERPRINT_ROOT / scenario.folder
    LOGGER.info("Loading fingerprints for %s from %s", scenario.label, base)
    ref_df = pd.read_parquet(base / f"{scenario.reference_key}.parquet")
    tgt_df = pd.read_parquet(base / f"{scenario.target_key}.parquet")
    return {"reference": ref_df, "target": tgt_df}


def _wilson_ci(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Compute Wilson score interval for a Bernoulli proportion."""
    if total == 0:
        return (0.0, 0.0)
    z = 1.959963984540054  # 95% CI
    phat = successes / total
    denom = 1 + z**2 / total
    centre = phat + z**2 / (2 * total)
    margin = z * np.sqrt((phat * (1 - phat) + z**2 / (4 * total)) / total)
    lower = max(0.0, (centre - margin) / denom)
    upper = min(1.0, (centre + margin) / denom)
    return lower, upper


def bootstrap_alert_probability(
    ref_df: pd.DataFrame,
    tgt_df: pd.DataFrame,
    sample_size: int,
    iterations: int,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """Bootstrap probability that weekly samples trigger each alert."""
    ref_attr = ref_df["attribution_abs_sum"].mean()
    ref_dice = ref_df["dice"].mean()

    attr_hits = 0
    dice_hits = 0

    for _ in range(iterations):
        sample_indices = rng.integers(0, len(tgt_df), size=sample_size)
        sample = tgt_df.iloc[sample_indices]
        attr_ratio = sample["attribution_abs_sum"].mean() / ref_attr if ref_attr else np.nan
        dice_drop = (sample["dice"].mean() - ref_dice) / ref_dice * 100 if ref_dice else np.nan

        if attr_ratio >= ALERT_THRESHOLDS["attr_ratio_high"] or attr_ratio <= ALERT_THRESHOLDS["attr_ratio_low"]:
            attr_hits += 1

        if not np.isnan(dice_drop) and dice_drop <= ALERT_THRESHOLDS["dice_drop_pct"]:
            dice_hits += 1

    attr_ci = _wilson_ci(attr_hits, iterations)
    dice_ci = _wilson_ci(dice_hits, iterations)
    return {
        "Attr Alert Rate (%)": round(attr_hits / iterations * 100, 1),
        "Attr Alert 95% CI (%)": f"[{attr_ci[0]*100:.1f}, {attr_ci[1]*100:.1f}]",
        "Dice Alert Rate (%)": round(dice_hits / iterations * 100, 1),
        "Dice Alert 95% CI (%)": f"[{dice_ci[0]*100:.1f}, {dice_ci[1]*100:.1f}]",
    }


def analyze(sample_sizes: List[int], iterations: int, seed: int) -> pd.DataFrame:
    """Run sensitivity analysis for all scenarios."""
    rows = []
    for scenario in SCENARIOS:
        data = load_fingerprints(scenario)
        for size in sample_sizes:
            scenario_seed = seed + hash((scenario.label, size)) % (2**32 - 1)
            rng = np.random.default_rng(scenario_seed)
            rates = bootstrap_alert_probability(data["reference"], data["target"], size, iterations, rng)
            rows.append(
                {
                    "Scenario": scenario.label,
                    "Weekly Sample Size": size,
                    **rates,
                }
            )
    return pd.DataFrame(rows)


def save_outputs(df: pd.DataFrame) -> Dict[str, Path]:
    """Save CSV, Markdown, and LaTeX versions."""
    csv_path = REPORTS_DIR / "deployment_alert_sensitivity.csv"
    md_path = REPORTS_DIR / "deployment_alert_sensitivity.md"
    tex_path = REPORTS_DIR / "deployment_alert_sensitivity.tex"
    df.to_csv(csv_path, index=False)
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("# Deployment Alert Sensitivity\n\n")
        handle.write("Bootstrap probability that weekly monitoring detects Shenzhen drift.\n\n")
        handle.write(df.to_markdown(index=False))
        handle.write("\n")

    lines = [
        "\\begin{table}[t]",
        "    \\centering",
        "    \\caption{Bootstrap alert sensitivity for Shenzhen monitoring (10,000 iterations, Wilson 95\\% CI).}",
        "    \\label{tab:alert-sensitivity}",
        "    \\begin{tabular}{lccc}",
        "        \\toprule",
        "        Scenario & Weekly $N$ & Attr Alert Rate (\\%) & Dice Alert Rate (\\%) \\\\",
        "        \\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(
            "        {scenario} & {size} & {attr:.1f}~{attr_ci} & {dice:.1f}~{dice_ci} \\\\".format(
                scenario=row["Scenario"].replace("→", "\\textrightarrow "),
                size=row["Weekly Sample Size"],
                attr=row["Attr Alert Rate (%)"],
                attr_ci=row["Attr Alert 95% CI (%)"],
                dice=row["Dice Alert Rate (%)"],
                dice_ci=row["Dice Alert 95% CI (%)"],
            )
        )
    lines.extend(["        \\bottomrule", "    \\end{tabular}", "\\end{table}"])
    tex_path.write_text("\n".join(lines), encoding="utf-8")

    LOGGER.info("Saved alert sensitivity outputs to %s / %s / %s", csv_path, md_path, tex_path)
    return {"csv": csv_path, "md": md_path, "tex": tex_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deployment alert sensitivity analysis.")
    parser.add_argument(
        "--sample-sizes",
        type=int,
        nargs="+",
        default=[25, 50, 100],
        help="Weekly sample sizes to evaluate.",
    )
    parser.add_argument("--iterations", type=int, default=10000, help="Bootstrap iterations (>=10000 recommended).")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducible bootstrap sampling.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    LOGGER.info(
        "Running alert sensitivity with samples=%s iterations=%d seed=%d",
        args.sample_sizes,
        args.iterations,
        args.seed,
    )
    df = analyze(args.sample_sizes, args.iterations, args.seed)
    outputs = save_outputs(df)
    LOGGER.info("Saved deployment alert sensitivity outputs: %s", outputs)


if __name__ == "__main__":
    main()
