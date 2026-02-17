#!/usr/bin/env python3
"""Update analysis notes with a Shenzhen vs NIH summary line."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = PROJECT_ROOT / "results" / "metrics" / "divergence" / "hypothesis_tests.csv"
DEFAULT_NOTES = [
    PROJECT_ROOT / "reports" / "analysis_notes.md",
    PROJECT_ROOT / "journal_submission_bundle" / "analysis_notes.md",
    PROJECT_ROOT / "journal_submission_bundle" / "reports" / "analysis_notes.md",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update analysis notes with Shenzhen vs NIH summary.")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS, help="Path to hypothesis_tests.csv")
    parser.add_argument(
        "--notes",
        type=Path,
        action="append",
        help="Analysis notes file to update (repeatable).",
    )
    return parser.parse_args()


def _format_p_value(p_value: float) -> str:
    if pd.isna(p_value):
        return "p=NA"
    if p_value < 0.001:
        return "p<0.001"
    return f"p={p_value:.3f}"


def _build_summary(df: pd.DataFrame) -> str:
    p_col = "permutation_p_fdr" if "permutation_p_fdr" in df.columns else "permutation_p"
    label = "permutation FDR " if p_col == "permutation_p_fdr" else "permutation "

    df = df[df["comparison"] == "Shenzhen Baseline vs NIH"].copy()
    if df.empty:
        raise ValueError("No Shenzhen Baseline vs NIH rows found in hypothesis_tests.csv")

    df = df[df[p_col].notna()]
    if df.empty:
        raise ValueError(f"No valid {p_col} values found for Shenzhen Baseline vs NIH")

    row = df.sort_values(p_col).iloc[0]
    metric_map = {
        "dice": "Dice",
        "coverage_auc": "Coverage AUC",
        "attribution_abs_sum": "Attr Mass",
        "border_abs_sum": "Border Mass",
        "hist_entropy": "Entropy",
    }
    metric = metric_map.get(str(row["metric"]), str(row["metric"]).replace("_", " ").title())
    p_str = _format_p_value(float(row[p_col]))
    if "cliffs_delta" in row:
        effect = row["cliffs_delta"]
        effect_label = "Cliff's delta"
    else:
        effect = row["permutation_effect_size"] if "permutation_effect_size" in row else float("nan")
        effect_label = "Cohen's d"
    effect_str = f"{effect:.2f}" if pd.notna(effect) else "NA"

    return (
        f"- Shenzhen vs NIH summary: fingerprint divergence detected (best FDR metric: {metric}; "
        f"{label}{p_str}; {effect_label}={effect_str}).\n"
    )


def _update_notes(path: Path, summary_line: str) -> None:
    if not path.exists():
        print(f"[WARN] Notes file missing: {path}")
        return

    lines = path.read_text().splitlines(keepends=True)
    replacement_prefix = "- Shenzhen vs NIH summary:"

    for idx, line in enumerate(lines):
        if line.lstrip().startswith(replacement_prefix):
            lines[idx] = summary_line
            path.write_text("".join(lines))
            print(f"[INFO] Updated summary line in: {path}")
            return

    header = "## External Validation (NIH ChestXray14)"
    header_idx = None
    for idx, line in enumerate(lines):
        if line.strip() == header:
            header_idx = idx
            break

    if header_idx is None:
        if lines and not lines[-1].endswith("\n"):
            lines[-1] = lines[-1] + "\n"
        lines.append("\n" + header + "\n")
        lines.append(summary_line)
        path.write_text("".join(lines))
        print(f"[INFO] Appended new NIH section in: {path}")
        return

    insert_idx = None
    for idx in range(header_idx + 1, len(lines)):
        if lines[idx].startswith("| Comparison"):
            insert_idx = idx
            break

    if insert_idx is None:
        insert_idx = header_idx + 1
        if insert_idx < len(lines) and lines[insert_idx].strip():
            lines.insert(insert_idx, "\n")
            insert_idx += 1

    lines.insert(insert_idx, summary_line)
    path.write_text("".join(lines))
    print(f"[INFO] Inserted summary line in: {path}")


def main() -> None:
    args = parse_args()
    notes_paths = args.notes if args.notes else DEFAULT_NOTES

    if not args.results.exists():
        raise FileNotFoundError(f"Hypothesis tests not found: {args.results}")

    df = pd.read_csv(args.results)
    summary_line = _build_summary(df)

    for notes_path in notes_paths:
        _update_notes(notes_path, summary_line)


if __name__ == "__main__":
    main()
