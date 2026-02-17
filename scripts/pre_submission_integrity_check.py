#!/usr/bin/env python3
"""Pre-submission integrity checks for MedIA revision readiness."""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import yaml


@dataclass
class Finding:
    severity: str
    check: str
    message: str


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def has_zero_pvalue(csv_path: Path, columns: Iterable[str]) -> bool:
    if not csv_path.exists():
        return False
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            for col in columns:
                value = (row.get(col) or "").strip()
                if value in {"0", "0.0", "0.00e+00", "0e+00"}:
                    return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Run integrity checks before manuscript submission.")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--report", default=None, help="Optional markdown report path.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    findings: List[Finding] = []

    paths_yaml = root / "configs" / "paths.yaml"
    experiments_yaml = root / "configs" / "experiments.yaml"
    main_tex = root / "submission_medical_image_analysis" / "main.tex"
    rise_csv = root / "reports" / "rise" / "rise_ig_correlation.csv"
    nih_effects = root / "reports" / "enhanced_statistics" / "nih_effect_sizes.csv"

    if not paths_yaml.exists():
        findings.append(Finding("error", "paths.yaml", "Missing configs/paths.yaml"))
    else:
        cfg = yaml.safe_load(paths_yaml.read_text(encoding="utf-8")) or {}
        paths_cfg = cfg.get("paths", {}) or {}
        models_root = Path(os.getenv("XFP_MODELS_ROOT", str(paths_cfg.get("models_root", ".")))).expanduser()
        models = (cfg.get("models") or {})
        jsrt = str(models.get("unet_jsrt_full", ""))
        shenzhen = str(models.get("unet_shenzhen_full", ""))
        if not shenzhen:
            findings.append(Finding("error", "shenzhen_model_path", "unet_shenzhen_full is not set"))
        elif shenzhen == jsrt:
            findings.append(
                Finding(
                    "error",
                    "shenzhen_model_independence",
                    "unet_shenzhen_full points to the same checkpoint as unet_jsrt_full",
                )
            )
        else:
            resolved = Path(shenzhen)
            if not resolved.is_absolute():
                resolved = models_root / resolved
            try:
                exists = resolved.exists()
            except PermissionError:
                findings.append(
                    Finding(
                        "warning",
                        "shenzhen_model_exists_permission",
                        f"Cannot access configured Shenzhen checkpoint path due to permissions: {resolved}",
                    )
                )
            else:
                if not exists:
                    findings.append(
                        Finding(
                            "warning",
                            "shenzhen_model_exists",
                            f"Dedicated Shenzhen checkpoint is configured but missing on disk: {resolved}",
                        )
                    )

    exp_text = read_text(experiments_yaml)
    if "reuses JSRT" in exp_text or "reuse" in exp_text.lower() and "shenzhen_baseline" in exp_text:
        findings.append(
            Finding(
                "error",
                "shenzhen_baseline_wording",
                "Shenzhen baseline text still implies JSRT checkpoint reuse",
            )
        )

    manuscript_text = read_text(main_tex)
    if "median effect-size changes below 10\\%" in manuscript_text:
        findings.append(
            Finding(
                "error",
                "robustness_overclaim",
                "Manuscript still contains the below-10% robustness overclaim",
            )
        )
    if "RISE" not in manuscript_text and rise_csv.exists():
        findings.append(
            Finding(
                "warning",
                "rise_reporting",
                "RISE results file exists, but manuscript text does not mention RISE validation",
            )
        )

    if has_zero_pvalue(nih_effects, columns=("levene_p_raw", "levene_p_log")):
        findings.append(
            Finding(
                "warning",
                "zero_pvalues",
                "nih_effect_sizes.csv contains exact zero p-values (consider bounded reporting)",
            )
        )

    errors = [f for f in findings if f.severity == "error"]
    warnings = [f for f in findings if f.severity == "warning"]

    lines = [
        "# Pre-submission Integrity Check",
        "",
        f"- Root: `{root}`",
        f"- Errors: {len(errors)}",
        f"- Warnings: {len(warnings)}",
        "",
        "## Findings",
    ]
    if findings:
        for f in findings:
            lines.append(f"- [{f.severity.upper()}] `{f.check}`: {f.message}")
    else:
        lines.append("- No blockers detected.")

    report_path = Path(args.report) if args.report else root / "reports" / "pre_submission_integrity_check.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"report={report_path}")
    print(f"errors={len(errors)} warnings={len(warnings)}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
