#!/usr/bin/env python3
"""Bundle manuscript-ready assets for journal submission."""

from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_DIR = PROJECT_ROOT / "results" / "submission"
DEFAULT_ZIP = SUBMISSION_DIR / "submission_package.zip"

SNIPPET_FILES = [
    PROJECT_ROOT / "docs" / "manuscript_snippets.md",
    PROJECT_ROOT / "results" / "manuscript" / "methods_qa.tex",
    PROJECT_ROOT / "results" / "manuscript" / "methods_bbox.tex",
    PROJECT_ROOT / "results" / "manuscript" / "deployment_ops.tex",
    PROJECT_ROOT / "results" / "manuscript" / "appendix_pipeline.tex",
    PROJECT_ROOT / "results" / "manuscript" / "cover_letter_template.tex",
]

ANALYSIS_ASSETS = [
    PROJECT_ROOT / "reports" / "external_validation" / "nih_mask_qc_summary.json",
    PROJECT_ROOT / "reports" / "external_validation" / "nih_mask_qc_findings.csv",
    PROJECT_ROOT / "reports" / "external_validation" / "bbox_stratified" / "nih_baseline_size_stats.csv",
    PROJECT_ROOT / "reports" / "external_validation" / "bbox_stratified" / "nih_baseline_tests.json",
    PROJECT_ROOT / "results" / "figures" / "nih_publication",
    PROJECT_ROOT / "reports" / "external_validation" / "nih_mask_qc_plots",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare journal submission package.")
    parser.add_argument("--output", type=Path, default=DEFAULT_ZIP, help="Path to output ZIP archive.")
    return parser.parse_args()


def validate_paths(paths: list[Path]) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing assets: {missing}")


def copy_asset(src: Path, dst_root: Path) -> None:
    target = dst_root / src.relative_to(PROJECT_ROOT)
    target.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, target, dirs_exist_ok=True)
    else:
        shutil.copy2(src, target)


def main() -> None:
    args = parse_args()
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    validate_paths(SNIPPET_FILES + ANALYSIS_ASSETS)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_root = Path(tmpdir)
        for path in SNIPPET_FILES + ANALYSIS_ASSETS:
            copy_asset(path, tmp_root)
        if args.output.exists():
            args.output.unlink()
        with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for file in tmp_root.rglob("*"):
                if file.is_file():
                    arcname = file.relative_to(tmp_root)
                    zf.write(file, arcname)
    print(f"Submission package created: {args.output}")


if __name__ == "__main__":
    main()
