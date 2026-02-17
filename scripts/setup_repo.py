#!/usr/bin/env python3
"""Generate a GitHub-ready repository bundle with docs and instructions."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "submission" / "github_repo"
README_TEMPLATE = """# Attribution Fingerprints for Chest X-ray Shift Detection

## Overview
This repository packages the scripts, documentation, and instructions needed to reproduce the NIH ChestXray14 attribution fingerprint pipeline described in our manuscript. Large datasets are **not** included due to NIH data-use restrictions; instead, we provide download/QA guides and configuration templates.

## Installation
```
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset Setup
1. Request NIH ChestXray14 access (see `docs/NIH_DATA_DOWNLOAD.md`).
2. Update `configs/paths.yaml` with local dataset/model paths.
3. Run `python scripts/prepare_nih_chestxray14.py` followed by `python scripts/generate_nih_masks.py`.

## Reproducibility Checklist
1. `bash run_all_analyses.sh` – regenerates divergence tables, bbox-stratified statistics, and figures.
2. `python scripts/run_nih_mask_qc.py --samples 300 --seed 2025` – verifies mask QA remains at 100% pass.
3. `python scripts/sensitivity_analysis.py` – recomputes threshold/bbox ablations and updates `results/tables/sensitivity/`.

## Citation
If you use this code, please cite our forthcoming journal article and the NIH dataset creators.
"""

REQUIREMENTS = """torch==2.3.0
numpy==1.26.4
pandas==2.2.2
scipy==1.16.3
seaborn==0.13.2
pyyaml==6.0.1
matplotlib==3.8.4
"""

LICENSE_TEXT = """MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

DOC_FILES = [
    PROJECT_ROOT / "docs" / "NIH_DATA_DOWNLOAD.md",
    PROJECT_ROOT / "docs" / "NIH_QA_METHODS.md",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bundle repo assets for GitHub.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        shutil.rmtree(args.output)
    (args.output / "docs").mkdir(parents=True, exist_ok=True)
    (args.output / "scripts").mkdir(parents=True, exist_ok=True)
    (args.output / "configs").mkdir(parents=True, exist_ok=True)

    (args.output / "README.md").write_text(README_TEMPLATE, encoding="utf-8")
    (args.output / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")
    (args.output / "LICENSE").write_text(LICENSE_TEXT, encoding="utf-8")

    for doc in DOC_FILES:
        shutil.copy2(doc, args.output / "docs" / doc.name)

    template_config = PROJECT_ROOT / "configs" / "paths.yaml"
    shutil.copy2(template_config, args.output / "configs" / "paths.yaml.template")

    shutil.copy2(PROJECT_ROOT / "scripts" / "run_nih_mask_qc.py", args.output / "scripts" / "run_nih_mask_qc.py")
    shutil.copy2(PROJECT_ROOT / "scripts" / "sensitivity_analysis.py", args.output / "scripts" / "sensitivity_analysis.py")

    print(f"Repo bundle created at {args.output}")


if __name__ == "__main__":
    main()
