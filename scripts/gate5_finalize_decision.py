#!/usr/bin/env python3
"""Combine Gate-5 clinical + determinism into a final PASS/NO-GO decision."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finalize Gate-5 decision.")
    parser.add_argument("--clinical-json", type=Path, required=True)
    parser.add_argument("--determinism-json", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, default=Path("reports_v2/audits/GATE5_FINAL_SUMMARY.json"))
    parser.add_argument("--out-md", type=Path, default=Path("reports_v2/audits/GATE5_FINAL_2026-02-17.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clinical = json.loads(args.clinical_json.read_text(encoding="utf-8"))
    determinism = json.loads(args.determinism_json.read_text(encoding="utf-8"))

    clinical_pass = bool(clinical.get("gate5_pass", False))
    determinism_pass = bool(determinism.get("all_match", False))
    final_pass = clinical_pass and determinism_pass

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "clinical_json": str(args.clinical_json),
        "determinism_json": str(args.determinism_json),
        "clinical_pass": clinical_pass,
        "determinism_pass": determinism_pass,
        "gate5_final_pass": final_pass,
        "clinical_score_method": clinical.get("score_method"),
        "clinical_score_top_k": clinical.get("score_top_k"),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Gate-5 Final Decision",
        "",
        f"- Generated UTC: `{summary['generated_utc']}`",
        f"- Clinical gate: `{'PASS' if clinical_pass else 'NO-GO'}`",
        f"- Determinism gate: `{'PASS' if determinism_pass else 'NO-GO'}`",
        f"- Clinical score method: `{summary['clinical_score_method']}`",
        f"- Clinical score top-k: `{summary['clinical_score_top_k']}`",
        "",
        "## Final Status",
        "",
        f"- Gate-5 final: `{'PASS' if final_pass else 'NO-GO'}`",
        f"- Clinical summary: `{args.clinical_json}`",
        f"- Determinism summary: `{args.determinism_json}`",
        f"- JSON output: `{args.out_json}`",
    ]
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[gate5-final] status={'PASS' if final_pass else 'NO-GO'}")
    print(f"[gate5-final] out_json={args.out_json}")
    print(f"[gate5-final] out_md={args.out_md}")


if __name__ == "__main__":
    main()
