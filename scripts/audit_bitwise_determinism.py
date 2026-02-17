#!/usr/bin/env python3
"""Audit bitwise determinism by comparing two replay output trees."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bitwise determinism audit for replay runs.")
    parser.add_argument("--root", type=Path, default=Path("reports_v2/gate5/determinism"))
    parser.add_argument("--run-a", default="run1")
    parser.add_argument("--run-b", default="run2")
    parser.add_argument("--experiment", default="jsrt_to_montgomery")
    parser.add_argument("--endpoints", nargs="+", default=["predicted_mask", "mask_free"])
    parser.add_argument("--out-csv", type=Path, default=Path("reports_v2/audits/GATE5_BITWISE_DETERMINISM.csv"))
    parser.add_argument("--out-json", type=Path, default=Path("reports_v2/audits/GATE5_BITWISE_DETERMINISM_SUMMARY.json"))
    parser.add_argument("--out-md", type=Path, default=Path("reports_v2/audits/GATE5_BITWISE_DETERMINISM_2026-02-17.md"))
    return parser.parse_args()


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _tree_manifest(root: Path) -> tuple[str, dict[str, str]]:
    files = sorted([p for p in root.rglob("*") if p.is_file()])
    rel_to_hash: dict[str, str] = {}
    for file_path in files:
        rel = file_path.relative_to(root).as_posix()
        rel_to_hash[rel] = _hash_file(file_path)

    h = hashlib.sha256()
    for rel in sorted(rel_to_hash):
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(rel_to_hash[rel].encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest(), rel_to_hash


def main() -> None:
    args = parse_args()
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    all_match = True

    for endpoint in args.endpoints:
        dir_a = args.root / args.run_a / endpoint / args.experiment
        dir_b = args.root / args.run_b / endpoint / args.experiment

        if not dir_a.exists() or not dir_b.exists():
            rows.append(
                {
                    "endpoint": endpoint,
                    "dir_a": str(dir_a),
                    "dir_b": str(dir_b),
                    "hash_a": "",
                    "hash_b": "",
                    "match": False,
                    "file_count_a": 0,
                    "file_count_b": 0,
                    "mismatch_count": -1,
                    "missing_dir": True,
                }
            )
            all_match = False
            continue

        hash_a, map_a = _tree_manifest(dir_a)
        hash_b, map_b = _tree_manifest(dir_b)

        keys = sorted(set(map_a.keys()) | set(map_b.keys()))
        mismatch = 0
        for key in keys:
            if map_a.get(key, "") != map_b.get(key, ""):
                mismatch += 1

        match = bool(hash_a == hash_b and mismatch == 0)
        all_match = all_match and match

        rows.append(
            {
                "endpoint": endpoint,
                "dir_a": str(dir_a),
                "dir_b": str(dir_b),
                "hash_a": hash_a,
                "hash_b": hash_b,
                "match": match,
                "file_count_a": len(map_a),
                "file_count_b": len(map_b),
                "mismatch_count": mismatch,
                "missing_dir": False,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)

    match_rate = float(df["match"].mean()) if len(df) else float("nan")
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(args.root),
        "run_a": args.run_a,
        "run_b": args.run_b,
        "experiment": args.experiment,
        "endpoints": args.endpoints,
        "match_rate": match_rate,
        "all_match": bool(all_match),
        "rows": rows,
    }
    args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Gate-5 Bitwise Determinism Re-Audit",
        "",
        f"- Generated UTC: `{summary['generated_utc']}`",
        f"- Root: `{args.root}`",
        f"- Replay A/B: `{args.run_a}` vs `{args.run_b}`",
        f"- Experiment: `{args.experiment}`",
        "",
        "| Endpoint | Match | Files A | Files B | Mismatched files | Hash A | Hash B |",
        "|---|---|---:|---:|---:|---|---|",
    ]

    for row in rows:
        hash_a = str(row["hash_a"])[:12]
        hash_b = str(row["hash_b"])[:12]
        lines.append(
            "| "
            f"{row['endpoint']} | "
            f"{'PASS' if row['match'] else 'FAIL'} | "
            f"{row['file_count_a']} | {row['file_count_b']} | {row['mismatch_count']} | "
            f"`{hash_a}` | `{hash_b}` |"
        )

    lines.extend(
        [
            "",
            "## Final Decision",
            "",
            f"- Match rate: `{match_rate:.2f}`",
            f"- Bitwise deterministic: `{'PASS' if all_match else 'FAIL'}`",
            f"- CSV: `{args.out_csv}`",
            f"- JSON: `{args.out_json}`",
        ]
    )

    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[determinism] all_match={all_match} match_rate={match_rate:.2f}")
    print(f"[determinism] csv -> {args.out_csv}")
    print(f"[determinism] md  -> {args.out_md}")


if __name__ == "__main__":
    main()
