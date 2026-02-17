#!/usr/bin/env python3
"""Gate-6 reproducibility packaging.

Builds frozen manifests and a final reproducibility bundle after Gates 1-5 PASS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import shutil


@dataclass
class GateCheck:
    name: str
    passed: bool
    detail: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Gate-6 reproducibility package.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--date-tag", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    parser.add_argument("--bundle-tag", default="gate6_final")
    parser.add_argument("--out-audit-json", type=Path, default=Path("reports_v2/audits/GATE6_REPRO_SUMMARY.json"))
    parser.add_argument("--out-audit-md", type=Path, default=Path("reports_v2/audits/GATE6_REPRODUCIBILITY_2026-02-17.md"))
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _iter_files(root: Path, rel_roots: list[Path]) -> list[Path]:
    out: list[Path] = []
    for rel in rel_roots:
        p = root / rel
        if not p.exists():
            continue
        if p.is_file():
            out.append(p)
            continue
        for f in sorted(p.rglob("*")):
            if not f.is_file():
                continue
            if "__pycache__" in f.parts:
                continue
            out.append(f)
    return out


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _gate_checks(root: Path) -> list[GateCheck]:
    checks: list[GateCheck] = []

    g3_path = root / "reports_v2/audits/GATE3_FULL_SEEDS_SUMMARY.json"
    g4_path = root / "reports_v2/audits/GATE4_IG_ROBUSTNESS_SUMMARY.json"
    g5_path = root / "reports_v2/audits/GATE5_FINAL_SUMMARY_REMEDIATED.json"
    det_path = root / "reports_v2/audits/GATE5_BITWISE_DETERMINISM_SUMMARY.json"
    proto_path = root / "configs/protocol_lock_v1.yaml"
    reg_path = root / "reports_v2/run_registry.csv"

    for p in [g3_path, g4_path, g5_path, det_path, proto_path, reg_path]:
        checks.append(GateCheck(name=f"exists:{p.as_posix()}", passed=p.exists(), detail="present" if p.exists() else "missing"))

    if g3_path.exists():
        g3 = _load_json(g3_path)
        ok = bool(g3.get("gate3_statistical_full_pass", False))
        checks.append(GateCheck(name="gate3_statistical_full_pass", passed=ok, detail=str(g3.get("gate3_statistical_full_pass"))))

    if g4_path.exists():
        g4 = _load_json(g4_path)
        ok = bool(g4.get("gate4_pass", False))
        checks.append(GateCheck(name="gate4_pass", passed=ok, detail=str(g4.get("gate4_pass"))))

    if g5_path.exists():
        g5 = _load_json(g5_path)
        ok = bool(g5.get("gate5_final_pass", False))
        checks.append(GateCheck(name="gate5_final_pass", passed=ok, detail=str(g5.get("gate5_final_pass"))))

    if det_path.exists():
        det = _load_json(det_path)
        ok = bool(det.get("all_match", False))
        checks.append(GateCheck(name="determinism_all_match", passed=ok, detail=str(det.get("all_match"))))

    if reg_path.exists():
        lines = [ln for ln in reg_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        checks.append(GateCheck(name="run_registry_nonempty", passed=len(lines) > 1, detail=f"rows={max(len(lines)-1, 0)}"))

    return checks


def _write_gate_artifacts_manifest(root: Path, out_path: Path) -> tuple[int, str]:
    targets = [
        Path("reports_v2/audits"),
        Path("reports_v2/gate3_seed_artifacts"),
        Path("reports_v2/gate4"),
        Path("reports_v2/gate5"),
        Path("reports_v2/run_registry.csv"),
    ]
    files = _iter_files(root, targets)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for file_path in files:
            rel = file_path.relative_to(root).as_posix()
            fh.write(f"{_sha256_file(file_path)}  {rel}\n")
    return len(files), _sha256_file(out_path)


def _copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_tree_if_exists(src: Path, dst: Path) -> None:
    if not src.exists() or not src.is_dir():
        return
    shutil.copytree(src, dst, dirs_exist_ok=True)


def _build_bundle(root: Path, out_dir: Path, date_tag: str, bundle_tag: str) -> tuple[Path, Path, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle_name = f"GATE6_REPRO_BUNDLE_{bundle_tag}_{date_tag}.tar.gz"
    bundle_path = out_dir / bundle_name
    filelist_path = out_dir / f"GATE6_REPRO_BUNDLE_FILELIST_{bundle_tag}_{date_tag}.txt"

    with TemporaryDirectory() as td:
        stage = Path(td) / "bundle"
        stage.mkdir(parents=True, exist_ok=True)

        # Core reproducibility controls
        _copy_tree_if_exists(root / "configs", stage / "configs")
        _copy_tree_if_exists(root / "src", stage / "src")
        _copy_tree_if_exists(root / "tests", stage / "tests")
        _copy_tree_if_exists(root / "scripts", stage / "scripts")

        # Governance / audit artifacts
        _copy_tree_if_exists(root / "reports_v2/audits", stage / "reports_v2/audits")
        _copy_tree_if_exists(root / "reports_v2/manifests", stage / "reports_v2/manifests")
        _copy_if_exists(root / "reports_v2/run_registry.csv", stage / "reports_v2/run_registry.csv")
        _copy_if_exists(root / "reports_v2/README.md", stage / "reports_v2/README.md")

        # Execution logs for the gate track
        logs_dst = stage / "logs"
        logs_dst.mkdir(parents=True, exist_ok=True)
        logs_src = root / "logs"
        if logs_src.exists():
            for f in sorted(logs_src.glob("slurm_gate*.out")):
                _copy_if_exists(f, logs_dst / f.name)
            for f in sorted(logs_src.glob("slurm_gate*.err")):
                _copy_if_exists(f, logs_dst / f.name)

        # Freeze file list and pack
        staged_files = sorted([p for p in stage.rglob("*") if p.is_file()])
        with filelist_path.open("w", encoding="utf-8") as fh:
            for p in staged_files:
                fh.write(f"{p.relative_to(stage).as_posix()}\n")

        if bundle_path.exists():
            bundle_path.unlink()
        with tarfile.open(bundle_path, "w:gz") as tf:
            tf.add(stage, arcname="gate6_repro_bundle")

    return bundle_path, filelist_path, len(staged_files)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()

    checks = _gate_checks(root)
    checks_pass = all(c.passed for c in checks)

    manifests_dir = root / "reports_v2/manifests"
    gate_manifest = manifests_dir / f"freeze_gate_artifacts_{args.date_tag}.sha256"
    gate_manifest_count, gate_manifest_hash = _write_gate_artifacts_manifest(root, gate_manifest)

    releases_dir = root / "reports_v2/releases"
    bundle_path, filelist_path, staged_count = _build_bundle(
        root=root,
        out_dir=releases_dir,
        date_tag=args.date_tag,
        bundle_tag=args.bundle_tag,
    )
    bundle_hash = _sha256_file(bundle_path)
    bundle_hash_path = releases_dir / f"{bundle_path.name}.sha256"
    bundle_hash_path.write_text(f"{bundle_hash}  {bundle_path.name}\n", encoding="utf-8")

    gate6_pass = bool(checks_pass and bundle_path.exists() and gate_manifest.exists())

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "date_tag": args.date_tag,
        "bundle_tag": args.bundle_tag,
        "gate6_pass": gate6_pass,
        "checks": [c.__dict__ for c in checks],
        "manifests": {
            "gate_artifacts_manifest": str(gate_manifest.relative_to(root)),
            "gate_artifacts_manifest_files": gate_manifest_count,
            "gate_artifacts_manifest_sha256": gate_manifest_hash,
        },
        "bundle": {
            "bundle_path": str(bundle_path.relative_to(root)),
            "bundle_sha256": bundle_hash,
            "bundle_sha256_path": str(bundle_hash_path.relative_to(root)),
            "bundle_filelist": str(filelist_path.relative_to(root)),
            "bundle_file_count": staged_count,
        },
    }

    out_json = root / args.out_audit_json
    out_md = root / args.out_audit_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Gate-6 Reproducibility Package",
        "",
        f"- Generated UTC: `{summary['generated_utc']}`",
        f"- Date tag: `{args.date_tag}`",
        f"- Bundle tag: `{args.bundle_tag}`",
        "",
        "## Integrity Checks",
        "",
        "| Check | Status | Detail |",
        "|---|---|---|",
    ]
    for c in checks:
        lines.append(f"| {c.name} | {'PASS' if c.passed else 'FAIL'} | {c.detail} |")

    lines.extend(
        [
            "",
            "## Frozen Manifests",
            "",
            f"- Gate artifacts manifest: `{summary['manifests']['gate_artifacts_manifest']}`",
            f"- Files hashed: `{summary['manifests']['gate_artifacts_manifest_files']}`",
            f"- Manifest SHA256: `{summary['manifests']['gate_artifacts_manifest_sha256']}`",
            "",
            "## Final Bundle",
            "",
            f"- Bundle: `{summary['bundle']['bundle_path']}`",
            f"- Bundle SHA256: `{summary['bundle']['bundle_sha256']}`",
            f"- SHA file: `{summary['bundle']['bundle_sha256_path']}`",
            f"- File list: `{summary['bundle']['bundle_filelist']}`",
            f"- Included files: `{summary['bundle']['bundle_file_count']}`",
            "",
            "## Final Decision",
            "",
            f"- Gate-6 status: `{'PASS' if gate6_pass else 'NO-GO'}`",
        ]
    )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[gate6] status={'PASS' if gate6_pass else 'NO-GO'}")
    print(f"[gate6] summary_json={out_json}")
    print(f"[gate6] summary_md={out_md}")
    print(f"[gate6] bundle={bundle_path}")


if __name__ == "__main__":
    main()
