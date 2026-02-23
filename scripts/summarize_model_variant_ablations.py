#!/usr/bin/env python3
"""Aggregate model-variant fingerprint comparison reports into one ablation table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _load_training_meta(path: Path) -> dict[str, Any]:
    meta_path = path / "training_metadata.json"
    if not meta_path.exists():
        return {}
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _variant_label(loss: str | None, features: list[int] | None, *, canonical_loss: str, canonical_features: list[int]) -> str:
    if loss is None or features is None:
        return "unknown"
    loss_changed = loss != canonical_loss
    features_changed = features != canonical_features
    if loss_changed and features_changed:
        return "combined_change"
    if loss_changed:
        return "loss_only"
    if features_changed:
        return "capacity_only"
    return "canonical_clone"


def _load_variant_rows(
    variant_dir: Path,
    *,
    experiment: str,
    canonical_loss: str,
    canonical_features: list[int],
) -> list[dict[str, Any]]:
    csv_path = variant_dir / f"model_variant_fingerprint_comparison_{experiment}.csv"
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path)
    meta = _load_training_meta(variant_dir)
    loss = meta.get("loss")
    features = meta.get("features")
    variant_type = _variant_label(loss, features, canonical_loss=canonical_loss, canonical_features=canonical_features)

    rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        rows.append(
            {
                "variant_tag": variant_dir.name,
                "variant_type": variant_type,
                "loss": loss,
                "features": ",".join(str(x) for x in features) if isinstance(features, list) else None,
                "endpoint": r["endpoint"],
                "canonical_auc": r.get("canonical_auc"),
                "variant_auc": r.get("variant_auc"),
                "delta_auc": r.get("delta_auc_variant_minus_canonical"),
                "canonical_aupr": r.get("canonical_aupr"),
                "variant_aupr": r.get("variant_aupr"),
                "delta_aupr": r.get("delta_aupr_variant_minus_canonical"),
                "canonical_fpr95": r.get("canonical_fpr95"),
                "variant_fpr95": r.get("variant_fpr95"),
                "delta_fpr95": r.get("delta_fpr95_variant_minus_canonical"),
                "topk_jaccard": r.get("topk_jaccard"),
                "effect_corr_pearson": r.get("effect_corr_pearson"),
                "score_corr_id": r.get("score_corr_id"),
                "score_corr_ood": r.get("score_corr_ood"),
                "n_match_id": r.get("n_match_id"),
                "n_match_ood": r.get("n_match_ood"),
            }
        )
    return rows


def _format_num(x: Any, ndigits: int = 4) -> str:
    if x is None or pd.isna(x):
        return "NA"
    return f"{float(x):.{ndigits}f}"


def _write_markdown(df: pd.DataFrame, out_md: Path, *, experiment: str) -> None:
    lines: list[str] = []
    lines.append(f"# Model Variant Ablation Summary ({experiment})")
    lines.append("")
    if df.empty:
        lines.append("No variant comparison reports found.")
        out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    lines.append(f"Variants found: `{df['variant_tag'].nunique()}`")
    lines.append("")

    for endpoint in sorted(df["endpoint"].unique()):
        sub = df[df["endpoint"] == endpoint].copy()
        sub = sub.sort_values(["variant_type", "variant_tag"])
        lines.append(f"## Endpoint: `{endpoint}`")
        lines.append("")
        lines.append("| Variant | Type | Loss | Features | dAUROC | dAUPR | dFPR95 | TopK Jaccard | Effect Corr | Score Corr ID | Score Corr OOD |")
        lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for _, r in sub.iterrows():
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(r["variant_tag"]),
                        str(r["variant_type"]),
                        str(r["loss"] or "NA"),
                        str(r["features"] or "NA"),
                        _format_num(r["delta_auc"]),
                        _format_num(r["delta_aupr"]),
                        _format_num(r["delta_fpr95"]),
                        _format_num(r["topk_jaccard"]),
                        _format_num(r["effect_corr_pearson"]),
                        _format_num(r["score_corr_id"]),
                        _format_num(r["score_corr_ood"]),
                    ]
                )
                + " |"
            )
        lines.append("")

        # Short machine-generated endpoint takeaway
        if {"delta_auc", "variant_type"}.issubset(sub.columns):
            by_type = sub.groupby("variant_type", dropna=False)["delta_auc"].mean().sort_values(ascending=False)
            if not by_type.empty:
                best_type = by_type.index[0]
                lines.append(f"Takeaway: best mean `dAUROC` for this endpoint is `{best_type}` ({_format_num(by_type.iloc[0])}).")
                lines.append("")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants-root", type=Path, default=Path("reports_v2/audits/model_variants"))
    ap.add_argument("--experiment", default="jsrt_to_montgomery")
    ap.add_argument("--canonical-loss", default="bce_dice")
    ap.add_argument("--canonical-features", default="32,64,128,256")
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--out-md", type=Path, required=True)
    args = ap.parse_args()

    canonical_features = [int(x) for x in args.canonical_features.split(",")]

    rows: list[dict[str, Any]] = []
    for variant_dir in sorted(args.variants_root.iterdir()):
        if not variant_dir.is_dir():
            continue
        rows.extend(
            _load_variant_rows(
                variant_dir,
                experiment=args.experiment,
                canonical_loss=args.canonical_loss,
                canonical_features=canonical_features,
            )
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        sort_cols = ["endpoint", "variant_type", "variant_tag"]
        df = df.sort_values(sort_cols).reset_index(drop=True)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    _write_markdown(df, args.out_md, experiment=args.experiment)

    print(f"Wrote {len(df)} rows to {args.out_csv}")
    print(f"Wrote markdown summary to {args.out_md}")


if __name__ == "__main__":
    main()
