#!/usr/bin/env python3
from __future__ import annotations

"""Robustness checks and preprocessing-shift alias cache builder.

Modes:
- `robustness_report` (legacy/default): perturb sampled caches and compile robustness summary
- `alias_builder`: create patient-disjoint clean/perturbed cache aliases for harder-shift studies
"""
import _path_setup  # noqa: F401 - ensures xfp is importable

from pathlib import Path
import argparse
import io
import json
import re
import numpy as np
import pandas as pd
from scipy import ndimage, stats
from scipy.stats import rankdata
from PIL import Image

try:  # Optional, but present in the current environment and preferred for CLAHE.
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency fallback
    cv2 = None

from xfp.config import load_experiment_config, load_paths_config, PathsConfig
from xfp.fingerprint.runner import run_fingerprint_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Robustness checks and preprocessing-shift cache alias builder.")
    parser.add_argument(
        "--mode",
        default="robustness_report",
        choices=["robustness_report", "alias_builder"],
        help="Execution mode.",
    )
    parser.add_argument("--datasets", default="jsrt,montgomery,shenzhen", help="Comma-separated dataset keys.")
    parser.add_argument("--subset-size", type=int, default=50, help="Number of samples per dataset.")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device for inference.")
    parser.add_argument(
        "--output-dir",
        default="reports/robustness",
        help="Output directory for robustness artifacts.",
    )
    parser.add_argument(
        "--experiment",
        default="jsrt_to_montgomery",
        help="Experiment key from configs/experiments.yaml.",
    )
    parser.add_argument(
        "--endpoint-mode",
        default="upper_bound_gt",
        choices=["upper_bound_gt", "predicted_mask", "mask_free"],
        help="Fingerprint endpoint mode.",
    )
    parser.add_argument(
        "--ig-steps",
        type=int,
        default=16,
        help="Integrated Gradients interpolation steps.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic runtime controls.",
    )
    # Alias-builder options (harder-shift preprocessing heterogeneity)
    parser.add_argument("--base-dataset", default="shenzhen", help="Base dataset used for clean/perturbed split.")
    parser.add_argument("--source-subset", default="full", help="Source subset under base dataset cache.")
    parser.add_argument(
        "--pilot-subset",
        default="hardshift_pilot",
        help="Subset name written for clean/perturbed aliases (used by run_fingerprint / baselines).",
    )
    parser.add_argument(
        "--id-fraction",
        type=float,
        default=0.50,
        help="Fraction of patients assigned to ID clean split (remaining patients go to OOD perturbed aliases).",
    )
    parser.add_argument(
        "--perturb-specs",
        default="gamma085,gamma115,clahe15,clahe30,jpeg90,jpeg70,downup448,downup320,quant6,quant4",
        help=(
            "Comma-separated preprocessing specs for alias_builder. Supported specs: "
            "gamma085,gamma115,clahe15,clahe30,jpeg90,jpeg70,downup448,downup320,quant6,quant4"
        ),
    )
    parser.add_argument(
        "--alias-cache-root",
        default=None,
        help="Optional cache root for alias_builder output (default: <output-dir>/cache_aliases).",
    )
    parser.add_argument(
        "--manifest-prefix",
        default="preproc_shift_alias_manifest",
        help="Prefix for alias-builder manifest outputs inside --output-dir.",
    )
    return parser.parse_args()


def _cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    mean_diff = np.mean(x) - np.mean(y)
    pooled_std = np.sqrt((np.var(x, ddof=1) + np.var(y, ddof=1)) / 2)
    return mean_diff / pooled_std if pooled_std > 0 else 0.0


def _cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    x = x.reshape(-1, 1)
    y = y.reshape(1, -1)
    greater = np.sum(x > y)
    less = np.sum(x < y)
    denom = x.size * y.size
    return (greater - less) / denom if denom else 0.0


def _compute_shift_stats(ref_df: pd.DataFrame, tgt_df: pd.DataFrame) -> dict:
    results = {}
    metrics = ["dice", "coverage_auc", "attribution_abs_sum", "border_abs_sum", "hist_entropy"]
    if "dice" in ref_df.columns and "dice" in tgt_df.columns:
        ref_df = ref_df.copy()
        tgt_df = tgt_df.copy()
        ref_df["iou"] = ref_df["dice"] / np.clip(2 - ref_df["dice"], 1e-6, None)
        tgt_df["iou"] = tgt_df["dice"] / np.clip(2 - tgt_df["dice"], 1e-6, None)
        metrics.append("iou")

    for metric in metrics:
        if metric not in ref_df.columns or metric not in tgt_df.columns:
            continue
        ref_vals = ref_df[metric].values
        tgt_vals = tgt_df[metric].values
        t_stat, t_pvalue = stats.ttest_ind(ref_vals, tgt_vals)
        results[f"{metric}_ttest_p"] = t_pvalue
        results[f"{metric}_cohens_d"] = _cohens_d(ref_vals, tgt_vals)
        results[f"{metric}_cliffs_delta"] = _cliffs_delta(ref_vals, tgt_vals)
    return results


def _percent_change(new: float, base: float) -> float:
    if not np.isfinite(base) or base == 0:
        return np.nan
    return (new - base) / abs(base) * 100.0


def _roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(score, method="average")
    rank_sum_pos = float(np.sum(ranks[y == 1]))
    return float((rank_sum_pos - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg))


def _shift_auc(ref_df: pd.DataFrame, tgt_df: pd.DataFrame) -> float:
    common_numeric = []
    for col in ref_df.columns:
        if col in {"dataset_key", "subset", "sample_id", "patient_id"} or col.startswith("_"):
            continue
        if col in tgt_df.columns and pd.api.types.is_numeric_dtype(ref_df[col]) and pd.api.types.is_numeric_dtype(tgt_df[col]):
            common_numeric.append(col)
    if not common_numeric:
        return float("nan")

    id_df = ref_df[common_numeric].copy()
    ood_df = tgt_df[common_numeric].copy()
    mu = id_df.mean(axis=0)
    sd = id_df.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)
    score_id = ((id_df - mu) / sd).abs().mean(axis=1).to_numpy(dtype=float)
    score_ood = ((ood_df - mu) / sd).abs().mean(axis=1).to_numpy(dtype=float)
    y = np.concatenate([np.zeros(len(score_id), dtype=int), np.ones(len(score_ood), dtype=int)])
    score = np.concatenate([score_id, score_ood])
    return _roc_auc(y, score)


def _apply_intensity_shift(image: np.ndarray, delta: float) -> np.ndarray:
    return np.clip(image + delta, 0.0, 1.0)


def _apply_gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    return ndimage.gaussian_filter(image, sigma=sigma)


def _apply_salt_pepper(image: np.ndarray, amount: float, salt_vs_pepper: float = 0.5) -> np.ndarray:
    rng = np.random.default_rng()
    noisy = image.copy()
    n_total = image.size
    n_salt = int(amount * n_total * salt_vs_pepper)
    n_pepper = int(amount * n_total * (1 - salt_vs_pepper))
    coords = rng.choice(n_total, size=n_salt + n_pepper, replace=False)
    flat = noisy.reshape(-1)
    flat[coords[:n_salt]] = 1.0
    flat[coords[n_salt:]] = 0.0
    return noisy


def _to_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(np.round(image * 255.0), 0, 255).astype(np.uint8)


def _from_uint8(image_u8: np.ndarray) -> np.ndarray:
    return np.clip(image_u8.astype(np.float32) / 255.0, 0.0, 1.0)


def _apply_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    image = np.clip(image, 0.0, 1.0)
    return np.power(image, gamma, dtype=np.float32)


def _apply_quantize(image: np.ndarray, bits: int) -> np.ndarray:
    levels = max(2, (1 << int(bits)) - 1)
    q = np.round(np.clip(image, 0.0, 1.0) * levels) / levels
    return q.astype(np.float32)


def _apply_downup(image: np.ndarray, size: int) -> np.ndarray:
    img = Image.fromarray(_to_uint8(image), mode="L")
    down = img.resize((int(size), int(size)), resample=Image.BILINEAR)
    up = down.resize((img.width, img.height), resample=Image.BICUBIC)
    return _from_uint8(np.asarray(up))


def _apply_jpeg(image: np.ndarray, quality: int) -> np.ndarray:
    img = Image.fromarray(_to_uint8(image), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(quality), optimize=False)
    buf.seek(0)
    restored = Image.open(buf).convert("L")
    return _from_uint8(np.asarray(restored))


def _apply_clahe(image: np.ndarray, clip_limit: float, tile_grid_size: int = 8) -> np.ndarray:
    image_u8 = _to_uint8(image)
    if cv2 is None:
        raise RuntimeError("CLAHE requested but OpenCV (cv2) is not available in this environment.")
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile_grid_size), int(tile_grid_size)))
    out = clahe.apply(image_u8)
    return _from_uint8(out)


def _parse_preproc_spec(spec: str):
    spec = spec.strip().lower()
    if not spec:
        raise ValueError("Empty perturbation spec.")

    if spec.startswith("gamma"):
        m = re.fullmatch(r"gamma(\d{3})", spec)
        if not m:
            raise ValueError(f"Unsupported gamma spec: {spec}")
        gamma = int(m.group(1)) / 100.0
        return spec, (lambda img, g=gamma: _apply_gamma(img, g)), {"kind": "gamma", "gamma": gamma}

    if spec.startswith("clahe"):
        m = re.fullmatch(r"clahe(\d{2})", spec)
        if not m:
            raise ValueError(f"Unsupported CLAHE spec: {spec}")
        clip = int(m.group(1)) / 10.0
        return spec, (lambda img, c=clip: _apply_clahe(img, c, 8)), {"kind": "clahe", "clip_limit": clip, "tile_grid_size": 8}

    if spec.startswith("jpeg"):
        m = re.fullmatch(r"jpeg(\d{2})", spec)
        if not m:
            raise ValueError(f"Unsupported JPEG spec: {spec}")
        q = int(m.group(1))
        return spec, (lambda img, q=q: _apply_jpeg(img, q)), {"kind": "jpeg", "quality": q}

    if spec.startswith("downup"):
        m = re.fullmatch(r"downup(\d{3})", spec)
        if not m:
            raise ValueError(f"Unsupported downup spec: {spec}")
        size = int(m.group(1))
        return spec, (lambda img, s=size: _apply_downup(img, s)), {"kind": "downup", "downup_size": size}

    if spec.startswith("quant"):
        m = re.fullmatch(r"quant(\d)", spec)
        if not m:
            raise ValueError(f"Unsupported quantization spec: {spec}")
        bits = int(m.group(1))
        return spec, (lambda img, b=bits: _apply_quantize(img, b)), {"kind": "quantize", "bits": bits}

    raise ValueError(f"Unsupported perturbation spec: {spec}")


def _prepare_cache_subset(
    cache_root: Path,
    original_cache_root: Path,
    dataset_key: str,
    subset_name: str,
    rows: pd.DataFrame,
    perturb_fn,
    perturb_label: str,
) -> None:
    cache_dir = cache_root / dataset_key / subset_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    for row in rows.itertuples(index=False):
        src = original_cache_root / dataset_key / "full" / row.cache_file
        if not src.exists():
            continue
        data = np.load(src)
        image = data["image"].astype(np.float32)
        mask = data["mask"].astype(np.uint8)
        perturbed = perturb_fn(image)
        np.savez_compressed(cache_dir / row.cache_file, image=perturbed, mask=mask)

    rows.to_parquet(cache_dir / "metadata.parquet", index=False)


def _patient_group_col(metadata: pd.DataFrame) -> str:
    if "patient_id" in metadata.columns and metadata["patient_id"].notna().any():
        return "patient_id"
    return "sample_id"


def _split_patient_disjoint(metadata: pd.DataFrame, id_fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not (0.05 <= float(id_fraction) <= 0.95):
        raise ValueError("--id-fraction must be in [0.05, 0.95]")
    group_col = _patient_group_col(metadata)
    groups = [str(x) for x in metadata[group_col].dropna().astype(str).unique().tolist()]
    if not groups:
        raise RuntimeError(f"No grouping values found in column '{group_col}' for patient-disjoint split.")
    rng = np.random.default_rng(seed)
    groups = list(np.array(groups)[rng.permutation(len(groups))])
    n_id = max(1, min(len(groups) - 1, int(round(len(groups) * float(id_fraction)))))
    id_groups = set(groups[:n_id])
    mask_id = metadata[group_col].astype(str).isin(id_groups)
    id_df = metadata[mask_id].copy()
    ood_df = metadata[~mask_id].copy()
    if id_df.empty or ood_df.empty:
        raise RuntimeError("Patient-disjoint split produced an empty partition.")
    return id_df, ood_df


def _write_cache_rows(
    *,
    source_cache_dir: Path,
    target_cache_dir: Path,
    rows: pd.DataFrame,
    dataset_key_out: str,
    subset_out: str,
    transform_fn=None,
) -> int:
    target_cache_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for row in rows.itertuples(index=False):
        src = source_cache_dir / row.cache_file
        if not src.exists():
            continue
        data = np.load(src)
        image = data["image"].astype(np.float32)
        mask = data["mask"].astype(np.uint8)
        if transform_fn is not None:
            image = np.clip(transform_fn(image), 0.0, 1.0).astype(np.float32)
        np.savez_compressed(target_cache_dir / row.cache_file, image=image, mask=mask)
        written += 1

    meta_out = rows.copy()
    if "dataset_key" in meta_out.columns:
        meta_out["dataset_key"] = dataset_key_out
    if "subset" in meta_out.columns:
        meta_out["subset"] = subset_out
    meta_out.to_parquet(target_cache_dir / "metadata.parquet", index=False)
    return written


def _run_alias_builder(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    alias_cache_root = Path(args.alias_cache_root) if args.alias_cache_root else (output_dir / "cache_aliases")
    alias_cache_root.mkdir(parents=True, exist_ok=True)

    paths_cfg = load_paths_config(Path("configs/paths.yaml"), validate=False)
    source_cache_dir = paths_cfg.cache_root / args.base_dataset / args.source_subset
    metadata_path = source_cache_dir / "metadata.parquet"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing source metadata: {metadata_path}")

    metadata = pd.read_parquet(metadata_path)
    if metadata.empty:
        raise RuntimeError(f"Source metadata is empty: {metadata_path}")

    id_rows, ood_rows = _split_patient_disjoint(metadata, id_fraction=args.id_fraction, seed=args.seed)
    group_col = _patient_group_col(metadata)

    # Clean ID partition is written under the canonical base dataset key so existing model-key routing still works.
    id_target_dir = alias_cache_root / args.base_dataset / args.pilot_subset
    n_id_written = _write_cache_rows(
        source_cache_dir=source_cache_dir,
        target_cache_dir=id_target_dir,
        rows=id_rows,
        dataset_key_out=args.base_dataset,
        subset_out=args.pilot_subset,
        transform_fn=None,
    )

    manifest_rows: list[dict[str, object]] = [
        {
            "dataset_alias": args.base_dataset,
            "role": "id_clean",
            "base_dataset": args.base_dataset,
            "source_subset": args.source_subset,
            "subset": args.pilot_subset,
            "perturb_spec": "clean",
            "n_samples": int(n_id_written),
            "n_patients": int(id_rows[group_col].astype(str).nunique()),
            "params_json": json.dumps({"kind": "clean"}),
        }
    ]

    specs = [s.strip().lower() for s in args.perturb_specs.split(",") if s.strip()]
    for spec in specs:
        suffix, fn, cfg = _parse_preproc_spec(spec)
        alias_key = f"{args.base_dataset}_pp_{suffix}"
        target_dir = alias_cache_root / alias_key / args.pilot_subset
        n_written = _write_cache_rows(
            source_cache_dir=source_cache_dir,
            target_cache_dir=target_dir,
            rows=ood_rows,
            dataset_key_out=alias_key,
            subset_out=args.pilot_subset,
            transform_fn=fn,
        )
        manifest_rows.append(
            {
                "dataset_alias": alias_key,
                "role": "ood_perturbed",
                "base_dataset": args.base_dataset,
                "source_subset": args.source_subset,
                "subset": args.pilot_subset,
                "perturb_spec": suffix,
                "n_samples": int(n_written),
                "n_patients": int(ood_rows[group_col].astype(str).nunique()),
                "params_json": json.dumps(cfg, sort_keys=True),
            }
        )

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_csv = output_dir / f"{args.manifest_prefix}.csv"
    manifest_json = output_dir / f"{args.manifest_prefix}.json"
    manifest_md = output_dir / f"{args.manifest_prefix}.md"
    manifest_df.to_csv(manifest_csv, index=False)
    manifest_json.write_text(
        json.dumps(
            {
                "mode": "alias_builder",
                "base_dataset": args.base_dataset,
                "source_subset": args.source_subset,
                "pilot_subset": args.pilot_subset,
                "id_fraction": float(args.id_fraction),
                "seed": int(args.seed),
                "group_col": group_col,
                "id_split": {
                    "n_samples": int(len(id_rows)),
                    "n_groups": int(id_rows[group_col].astype(str).nunique()),
                },
                "ood_split": {
                    "n_samples": int(len(ood_rows)),
                    "n_groups": int(ood_rows[group_col].astype(str).nunique()),
                },
                "alias_cache_root": str(alias_cache_root),
                "aliases": manifest_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    manifest_md.write_text(
        "\n".join(
            [
                "# Preprocessing Shift Alias Cache Manifest",
                "",
                f"- Base dataset: `{args.base_dataset}`",
                f"- Source subset: `{args.source_subset}`",
                f"- Pilot subset: `{args.pilot_subset}`",
                f"- Alias cache root: `{alias_cache_root}`",
                f"- Patient grouping column: `{group_col}`",
                f"- ID fraction: `{float(args.id_fraction):.2f}`",
                f"- Seed: `{int(args.seed)}`",
                f"- ID clean samples: `{len(id_rows)}` ({id_rows[group_col].astype(str).nunique()} patients)",
                f"- OOD perturbed samples (per alias): `{len(ood_rows)}` ({ood_rows[group_col].astype(str).nunique()} patients)",
                "",
                "## Aliases",
                "",
                "| Dataset Alias | Role | Perturbation | Samples | Patients |",
                "|---|---|---|---:|---:|",
                *[
                    f"| {r['dataset_alias']} | {r['role']} | {r['perturb_spec']} | {r['n_samples']} | {r['n_patients']} |"
                    for r in manifest_rows
                ],
                "",
                f"- CSV: `{manifest_csv}`",
                f"- JSON: `{manifest_json}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"[INFO] Alias cache root: {alias_cache_root}")
    print(f"[INFO] Alias manifest CSV: {manifest_csv}")
    print(f"[INFO] Alias manifest JSON: {manifest_json}")
    print(f"[INFO] Alias manifest MD: {manifest_md}")


def _run_legacy_robustness(args: argparse.Namespace) -> None:
    datasets = [d.strip().lower() for d in args.datasets.split(",") if d.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_root = output_dir / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    paths_cfg = load_paths_config(Path("configs/paths.yaml"), validate=False)
    exp_keys = [args.experiment]

    perturbations = {
        "intensity_shift": lambda img: _apply_intensity_shift(img, delta=0.1),
        "gaussian_blur": lambda img: _apply_gaussian_blur(img, sigma=1.0),
        "salt_pepper": lambda img: _apply_salt_pepper(img, amount=0.01),
    }

    sampled_rows = {}
    for dataset_key in datasets:
        metadata_path = paths_cfg.cache_root / dataset_key / "full" / "metadata.parquet"
        if not metadata_path.exists():
            print(f"[WARN] Missing metadata for {dataset_key}: {metadata_path}")
            continue
        metadata = pd.read_parquet(metadata_path)
        if len(metadata) == 0:
            continue
        sample = metadata.sample(n=min(args.subset_size, len(metadata)), random_state=args.seed)
        sampled_rows[dataset_key] = sample

    summary_rows = []
    for perturb_label, perturb_fn in perturbations.items():
        subset_name = f"robustness_{perturb_label}"
        for dataset_key, sample in sampled_rows.items():
            _prepare_cache_subset(
                cache_root=cache_root,
                original_cache_root=paths_cfg.cache_root,
                dataset_key=dataset_key,
                subset_name=subset_name,
                rows=sample,
                perturb_fn=perturb_fn,
                perturb_label=perturb_label,
            )

        robust_paths = PathsConfig(
            datasets_root=paths_cfg.datasets_root,
            models_root=paths_cfg.models_root,
            fingerprints_root=output_dir / "fingerprints" / perturb_label,
            cache_root=cache_root,
            datasets=paths_cfg.datasets,
            models=paths_cfg.models,
        )
        robust_paths.fingerprints_root.mkdir(parents=True, exist_ok=True)

        for exp_key in exp_keys:
            exp_cfg = load_experiment_config(Path("configs/experiments.yaml"), exp_key)
            exp_cfg.subset = subset_name
            run_fingerprint_experiment(
                exp_cfg=exp_cfg,
                paths_cfg=robust_paths,
                device=args.device,
                endpoint_mode=args.endpoint_mode,
                seed=args.seed,
                ig_steps=args.ig_steps,
                deterministic=args.deterministic,
            )

            if args.endpoint_mode == "upper_bound_gt":
                perturbed_root = robust_paths.fingerprints_root / exp_cfg.key
                base_root = paths_cfg.fingerprints_root / exp_cfg.key
            else:
                perturbed_root = robust_paths.fingerprints_root / args.endpoint_mode / exp_cfg.key
                base_root = paths_cfg.fingerprints_root / args.endpoint_mode / exp_cfg.key

            ref_key = exp_cfg.train_dataset
            for tgt_key in exp_cfg.test_datasets:
                ref_base = pd.read_parquet(base_root / f"{ref_key}.parquet")
                tgt_base = pd.read_parquet(base_root / f"{tgt_key}.parquet")
                ref_pert = pd.read_parquet(perturbed_root / f"{ref_key}.parquet")
                tgt_pert = pd.read_parquet(perturbed_root / f"{tgt_key}.parquet")

                base_stats = _compute_shift_stats(ref_base, tgt_base)
                pert_stats = _compute_shift_stats(ref_pert, tgt_pert)

                for key, base_val in base_stats.items():
                    pert_val = pert_stats.get(key, np.nan)
                    summary_rows.append(
                        {
                            "perturbation": perturb_label,
                            "comparison": f"{ref_key}_to_{tgt_key}",
                            "metric": key,
                            "baseline": base_val,
                            "perturbed": pert_val,
                            "percent_change": _percent_change(pert_val, base_val),
                        }
                    )

                base_auc = _shift_auc(ref_base, tgt_base)
                pert_auc = _shift_auc(ref_pert, tgt_pert)
                summary_rows.append(
                    {
                        "perturbation": perturb_label,
                        "comparison": f"{ref_key}_to_{tgt_key}",
                        "metric": "auroc_shift_detection",
                        "baseline": base_auc,
                        "perturbed": pert_auc,
                        "percent_change": _percent_change(pert_auc, base_auc),
                    }
                )

                for dataset_key, df_base, df_pert in [(ref_key, ref_base, ref_pert), (tgt_key, tgt_base, tgt_pert)]:
                    if "attribution_abs_sum" not in df_base.columns or "dice" not in df_base.columns:
                        continue
                    base_mask = np.isfinite(df_base["attribution_abs_sum"]) & np.isfinite(df_base["dice"])
                    pert_mask = np.isfinite(df_pert["attribution_abs_sum"]) & np.isfinite(df_pert["dice"])
                    if base_mask.sum() < 3 or pert_mask.sum() < 3:
                        continue
                    base_pearson, _ = stats.pearsonr(df_base["attribution_abs_sum"][base_mask], df_base["dice"][base_mask])
                    base_spearman, _ = stats.spearmanr(df_base["attribution_abs_sum"][base_mask], df_base["dice"][base_mask])
                    pert_pearson, _ = stats.pearsonr(df_pert["attribution_abs_sum"][pert_mask], df_pert["dice"][pert_mask])
                    pert_spearman, _ = stats.spearmanr(df_pert["attribution_abs_sum"][pert_mask], df_pert["dice"][pert_mask])
                    summary_rows.append(
                        {
                            "perturbation": perturb_label,
                            "comparison": dataset_key,
                            "metric": "pearson_r",
                            "baseline": base_pearson,
                            "perturbed": pert_pearson,
                            "percent_change": _percent_change(pert_pearson, base_pearson),
                        }
                    )
                    summary_rows.append(
                        {
                            "perturbation": perturb_label,
                            "comparison": dataset_key,
                            "metric": "spearman_r",
                            "baseline": base_spearman,
                            "perturbed": pert_spearman,
                            "percent_change": _percent_change(pert_spearman, base_spearman),
                        }
                    )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "robustness_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    summary_md = output_dir / "robustness_summary.md"
    summary_md.write_text(
        "# Robustness Summary\n\n"
        f"- Experiment: {args.experiment}\n"
        f"- Endpoint: {args.endpoint_mode}\n"
        f"- IG steps: {args.ig_steps}\n"
        f"- Samples per dataset: {args.subset_size}\n"
        f"- Perturbations: {', '.join(perturbations.keys())}\n"
        "\nSee `robustness_summary.csv` for full details.\n",
        encoding="utf-8",
    )

    print(f"[INFO] Saved robustness summary to: {summary_path}")
    print(f"[INFO] Saved robustness notes to: {summary_md}")


def main() -> None:
    args = parse_args()
    if args.mode == "alias_builder":
        _run_alias_builder(args)
        return
    _run_legacy_robustness(args)


if __name__ == "__main__":
    main()
