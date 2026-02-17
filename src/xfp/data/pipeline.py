"""Dataset caching pipeline for attribution fingerprinting."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from PIL import Image

from xfp.config import PathsConfig

TARGET_SIZE = (512, 512)


try:  # Pillow >= 9.1
    Resampling = Image.Resampling
except AttributeError:  # pragma: no cover - legacy Pillow
    Resampling = Image  # type: ignore[attr-defined]


def infer_patient_id(dataset_key: str, sample_id: str) -> str:
    """Infer patient identifier from sample/image identifier.

    This enables patient-level grouped splitting even when no explicit patient
    metadata is available.
    """
    value = str(sample_id)
    dataset_key = str(dataset_key).lower()
    if dataset_key in {"montgomery", "shenzhen", "nih_chestxray14"}:
        match = re.match(r"^(.*)_\d+$", value)
        if match:
            return match.group(1)
    return value


def build_dataset_cache(*, dataset_key: str, subset: str, paths_cfg: PathsConfig) -> None:
    """Build cached NPZ tensors and metadata parquet for a dataset."""
    dataset_cfg = paths_cfg.datasets.get(dataset_key)
    if dataset_cfg is None:
        raise KeyError(f"Dataset '{dataset_key}' not configured in configs/paths.yaml.")

    images_dir = dataset_cfg.images.resolve()
    masks_dir = dataset_cfg.masks.resolve()
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"Masks directory not found: {masks_dir}")

    cache_dir = paths_cfg.cache_root / dataset_key / subset
    cache_dir.mkdir(parents=True, exist_ok=True)

    sample_ids = _load_subset_ids(dataset_key, subset)
    image_files = _list_images(images_dir, sample_ids=sample_ids)

    records: List[dict] = []
    skipped = 0

    for image_path in image_files:
        sample_id = image_path.stem
        mask = _load_mask(dataset_key, masks_dir, sample_id)
        if mask is None:
            skipped += 1
            continue

        image = Image.open(image_path).convert("L")
        original_width, original_height = image.size
        image_arr = np.asarray(image, dtype=np.float32) / 255.0
        mask_arr = _ensure_binary_mask(mask)

        processed_image, processed_mask, pad, scale = _resize_and_pad(
            image_arr, mask_arr, target_size=TARGET_SIZE
        )

        cache_file = f"{sample_id}.npz"
        np.savez_compressed(
            cache_dir / cache_file,
            image=processed_image.astype(np.float32),
            mask=processed_mask.astype(np.uint8),
        )

        records.append(
            {
                "dataset_key": dataset_key,
                "subset": subset,
                "sample_id": sample_id,
                "patient_id": infer_patient_id(dataset_key, sample_id),
                "cache_file": cache_file,
                "source_image": str(image_path),
                "source_mask": mask["source"] if isinstance(mask, dict) else str(mask),
                "original_height": original_height,
                "original_width": original_width,
                "processed_height": TARGET_SIZE[1],
                "processed_width": TARGET_SIZE[0],
                "pad_left": pad["left"],
                "pad_top": pad["top"],
                "pad_right": pad["right"],
                "pad_bottom": pad["bottom"],
                "scale": scale,
                "mask_coverage": float(processed_mask.mean()),
            }
        )

    metadata_path = cache_dir / "metadata.parquet"
    pd.DataFrame.from_records(records).to_parquet(metadata_path, index=False)
    print(
        json.dumps(
            {
                "dataset": dataset_key,
                "subset": subset,
                "cache_dir": str(cache_dir),
                "samples": len(records),
                "skipped_missing_mask": skipped,
            },
            indent=2,
        )
    )


def _load_subset_ids(dataset_key: str, subset: str) -> Optional[set[str]]:
    if subset == "full":
        return None
    subset_file = Path(__file__).resolve().parents[3] / "configs" / "subsets" / f"{dataset_key}_{subset}.txt"
    if not subset_file.exists():
        raise FileNotFoundError(f"Subset file not found: {subset_file}")
    ids = set()
    for line in subset_file.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value:
            continue
        ids.add(Path(value).stem)
    return ids


def _list_images(images_dir: Path, sample_ids: Optional[set[str]]) -> List[Path]:
    patterns = ("*.png", "*.jpg", "*.jpeg")
    files: List[Path] = []
    for pattern in patterns:
        files.extend(sorted(images_dir.rglob(pattern)))
    if not files:
        raise FileNotFoundError(f"No images found in {images_dir}")
    if sample_ids is None:
        return files
    return [path for path in files if path.stem in sample_ids]


def _load_mask(dataset_key: str, masks_dir: Path, sample_id: str) -> Optional[object]:
    if dataset_key == "montgomery":
        left = masks_dir / "leftMask" / f"{sample_id}.png"
        right = masks_dir / "rightMask" / f"{sample_id}.png"
        if not left.exists() or not right.exists():
            return None
        left_arr = np.asarray(Image.open(left).convert("L"))
        right_arr = np.asarray(Image.open(right).convert("L"))
        mask = np.maximum(left_arr, right_arr)
        return {"array": mask, "source": f"{left}|{right}"}

    if dataset_key == "shenzhen":
        mask_path = masks_dir / f"{sample_id}_mask.png"
        if not mask_path.exists():
            return None
        return mask_path

    if dataset_key == "nih_chestxray14":
        mask_path = masks_dir / f"{sample_id}_mask.png"
        if not mask_path.exists():
            return None
        return mask_path

    # JSRT and default: direct filename match
    mask_path = masks_dir / f"{sample_id}.png"
    if not mask_path.exists():
        return None
    return mask_path


def _ensure_binary_mask(mask: object) -> np.ndarray:
    if isinstance(mask, dict):
        mask_arr = np.asarray(mask["array"])
    else:
        mask_arr = np.asarray(Image.open(mask).convert("L"))
    return (mask_arr > 0).astype(np.uint8)


def _resize_and_pad(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    target_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, dict, float]:
    target_w, target_h = target_size
    height, width = image.shape
    scale = min(target_w / width, target_h / height)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))

    image_resized = Image.fromarray((image * 255).astype(np.uint8)).resize(
        (new_w, new_h), resample=Resampling.BILINEAR
    )
    mask_resized = Image.fromarray((mask * 255).astype(np.uint8)).resize(
        (new_w, new_h), resample=Resampling.NEAREST
    )

    pad_left = (target_w - new_w) // 2
    pad_top = (target_h - new_h) // 2
    pad_right = target_w - new_w - pad_left
    pad_bottom = target_h - new_h - pad_top

    canvas_img = np.zeros((target_h, target_w), dtype=np.float32)
    canvas_mask = np.zeros((target_h, target_w), dtype=np.uint8)

    image_arr = np.asarray(image_resized, dtype=np.float32) / 255.0
    mask_arr = (np.asarray(mask_resized) > 0).astype(np.uint8)

    canvas_img[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = image_arr
    canvas_mask[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = mask_arr

    return canvas_img, canvas_mask, {
        "left": pad_left,
        "top": pad_top,
        "right": pad_right,
        "bottom": pad_bottom,
    }, float(scale)
