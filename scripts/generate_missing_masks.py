"""Generate missing Shenzhen lung masks using trained UNet models."""

from __future__ import annotations

import _path_setup  # noqa: F401 - ensures xfp is importable

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from tqdm import tqdm

# Add project to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xfp.config import load_paths_config
from xfp.models import load_unet_checkpoint

TARGET_SIZE = (512, 512)

try:  # Pillow >= 9.1
    Resampling = Image.Resampling
except AttributeError:
    Resampling = Image  # type: ignore[attr-defined]


def generate_mask(
    image_path: Path,
    model: torch.nn.Module,
    device: str,
) -> np.ndarray:
    """Generate lung mask for a single image."""

    # Load and preprocess image
    image = Image.open(image_path).convert("L")
    original_size = image.size

    # Resize to model input size
    image_resized = ImageOps.fit(image, TARGET_SIZE, method=Resampling.BILINEAR)
    image_array = np.asarray(image_resized, dtype=np.float32) / 255.0

    # Convert to tensor
    tensor = torch.from_numpy(image_array).unsqueeze(0).unsqueeze(0).to(device)

    # Generate prediction
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)
        mask_pred = (probs.cpu().numpy()[0, 0] > 0.5).astype(np.uint8)

    # Clean up
    del tensor, logits, probs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Convert back to original size
    mask_image = Image.fromarray((mask_pred * 255).astype(np.uint8), mode="L")
    mask_resized = mask_image.resize(original_size, resample=Resampling.NEAREST)

    return np.asarray(mask_resized)


def main():
    """Generate masks for all missing Shenzhen images."""

    # Load configuration
    config_path = Path(__file__).resolve().parents[1] / "configs" / "paths.yaml"
    paths_cfg = load_paths_config(config_path)

    # Paths
    shenzhen_images_dir = paths_cfg.datasets_root / "ShenzenDataset" / "CXR_png"
    shenzhen_masks_dir = paths_cfg.datasets_root / "ShenzenDataset" / "mask"
    missing_masks_file = Path(__file__).resolve().parents[1] / "data" / "interim" / "shenzhen" / "full" / "missing_masks.txt"

    if not missing_masks_file.exists():
        print(f"No missing masks file found at {missing_masks_file}")
        return

    # Read missing mask list (excluding Thumbs.db)
    missing_files = [
        line.strip()
        for line in missing_masks_file.read_text().splitlines()
        if line.strip() and not line.strip().lower().endswith(".db")
    ]

    print(f"Found {len(missing_files)} missing masks")

    if not missing_files:
        print("No masks to generate!")
        return

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Try to load Montgomery model first (better for TB cases), fall back to JSRT
    model_choices = [
        ("Montgomery", paths_cfg.models.get("unet_montgomery_full")),
        ("JSRT", paths_cfg.models.get("unet_jsrt_full")),
    ]

    model = None
    model_name = None
    for name, checkpoint_path in model_choices:
        if checkpoint_path and checkpoint_path.exists():
            try:
                print(f"Loading {name} model from {checkpoint_path}")
                model = load_unet_checkpoint(checkpoint_path, device=device)
                model_name = name
                break
            except Exception as e:
                print(f"Failed to load {name} model: {e}")
                continue

    if model is None:
        raise RuntimeError("No valid model checkpoint found!")

    print(f"Successfully loaded {model_name} model")

    # Generate masks
    shenzhen_masks_dir.mkdir(parents=True, exist_ok=True)
    generated_count = 0
    failed_count = 0

    for image_filename in tqdm(missing_files, desc="Generating masks"):
        image_path = shenzhen_images_dir / image_filename

        if not image_path.exists():
            print(f"Warning: Image not found: {image_path}")
            failed_count += 1
            continue

        # Determine mask filename
        stem = image_path.stem
        mask_filename = f"{stem}_mask.png"
        mask_path = shenzhen_masks_dir / mask_filename

        # Skip if mask already exists
        if mask_path.exists():
            print(f"Mask already exists: {mask_filename}")
            continue

        try:
            # Generate mask
            mask_array = generate_mask(image_path, model, device)

            # Save mask
            mask_image = Image.fromarray(mask_array, mode="L")
            mask_image.save(mask_path)
            generated_count += 1

        except Exception as e:
            print(f"Failed to generate mask for {image_filename}: {e}")
            failed_count += 1
            continue

    print(f"\n{'='*60}")
    print(f"Mask Generation Complete!")
    print(f"{'='*60}")
    print(f"Generated: {generated_count} masks")
    print(f"Failed: {failed_count} masks")
    print(f"Model used: {model_name}")
    print(f"Masks saved to: {shenzhen_masks_dir}")

    # Update missing masks file
    if generated_count > 0:
        remaining_missing = []
        for filename in missing_files:
            image_path = shenzhen_images_dir / filename
            stem = image_path.stem
            mask_path = shenzhen_masks_dir / f"{stem}_mask.png"
            if not mask_path.exists():
                remaining_missing.append(filename)

        if remaining_missing:
            missing_masks_file.write_text("\n".join(remaining_missing), encoding="utf-8")
            print(f"\nStill missing {len(remaining_missing)} masks")
        else:
            missing_masks_file.write_text("", encoding="utf-8")
            print(f"\nAll masks generated! Empty missing_masks.txt")


if __name__ == "__main__":
    main()
