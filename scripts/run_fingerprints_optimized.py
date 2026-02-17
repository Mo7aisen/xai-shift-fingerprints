"""Optimized fingerprint runner with GPU memory management."""

import os
import sys
from pathlib import Path

# Set GPU memory management environment variables BEFORE importing torch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Allow async operations for speed

# Add project to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import argparse
import torch

from xfp.config import load_experiment_config, load_paths_config
from xfp.fingerprint.runner import run_fingerprint_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run attribution fingerprinting with GPU optimization"
    )
    parser.add_argument(
        "--experiment",
        required=True,
        help="Experiment key from configs/fingerprints.yaml",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device to use (default: auto-detect)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run even if fingerprint already exists",
    )
    return parser.parse_args()


def main():
    """Run fingerprint with optimized GPU settings."""
    import _path_setup  # noqa: F401 - ensures xfp is importable

    args = parse_args()

    # Load configurations
    config_root = ROOT / "configs"
    paths_cfg = load_paths_config(config_root / "paths.yaml")
    exp_cfg = load_experiment_config(
        config_root / "experiments.yaml",
        args.experiment
    )

    print(f"\n{'='*70}")
    print(f"Running Fingerprint Experiment: {args.experiment}")
    print(f"{'='*70}")
    print(f"Device: {args.device}")
    print(f"Train Dataset: {exp_cfg.train_dataset}")
    print(f"Test Datasets: {', '.join(exp_cfg.test_datasets)}")
    print(f"Subset: {exp_cfg.subset}")

    # Check if already exists
    fingerprint_dir = paths_cfg.fingerprints_root / exp_cfg.key
    summary_file = fingerprint_dir / "summary.json"

    if summary_file.exists() and not args.force:
        print(f"\n⚠️  Fingerprint already exists: {summary_file}")
        print("Use --force to re-run")
        return

    # Set memory fraction if using CUDA
    if args.device == "cuda":
        # Don't pre-allocate all GPU memory
        torch.cuda.set_per_process_memory_fraction(0.9)
        print(f"\nGPU Memory Management:")
        print(f"  - Expandable segments: Enabled")
        print(f"  - Memory fraction: 90%")
        print(f"  - Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Run fingerprint experiment
    print(f"\nStarting fingerprint generation...")

    try:
        result = run_fingerprint_experiment(
            exp_cfg=exp_cfg,
            paths_cfg=paths_cfg,
            device=args.device,
        )

        print(f"\n{'='*70}")
        print(f"✅ Fingerprint Complete!")
        print(f"{'='*70}")
        print(f"Summary saved: {result.fingerprint_path}")
        print(f"\nDataset Results:")
        for dataset_key, summary in result.summaries.items():
            count = summary.get('count', 0)
            print(f"  - {dataset_key}: {int(count)} samples processed")

    except Exception as e:
        print(f"\n❌ Fingerprint failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        # Final cleanup
        if args.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


if __name__ == "__main__":
    main()
