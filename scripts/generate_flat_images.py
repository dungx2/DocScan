#!/usr/bin/env python3
"""
generate_flat_images.py
-----------------------
Generate flat/rectified ground truth images from Doc3D img/ and bm/ directories.

For each distorted image, loads the corresponding backward map and applies F.grid_sample
to generate the flat reference image. Useful for MS-SSIM loss during training.

Usage:
    python scripts/generate_flat_images.py --data data/doc3d_subset --workers 4

Output:
    Saves PNG images to data/doc3d_subset/flat/ with same filenames as input images.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Tuple, Optional
from multiprocessing import Pool

import cv2
import numpy as np
import h5py
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def process_sample(args: Tuple[Path, Path, Path]) -> Tuple[str, bool, Optional[str]]:
    """
    Process a single sample: load distorted image + backward map,
    apply grid_sample to generate flat image.

    Args:
        args: Tuple of (img_path, bm_path, output_path)

    Returns:
        Tuple of (filename, success_flag, error_message)
        - success_flag: True if generated, False if failed, None if skipped
        - error_message: Error string if failed, None otherwise
    """
    img_path, bm_path, output_path = args

    try:
        # Skip if already exists (for resuming)
        if output_path.exists():
            return (img_path.name, None, None)

        # ── Load distorted image ──────────────────────────────────────────────
        img = cv2.imread(str(img_path))
        if img is None:
            return (img_path.name, False, "Could not read image file")

        # BGR → RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize to 448×448
        img = cv2.resize(img, (448, 448), interpolation=cv2.INTER_LINEAR)

        # Normalize to [0, 1] float32
        img = img.astype(np.float32) / 255.0

        # ── Load backward map from HDF5 ──────────────────────────────────────
        with h5py.File(str(bm_path), 'r') as f:
            # Try common key names
            if 'bm' in f:
                bm = f['bm'][:]
            else:
                # Fall back to first dataset in file
                key = list(f.keys())[0]
                bm = f[key][:]

        # Shape from h5py should be (2, 448, 448)
        if bm.shape != (2, 448, 448):
            return (img_path.name, False, f"Unexpected bm shape: {bm.shape}")

        bm = bm.astype(np.float32)

        # ── Normalize backward map ────────────────────────────────────────────
        # Backward map contains pixel coordinates [0, 447]
        # Normalize to [0, 1]
        bm = bm / 447.0

        # Remap to [-1, 1] for grid_sample align_corners=True
        bm = bm * 2.0 - 1.0

        # ── Convert to torch tensors ──────────────────────────────────────────
        # bm shape: (2, 448, 448) → transpose to (448, 448, 2) → (1, 448, 448, 2)
        bm_torch = torch.from_numpy(bm.transpose(1, 2, 0)).unsqueeze(0)

        # img shape: (448, 448, 3) → (3, 448, 448) → (1, 3, 448, 448)
        img_torch = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)

        # ── Apply grid_sample to warp image ───────────────────────────────────
        # grid_sample(input, grid) samples input at grid coordinates
        # input: (N, C, H, W) — distorted image
        # grid: (N, H, W, 2) — normalized coordinates to sample from
        flat_torch = F.grid_sample(
            img_torch,
            bm_torch,
            align_corners=True,
            mode='bilinear',
            padding_mode='border'
        )

        # ── Convert back to numpy ─────────────────────────────────────────────
        # flat_torch: (1, 3, 448, 448) → permute to (1, 448, 448, 3) → squeeze to (448, 448, 3)
        flat = flat_torch[0].permute(1, 2, 0).cpu().detach().numpy()

        # Scale to [0, 255] and convert to uint8
        flat = (flat * 255.0).astype(np.uint8)

        # ── Save as PNG ───────────────────────────────────────────────────────
        # Log output directory creation
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert RGB → BGR for cv2.imwrite
        flat_bgr = cv2.cvtColor(flat, cv2.COLOR_RGB2BGR)
        success = cv2.imwrite(str(output_path), flat_bgr)

        if not success:
            return (img_path.name, False, "Failed to write PNG file")

        return (img_path.name, True, None)

    except Exception as e:
        return (img_path.name, False, f"{type(e).__name__}: {str(e)}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate flat/rectified ground truth images from Doc3D dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single process
  python scripts/generate_flat_images.py --data data/doc3d_subset
  
  # 4 parallel workers
  python scripts/generate_flat_images.py --data data/doc3d_subset --workers 4
        """
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/doc3d_subset"),
        help="Path to Doc3D dataset root (default: data/doc3d_subset)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of worker processes (default: 0 = single process)"
    )
    args = parser.parse_args()

    # ── Validate input paths ──────────────────────────────────────────────────
    data_root = args.data.resolve()
    img_dir = data_root / "img"
    bm_dir = data_root / "bm"
    flat_dir = data_root / "flat"

    logger.info(f"Data root: {data_root}")
    logger.info(f"Image dir: {img_dir}")
    logger.info(f"BM dir:    {bm_dir}")
    logger.info(f"Output:    {flat_dir}")

    if not img_dir.exists():
        logger.error(f"❌ img/ directory not found: {img_dir}")
        sys.exit(1)

    if not bm_dir.exists():
        logger.error(f"❌ bm/ directory not found: {bm_dir}")
        sys.exit(1)

    # Create output directory
    flat_dir.mkdir(parents=True, exist_ok=True)

    # ── Collect all samples ───────────────────────────────────────────────────
    img_files = sorted(img_dir.glob("*.png"))
    if not img_files:
        logger.error(f"❌ No PNG images found in {img_dir}")
        sys.exit(1)

    logger.info(f"Found {len(img_files)} images to process\n")

    # Prepare arguments for each sample
    sample_args = []
    for img_path in img_files:
        bm_name = img_path.stem + ".mat"
        bm_path = bm_dir / bm_name
        output_path = flat_dir / img_path.name

        if not bm_path.exists():
            logger.warning(f"⚠️  Backward map not found (skipping): {bm_name}")
            continue

        sample_args.append((img_path, bm_path, output_path))

    logger.info(f"Processing {len(sample_args)} samples with {args.workers or 1} worker(s)...\n")

    # ── Process samples (parallel or single) ──────────────────────────────────
    if args.workers > 0:
        with Pool(args.workers) as pool:
            results = list(tqdm(
                pool.imap_unordered(process_sample, sample_args),
                total=len(sample_args),
                desc="Generating flat images",
                unit="img"
            ))
    else:
        results = [
            process_sample(arg)
            for arg in tqdm(sample_args, desc="Generating flat images", unit="img")
        ]

    # ── Summarize results ─────────────────────────────────────────────────────
    generated = sum(1 for _, success, _ in results if success is True)
    skipped = sum(1 for _, success, _ in results if success is None)
    failed = sum(1 for _, success, _ in results if success is False)

    logger.info(f"\n{'='*70}")
    logger.info(f"SUMMARY:")
    logger.info(f"  ✅ Generated: {generated}")
    logger.info(f"  ⏭️  Skipped:   {skipped}")
    logger.info(f"  ❌ Failed:    {failed}")
    logger.info(f"  📊 Total:     {len(results)}")
    logger.info(f"{'='*70}\n")

    # Log failed samples
    if failed > 0:
        logger.warning("Failed samples:")
        for filename, success, error in results:
            if success is False:
                logger.warning(f"  ❌ {filename}: {error}")
        logger.info()

    logger.info(f"✅ Flat images saved to: {flat_dir}\n")

    # Exit with error code if any failed
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()

