"""
scripts/test_inference.py
--------------------------
Smoke-test the full inference pipeline on a single image.
Runs even without trained weights (model outputs near-identity flow).

Usage:
  python scripts/test_inference.py --image path/to/photo.jpg
  python scripts/test_inference.py --image path/to/photo.jpg --weights models/best_model.pth
  python scripts/test_inference.py --arch-only   # just print model summary, no image needed
"""

import argparse
import sys
import time
import logging
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser(description="Test LightDewarpNet inference")
    p.add_argument("--image",     default=None, help="Input image path")
    p.add_argument("--weights",   default=None, help="Path to .pth weights (optional)")
    p.add_argument("--output",    default="test_output.png")
    p.add_argument("--arch-only", action="store_true", help="Print arch summary and exit")
    p.add_argument("--cpu",       action="store_true")
    args = p.parse_args()

    device = torch.device("cpu" if args.cpu else
                          ("cuda" if torch.cuda.is_available() else "cpu"))

    # ── Model summary ─────────────────────────────────────────────────
    from core.model_arch import LightDewarpNet
    model = LightDewarpNet()

    print("\n" + "═" * 55)
    print("  LightDewarpNet — Architecture Summary")
    print("═" * 55)
    print(f"  Parameters  : {model.count_parameters() / 1e6:.2f} M")
    print(f"  VRAM (B=1)  : ~{model.estimate_vram_mb(1):.0f} MB")
    print(f"  VRAM (B=4)  : ~{model.estimate_vram_mb(4):.0f} MB")
    print(f"  Input size  : {LightDewarpNet.INPUT_SIZE}")
    print(f"  Device      : {device}")
    print("═" * 55)

    if args.arch_only:
        return

    if args.image is None:
        print("\nNo --image provided. Generating synthetic test image…")
        # Create a synthetic curved document for testing
        img = np.ones((600, 480, 3), dtype=np.uint8) * 240
        for i in range(10, 580, 30):
            cv2.line(img, (20, i), (460, i), (80, 80, 80), 1)
        for i in range(20, 460, 40):
            cv2.line(img, (i, 10), (i, 590), (80, 80, 80), 1)
        # Apply a synthetic warp to simulate a curved page
        h, w = img.shape[:2]
        map_x = np.zeros((h, w), np.float32)
        map_y = np.zeros((h, w), np.float32)
        for y in range(h):
            for x in range(w):
                nx = x / w - 0.5
                ny = y / h - 0.5
                r = np.sqrt(nx**2 + ny**2)
                factor = 1 + 0.3 * r
                map_x[y, x] = x + nx * factor * 40
                map_y[y, x] = y + ny * factor * 20
        image = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
        cv2.putText(image, "SYNTHETIC TEST", (80, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (40, 40, 200), 2)
    else:
        image = cv2.imread(args.image)
        if image is None:
            logger.error(f"Cannot read: {args.image}")
            sys.exit(1)
        logger.info(f"Loaded image: {args.image} — {image.shape[1]}×{image.shape[0]}")

    # ── Load weights ──────────────────────────────────────────────────
    if args.weights:
        from core.dewarp import load_model
        model = load_model(args.weights, device)
        if model is None:
            logger.warning("Falling back to untrained model.")
            model = LightDewarpNet().to(device)
    else:
        logger.info("No weights provided — using randomly initialised model.")
        logger.info("Output will look scrambled; this just tests the pipeline.")
        model = model.to(device)

    model.eval()

    # ── Run inference ─────────────────────────────────────────────────
    from core.dewarp import dewarp_document
    from core.corner_detection import detect_and_crop
    from core.shadow_removal import remove_shadows_and_binarize

    print("\nRunning full pipeline…")

    t0 = time.perf_counter()
    cropped, found = detect_and_crop(image)
    t1 = time.perf_counter()
    print(f"  Corner detection : {(t1-t0)*1000:.1f} ms | found={found}")

    t0 = time.perf_counter()
    dewarped = dewarp_document(cropped, model=model, device=device)
    t1 = time.perf_counter()
    print(f"  Neural dewarp    : {(t1-t0)*1000:.1f} ms")

    t0 = time.perf_counter()
    final = remove_shadows_and_binarize(dewarped, output_mode="binary")
    t1 = time.perf_counter()
    print(f"  Shadow removal   : {(t1-t0)*1000:.1f} ms")

    # ── Save side-by-side comparison ──────────────────────────────────
    input_resized  = cv2.resize(image,  (448, 448))
    output_resized = cv2.resize(final,  (448, 448))
    if len(output_resized.shape) == 2:
        output_resized = cv2.cvtColor(output_resized, cv2.COLOR_GRAY2BGR)

    separator = np.ones((448, 4, 3), np.uint8) * 180
    comparison = np.hstack([input_resized, separator, output_resized])

    # Add labels
    cv2.putText(comparison, "INPUT",  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,0), 2)
    cv2.putText(comparison, "OUTPUT", (462, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,0), 2)

    out_path = Path(args.output)
    cv2.imwrite(str(out_path), comparison)
    print(f"\nSaved comparison to: {out_path.resolve()}")
    print("Pipeline test complete ✓\n")


if __name__ == "__main__":
    main()
