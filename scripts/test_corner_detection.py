#!/usr/bin/env python3
"""
test_corner_detection.py
------------------------
Test script for the new multi-strategy corner detection.

Usage:
    python scripts/test_corner_detection.py --input path/to/image.jpg --debug debug_output
"""

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.corner_detection import detect_and_crop


def main():
    parser = argparse.ArgumentParser(description="Test multi-strategy corner detection")
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument("--debug", help="Debug output directory for intermediate images")
    parser.add_argument("--output", default="test_output.png", help="Output image path")
    args = parser.parse_args()

    # Load image
    image = cv2.imread(args.input)
    if image is None:
        print(f"❌ Could not load image: {args.input}")
        return 1

    print(f"📷 Loaded image: {image.shape}")

    # Set up debug directory
    debug_dir = Path(args.debug) if args.debug else None
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)
        print(f"🐛 Debug images will be saved to: {debug_dir}")

    # Run corner detection
    print("🔍 Running multi-strategy corner detection...")
    cropped, success = detect_and_crop(image, debug_dir)

    if success:
        print("✅ Corner detection successful!")
        print(f"📐 Cropped image size: {cropped.shape}")
    else:
        print("⚠️  Corner detection failed, using full image")

    # Save output
    cv2.imwrite(args.output, cropped)
    print(f"💾 Saved result to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
