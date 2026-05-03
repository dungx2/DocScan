"""
corner_detection.py
--------------------
Phase 1: Multi-strategy document corner detection and perspective crop.

Tries multiple strategies in order for robust detection on real photos:
1. GrabCut segmentation (best for cluttered backgrounds)
2. HSV color-based segmentation (for white/light paper)
3. Canny edge detection + contour finding (original method)
4. Full image bounds (last resort fallback)
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


def sort_corners(pts):
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Apply a perspective transform to obtain a top-down view of the document.
    """
    corners = sort_corners(pts)
    (tl, tr, br, bl) = corners

    width = int(max(
        np.linalg.norm(br - bl),
        np.linalg.norm(tr - tl)
    ))
    height = int(max(
        np.linalg.norm(tr - br),
        np.linalg.norm(tl - bl)
    ))

    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(image, M, (width, height))
    return warped


def _save_debug_image(image: np.ndarray, name: str, debug_dir: Path) -> None:
    """Save debug image if debug directory is provided."""
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_dir / f"{name}.png"), image)


def _extract_quadrilateral_from_mask(mask: np.ndarray, image_shape: Tuple[int, int], min_area_fraction: float = 0.05) -> Optional[np.ndarray]:
    """
    Extract quadrilateral corners from a binary mask.
    Returns 4 corner points or None if not found.
    Tries multiple epsilon values and convex hull approach if needed.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    h, w = image_shape
    min_area = min_area_fraction * (h * w)
    
    # Find largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) < min_area:
        return None

    # Try approxPolyDP with multiple epsilon values
    perimeter = cv2.arcLength(largest_contour, closed=True)
    for epsilon_factor in [0.01, 0.02, 0.03, 0.04, 0.05]:
        approx = cv2.approxPolyDP(largest_contour, epsilon_factor * perimeter, closed=True)
        if len(approx) == 4:
            logger.info(f"Quadrilateral found with area: {cv2.contourArea(largest_contour):.0f} (epsilon={epsilon_factor})")
            return approx.reshape(4, 2).astype("float32")

    # Fall back to convex hull if approxPolyDP doesn't give 4 points
    hull = cv2.convexHull(largest_contour)
    if len(hull) >= 4:
        # Try approxPolyDP on the hull
        hull_perimeter = cv2.arcLength(hull, True)
        for epsilon_factor in [0.01, 0.02, 0.03, 0.04, 0.05]:
            approx = cv2.approxPolyDP(hull, epsilon_factor * hull_perimeter, True)
            if len(approx) == 4:
                logger.info(f"Quadrilateral found from convex hull with area: {cv2.contourArea(largest_contour):.0f}")
                return approx.reshape(4, 2).astype("float32")

    return None


def strategy_grabcut(image: np.ndarray, debug_dir: Optional[Path] = None) -> Optional[np.ndarray]:
    """
    Strategy 1: GrabCut segmentation for cluttered backgrounds.

    Assumes document occupies center 60% of image, uses GrabCut to separate
    foreground from background, then finds quadrilateral in foreground mask.
    """
    logger.debug("Trying Strategy 1: GrabCut")

    h, w = image.shape[:2]
    margin_x = int(w * 0.05)
    margin_y = int(h * 0.05)
    rect = (margin_x, margin_y, w - 2*margin_x, h - 2*margin_y)

    # Ensure rect is valid
    if rect[2] <= 0 or rect[3] <= 0:
        logger.warning("GrabCut rectangle invalid, skipping")
        return None

    try:
        # Initialize mask for GrabCut
        mask = np.zeros((h, w), dtype=np.uint8)
        bg_model = np.zeros((1, 65), dtype=np.float64)
        fg_model = np.zeros((1, 65), dtype=np.float64)

        # Run GrabCut
        cv2.grabCut(image, mask, rect, bg_model, fg_model, 3, cv2.GC_INIT_WITH_RECT)

        # Create binary mask: include both likely (1) and probably (3) foreground
        fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

        _save_debug_image(fg_mask, "grabcut_mask", debug_dir)

        # Clean up mask with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        _save_debug_image(fg_mask, "grabcut_mask_cleaned", debug_dir)

        return _extract_quadrilateral_from_mask(fg_mask, (h, w), min_area_fraction=0.05)
    except Exception as e:
        logger.warning(f"GrabCut failed with error: {e}")
        return None


def strategy_hsv_segmentation(image: np.ndarray, debug_dir: Optional[Path] = None) -> Optional[np.ndarray]:
    """
    Strategy 2: HSV color-based segmentation for white/light paper.

    Tries two passes with different thresholds to detect paper on various backgrounds.
    """
    logger.debug("Trying Strategy 2: HSV")

    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _save_debug_image(hsv, "hsv_image", debug_dir)

    # First pass: more restrictive for white paper (S<80, V>140)
    mask = cv2.inRange(hsv, (0, 0, 140), (179, 80, 255))
    _save_debug_image(mask, "hsv_mask_pass1", debug_dir)

    # Clean up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    _save_debug_image(mask, "hsv_mask_cleaned_pass1", debug_dir)

    # Try to extract quadrilateral from first pass
    result = _extract_quadrilateral_from_mask(mask, image.shape[:2], min_area_fraction=0.05)
    if result is not None:
        return result

    # Second pass: more permissive (S<100, V>120) if first pass found nothing
    logger.debug("HSV pass 1 failed, trying pass 2 with lower thresholds")
    mask = cv2.inRange(hsv, (0, 0, 120), (179, 100, 255))
    _save_debug_image(mask, "hsv_mask_pass2", debug_dir)

    # Clean up mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    _save_debug_image(mask, "hsv_mask_cleaned_pass2", debug_dir)

    return _extract_quadrilateral_from_mask(mask, image.shape[:2], min_area_fraction=0.05)


def strategy_canny_contour(image: np.ndarray, debug_dir: Optional[Path] = None) -> Optional[np.ndarray]:
    """
    Strategy 3: Canny edge detection + contour finding.
    
    Uses lower thresholds and aggressive dilation for better edge detection
    on documents with poor contrast or textured backgrounds.
    """
    logger.debug("Trying Strategy 3: Canny")

    # Convert to grayscale and apply Gaussian blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny edge detection with lower thresholds
    edged = cv2.Canny(blurred, threshold1=20, threshold2=80)
    _save_debug_image(edged, "canny_edges", debug_dir)

    # Dilate edges aggressively to close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edged, kernel, iterations=3)
    _save_debug_image(dilated, "canny_dilated", debug_dir)

    h, w = image.shape[:2]
    min_area = 0.05 * (h * w)
    
    # Find contours and filter by area
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    # Filter contours by area
    large_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    if not large_contours:
        return None
    
    largest_contour = max(large_contours, key=cv2.contourArea)
    
    # Try multiple epsilon values for polygon approximation
    perimeter = cv2.arcLength(largest_contour, closed=True)
    for epsilon_factor in [0.01, 0.02, 0.03, 0.04, 0.05]:
        approx = cv2.approxPolyDP(largest_contour, epsilon_factor * perimeter, closed=True)
        if len(approx) == 4:
            logger.info(f"Quadrilateral found from Canny with area: {cv2.contourArea(largest_contour):.0f}")
            return approx.reshape(4, 2).astype("float32")
    
    # Try convex hull as fallback
    hull = cv2.convexHull(largest_contour)
    if len(hull) >= 4:
        hull_perimeter = cv2.arcLength(hull, True)
        for epsilon_factor in [0.01, 0.02, 0.03, 0.04, 0.05]:
            approx = cv2.approxPolyDP(hull, epsilon_factor * hull_perimeter, True)
            if len(approx) == 4:
                logger.info(f"Quadrilateral found from Canny hull with area: {cv2.contourArea(largest_contour):.0f}")
                return approx.reshape(4, 2).astype("float32")

    return None


def strategy_full_image_bounds(image: np.ndarray, debug_dir: Optional[Path] = None) -> Optional[np.ndarray]:
    """
    Strategy 4: Fallback detection using largest contour or full image bounds.
    
    First tries to find the largest contour and use its bounding box corners.
    Only uses full image bounds as absolute last resort.
    """
    logger.debug("Trying Strategy 4: Fallback detection")

    h, w = image.shape[:2]
    
    # Try to find the largest contour as a fallback
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=2)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        # Get bounding rectangle
        x, y, bw, bh = cv2.boundingRect(largest_contour)
        
        # Convert bounding box to corners
        if bw > 10 and bh > 10:  # Sanity check
            corners = np.array([
                [x, y],           # top-left
                [x + bw, y],      # top-right
                [x + bw, y + bh], # bottom-right
                [x, y + bh]       # bottom-left
            ], dtype="float32")
            
            logger.info(f"Using largest contour bounding box: {bw}×{bh}")
            return corners
    
    # Absolute last resort: full image bounds
    logger.warning("All detection strategies failed; using full image bounds as absolute last resort.")
    corners = np.array([
        [0, 0],        # top-left
        [w-1, 0],      # top-right
        [w-1, h-1],    # bottom-right
        [0, h-1]       # bottom-left
    ], dtype="float32")

    return corners


def detect_and_crop(image: np.ndarray, debug_dir: Optional[Path] = None) -> tuple[np.ndarray, bool]:
    """
    Main entry point: detect document corners using multi-strategy approach.

    Tries strategies in order:
    1. GrabCut segmentation
    2. HSV color-based segmentation
    3. Canny edge detection + contour finding
    4. Full image bounds (fallback - returns original image unchanged)

    Args:
        image: BGR image as numpy array.
        debug_dir: Optional path to save debug images for diagnosis.

    Returns:
        (cropped_image, success_flag)
        If detection fails completely, returns (original_image, False).
        For fallback strategy, returns (original_image, True).
    """
    orig_h, orig_w = image.shape[:2]

    # Work on a resized copy for faster processing, then scale corners back
    max_dim = 400
    scale = max_dim / max(orig_h, orig_w)
    resized = cv2.resize(image, (int(orig_w * scale), int(orig_h * scale)))

    # Try strategies in order
    strategies = [
        strategy_grabcut,
        strategy_hsv_segmentation,
        strategy_canny_contour,
        strategy_full_image_bounds,
    ]

    corners = None
    used_strategy = None
    for strategy in strategies:
        try:
            corners = strategy(resized, debug_dir)
            if corners is not None:
                used_strategy = strategy
                break
        except Exception as e:
            logger.warning(f"Strategy {strategy.__name__} failed with error: {e}")
            continue

    if corners is None:
        logger.error("All detection strategies failed!")
        return image, False

    # For fallback strategy, return original image unchanged
    if used_strategy == strategy_full_image_bounds:
        return image, True

    # Scale corners back to original image dimensions
    corners /= scale

    # Apply perspective transform
    try:
        cropped = four_point_transform(image, corners)
        success = cropped.shape[0] > 0 and cropped.shape[1] > 0
        return cropped, success
    except Exception as e:
        logger.error(f"Perspective transform failed: {e}")
        return image, False
