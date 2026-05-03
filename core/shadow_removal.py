"""
shadow_removal.py
-----------------
Phase 4: Shadow removal, illumination normalisation, and binarization.

Implements a two-stage approach:
  1. Background illumination estimation via morphological operations to remove
     cast shadows and balance uneven lighting.
  2. Adaptive thresholding (Sauvola / OpenCV's adaptiveThreshold) to produce
     a clean high-contrast binary or near-binary output.

No deep learning required — this runs entirely on CPU via OpenCV.
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage 1: Shadow / illumination normalisation
# ---------------------------------------------------------------------------

def _estimate_background(gray: np.ndarray, kernel_size: int = 91) -> np.ndarray:
    """
    Estimate the background illumination by applying a large morphological
    closing operation. This captures slow-varying shading while ignoring text.

    kernel_size should be larger than the largest text stroke (~3–5x the font size).
    Use an odd number.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    background = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel)
    background = cv2.GaussianBlur(background, (kernel_size, kernel_size), 0)
    return background


def normalise_illumination(image: np.ndarray) -> np.ndarray:
    """
    Normalise the illumination of a BGR image to remove cast shadows and
    balance uneven lighting, returning a normalised BGR image.

    Algorithm:
      1. Convert to grayscale.
      2. Estimate background (illumination envelope).
      3. Divide the image by the background, then rescale.
      4. Apply the correction per-channel on the original BGR image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Adapt kernel size to image dimensions — at least 1% of the smallest side
    min_side = min(gray.shape[:2])
    kernel_size = max(91, int(min_side * 0.15) | 1)  # bitwise OR 1 ensures odd

    background = _estimate_background(gray, kernel_size).astype(np.float32)

    # Normalise: bright pixels in background → divide out the shading
    normalized_gray = (gray / (background + 1e-6)) * 255.0
    normalized_gray = np.clip(normalized_gray, 0, 255).astype(np.uint8)

    # Apply channel-wise correction to preserve colour (for colour output mode)
    result = np.zeros_like(image)
    for c in range(3):
        channel = image[:, :, c].astype(np.float32)
        result[:, :, c] = np.clip(
            (channel / (background + 1e-6)) * 255.0, 0, 255
        ).astype(np.uint8)

    logger.info("Illumination normalisation complete.")
    return result


# ---------------------------------------------------------------------------
# Stage 2: Binarization
# ---------------------------------------------------------------------------

def _sauvola_binarize(gray: np.ndarray, window_size: int = 25, k: float = 0.2) -> np.ndarray:
    """
    Sauvola's adaptive thresholding — robust against local lighting variations.

    T(x,y) = mean * (1 + k * ((std / R) - 1))
    where R is the dynamic range of standard deviation (default 128).
    """
    pad = window_size // 2
    padded = np.pad(gray.astype(np.float32), pad, mode="reflect")

    # Use integral images for O(1) window statistics
    integral = cv2.integral(padded)
    integral_sq = cv2.integral(padded ** 2)

    # Sliding window sums
    y, x = np.mgrid[0:gray.shape[0], 0:gray.shape[1]]
    r = pad

    def _integral_sum(ii, y1, x1, y2, x2):
        return ii[y2+1, x2+1] - ii[y1, x2+1] - ii[y2+1, x1] + ii[y1, x1]

    n = window_size ** 2
    sums = _integral_sum(integral, y, x, y + 2*r, x + 2*r).astype(np.float64)
    sums_sq = _integral_sum(integral_sq, y, x, y + 2*r, x + 2*r).astype(np.float64)

    mean = sums / n
    var = np.maximum((sums_sq / n) - mean ** 2, 0)
    std = np.sqrt(var)

    R = 128.0
    threshold = mean * (1.0 + k * ((std / R) - 1.0))
    binary = np.where(gray.astype(np.float64) >= threshold, 255, 0).astype(np.uint8)
    return binary


def binarize(gray: np.ndarray, method: str = "adaptive") -> np.ndarray:
    """
    Convert a grayscale image to binary (black text on white background).

    Args:
        gray:   Grayscale image.
        method: "adaptive" (OpenCV adaptive Gaussian) | "sauvola" | "otsu"

    Returns:
        Binary image (uint8, values 0 or 255).
    """
    if method == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "sauvola":
        binary = _sauvola_binarize(gray)
    else:  # default: adaptive Gaussian
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=31,
            C=10
        )

    logger.info(f"Binarization complete using method: {method}")
    return binary


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def remove_shadows_and_binarize(
    image: np.ndarray,
    output_mode: str = "binary",
    binarize_method: str = "adaptive"
) -> np.ndarray:
    """
    Full shadow-removal + binarization pipeline.

    Args:
        image:            BGR image as numpy array.
        output_mode:      "binary"    → black & white output (recommended for OCR)
                          "grayscale" → normalised grayscale
                          "color"     → illumination-corrected colour
        binarize_method:  "adaptive" | "sauvola" | "otsu"

    Returns:
        Processed image as numpy array (BGR for color, single-channel for others).
    """
    normalised = normalise_illumination(image)

    if output_mode == "color":
        return normalised

    gray = cv2.cvtColor(normalised, cv2.COLOR_BGR2GRAY)

    if output_mode == "grayscale":
        return gray

    # Binary output
    binary = binarize(gray, method=binarize_method)

    # Convert back to BGR so downstream code always receives a 3-channel image
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
