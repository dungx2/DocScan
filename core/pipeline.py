"""
pipeline.py
-----------
Orchestrates the full document processing pipeline:
  1. Corner detection & perspective crop
  2. Dewarping (neural, if model is loaded)
  3. Shadow removal & binarization

Designed to be called from the Flask API layer.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .corner_detection import detect_and_crop
from .dewarp import dewarp_document, load_model, DewarpModel
from .shadow_removal import remove_shadows_and_binarize

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    success: bool
    output_image: np.ndarray | None = None
    stages: dict = field(default_factory=dict)   # stage name → elapsed seconds
    warnings: list[str] = field(default_factory=list)
    error: str | None = None


class DocumentProcessor:
    """
    Stateful processor that holds the loaded ML model so it is only
    initialised once per application lifetime (not per request).
    """

    def __init__(self, model_path: str | None = None, device_str: str = "auto"):
        import torch

        if device_str == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_str)

        logger.info(f"DocumentProcessor initialised on device: {self.device}")

        self.model: DewarpModel | None = None
        if model_path:
            self.model = load_model(model_path, self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        image: np.ndarray,
        output_mode: str = "binary",
        binarize_method: str = "adaptive",
        skip_dewarp: bool = False,
        debug_dir: str | None = None,
    ) -> ProcessingResult:
        """
        Run the full pipeline on a BGR image.

        Args:
            image:            Raw BGR image from cv2.imdecode.
            output_mode:      "binary" | "grayscale" | "color"
            binarize_method:  "adaptive" | "sauvola" | "otsu"
            skip_dewarp:      If True, skip the neural dewarping step.
            debug_dir:        Optional path to save debug images from corner detection.

        Returns:
            ProcessingResult with the final image and per-stage timings.
        """
        result = ProcessingResult(success=False)
        timings: dict[str, float] = {}

        try:
            # ── Stage 1: Corner detection & crop ──────────────────────
            t0 = time.perf_counter()
            debug_path = Path(debug_dir) if debug_dir else None
            cropped, corners_found = detect_and_crop(image, debug_path)
            timings["corner_detection"] = round(time.perf_counter() - t0, 3)

            if not corners_found:
                result.warnings.append(
                    "Document corners not detected; processing full image."
                )

            # ── Stage 2: Dewarping ────────────────────────────────────
            # If corners were found successfully, skip neural dewarping because
            # the perspective crop already flattened the document geometrically.
            if corners_found:
                logger.info("Corners successfully found; skipping neural dewarping.")
                actual_skip_dewarp = True
            else:
                actual_skip_dewarp = skip_dewarp

            if not actual_skip_dewarp:
                t0 = time.perf_counter()
                dewarped = dewarp_document(cropped, model=self.model, device=self.device)
                timings["dewarping"] = round(time.perf_counter() - t0, 3)
            else:
                dewarped = cropped
                timings["dewarping"] = 0.0

            # ── Stage 3: Shadow removal & binarization ────────────────
            t0 = time.perf_counter()
            final = remove_shadows_and_binarize(
                dewarped,
                output_mode=output_mode,
                binarize_method=binarize_method,
            )
            timings["shadow_removal"] = round(time.perf_counter() - t0, 3)

            result.success = True
            result.output_image = final
            result.stages = timings
            logger.info(f"Pipeline complete. Timings: {timings}")

        except Exception as e:
            logger.exception("Pipeline failed.")
            result.error = str(e)

        return result


# ---------------------------------------------------------------------------
# Module-level convenience function (for simple scripts / tests)
# ---------------------------------------------------------------------------

def process_document(
    image: np.ndarray,
    output_mode: str = "binary",
    binarize_method: str = "adaptive",
) -> np.ndarray | None:
    """
    Stateless convenience wrapper — creates a temporary processor with no
    dewarping model. Suitable for Phase 1 (traditional CV only).
    """
    processor = DocumentProcessor(model_path=None)
    result = processor.process(
        image,
        output_mode=output_mode,
        binarize_method=binarize_method,
        skip_dewarp=True,
    )
    return result.output_image if result.success else None
