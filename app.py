"""
app.py
------
Flask backend for DocScan.
Exposes a RESTful API and serves the single-page frontend.
"""

import io
import logging
import uuid
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_file

from dotenv import load_dotenv
load_dotenv()
import config
from core.pipeline import DocumentProcessor

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG if config.DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── App factory ───────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH
app.config["SECRET_KEY"] = config.SECRET_KEY

# Initialise the document processor once (loads the ML model if available)
processor = DocumentProcessor(
    model_path=config.MODEL_PATH,
    device_str=config.DEVICE,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in config.ALLOWED_EXTENSIONS
    )


def _decode_image(file_bytes: bytes) -> np.ndarray | None:
    """Decode raw bytes to a BGR numpy array."""
    nparr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image


def _encode_image(image: np.ndarray, fmt: str = "png") -> bytes:
    """Encode a numpy array to image bytes."""
    ext = f".{fmt}"
    success, buf = cv2.imencode(ext, image)
    if not success:
        raise ValueError(f"Could not encode image as {fmt}")
    return buf.tobytes()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/process-document", methods=["POST"])
def process_document():
    """
    POST /api/process-document

    Accepts a multipart/form-data request with:
      - file:             image file (JPG or PNG)
      - output_mode:      "binary" | "grayscale" | "color"   (optional)
      - binarize_method:  "adaptive" | "sauvola" | "otsu"    (optional)
      - skip_dewarp:      "true" | "false"                   (optional)

    Returns:
      200 application/octet-stream  — processed PNG image
      400 / 422 / 500 application/json — error details
    """
    # ── Validate file ─────────────────────────────────────────────────────────
    if "file" not in request.files:
        return jsonify({"error": "No file part in request."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not _allowed_file(file.filename):
        return jsonify({
            "error": f"Unsupported file type. Allowed: {config.ALLOWED_EXTENSIONS}"
        }), 422

    # ── Parse options ─────────────────────────────────────────────────────────
    output_mode      = request.form.get("output_mode", config.DEFAULT_OUTPUT_MODE)
    binarize_method  = request.form.get("binarize_method", config.DEFAULT_BINARIZE_METHOD)
    skip_dewarp      = request.form.get("skip_dewarp", "false").lower() == "true"

    if output_mode not in ("binary", "grayscale", "color"):
        return jsonify({"error": f"Invalid output_mode: {output_mode}"}), 422
    if binarize_method not in ("adaptive", "sauvola", "otsu"):
        return jsonify({"error": f"Invalid binarize_method: {binarize_method}"}), 422

    # ── Decode image ──────────────────────────────────────────────────────────
    file_bytes = file.read()
    image = _decode_image(file_bytes)
    if image is None:
        return jsonify({"error": "Could not decode image. Is it a valid JPG/PNG?"}), 422

    logger.info(
        f"Processing image: {file.filename} | "
        f"shape={image.shape} | mode={output_mode} | "
        f"binarize={binarize_method} | skip_dewarp={skip_dewarp}"
    )

    # ── Run pipeline ──────────────────────────────────────────────────────────
    result = processor.process(
        image,
        output_mode=output_mode,
        binarize_method=binarize_method,
        skip_dewarp=skip_dewarp,
    )

    if not result.success:
        logger.error(f"Pipeline failed: {result.error}")
        return jsonify({"error": result.error or "Processing failed."}), 500

    # ── Optionally persist output ─────────────────────────────────────────────
    out_filename = f"{uuid.uuid4().hex}.png"
    out_path = config.OUTPUT_DIR / out_filename
    cv2.imwrite(str(out_path), result.output_image)
    logger.info(f"Output saved to {out_path}")

    # ── Return processed image ────────────────────────────────────────────────
    img_bytes = _encode_image(result.output_image, fmt="png")
    response = send_file(
        io.BytesIO(img_bytes),
        mimetype="image/png",
        as_attachment=False,
        download_name="processed.png",
    )

    # Attach timing metadata as a custom header
    import json
    response.headers["X-Processing-Stages"] = json.dumps(result.stages)
    if result.warnings:
        response.headers["X-Warnings"] = " | ".join(result.warnings)

    return response


@app.route("/api/health", methods=["GET"])
def health():
    """Simple health check for monitoring."""
    import torch
    return jsonify({
        "status": "ok",
        "device": str(processor.device),
        "model_loaded": processor.model is not None,
        "cuda_available": torch.cuda.is_available(),
    })


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=config.PORT, debug=config.DEBUG)
