"""
dewarp.py  (FIXED — v2)
---------
Fixes applied vs the original:

  FIX 1 — Normalization mismatch (was the main cause of white output)
    BEFORE: _preprocess applied ImageNet mean/std at inference, but training
            used raw pixel values / 255.0 only. The model received inputs in
            [-2.1, 2.6] at inference vs [0, 1] at training → flow saturated
            at ±1 → every pixel sampled from the border → white output.
    AFTER:  _preprocess now just divides by 255.0, matching training exactly.

  FIX 2 — Displacement-based flow (better training stability)
    BEFORE: model output = absolute normalized coords in [-1, 1].
            Zero-init meant tanh(0)=0 everywhere → all pixels sampled from
            center of image → centre pixel smeared across output.
            Identity mapping requires spatially varying values (not 0), so
            the model had to learn very large deviations from its init.
    AFTER:  model output = displacement from identity grid.
            Zero-init → identity transform (no deformation) = correct start.
            _apply_flow adds the displacement to a pre-computed identity grid.

  NOTE: If you have existing weights trained with the OLD absolute-coord
        approach, set USE_DISPLACEMENT=False in this file to use them.
        They will still benefit from FIX 1 (the normalization fix).
        For any new training run, keep USE_DISPLACEMENT=True.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .model_arch import LightDewarpNet

logger = logging.getLogger(__name__)

DewarpModel = LightDewarpNet

# ── Compatibility flag ────────────────────────────────────────────────────────
# True  = new displacement-based training (recommended, train from scratch)
# False = old absolute-coord training (use if you have existing weights)
USE_DISPLACEMENT = True


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(weights_path: str, device: torch.device) -> LightDewarpNet | None:
    path = Path(weights_path)
    if not path.exists():
        logger.warning(
            f"Weights not found: {weights_path}\n"
            "  → Run: python scripts/train.py --data <doc3d_root>\n"
            "  → Or:  python scripts/generate_synthetic_dataset.py\n"
            "  → Continuing in Phase-1 (traditional CV) mode."
        )
        return None

    try:
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)

        if isinstance(checkpoint, dict):
            state_dict = (
                checkpoint.get("model_state_dict")
                or checkpoint.get("state_dict")
                or checkpoint
            )
            if "epoch" in checkpoint:
                logger.info(
                    f"Checkpoint: epoch={checkpoint.get('epoch')}, "
                    f"val_loss={checkpoint.get('val_loss', 'n/a')}"
                )
        else:
            state_dict = checkpoint

        model = LightDewarpNet().to(device)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"Missing keys ({len(missing)}): {missing[:3]} …")
        if unexpected:
            logger.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:3]} …")

        model.eval()
        logger.info(
            f"LightDewarpNet loaded from {weights_path} "
            f"on {device} ({model.count_parameters()/1e6:.1f} M params)"
        )
        return model

    except Exception as exc:
        logger.error(f"Could not load model: {exc}")
        return None


# ── Pre-processing ────────────────────────────────────────────────────────────
# FIX 1: Only divide by 255 — no ImageNet normalization.
# Training pipeline must match this exactly (see train.py).

def _preprocess(image: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    BGR uint8 → (1, 3, 448, 448) float32 tensor in [0, 1] on device.

    IMPORTANT: No ImageNet mean/std normalization here.
    Training uses the same [0,1] range (see Doc3DDataset in train.py).
    """
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    resized = cv2.resize(rgb, LightDewarpNet.INPUT_SIZE[::-1])   # (W, H) for cv2
    tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


# ── Identity grid helper ──────────────────────────────────────────────────────

def _make_identity_grid(h: int, w: int, device: torch.device) -> torch.Tensor:
    """
    Returns a (1, H, W, 2) grid where grid[0, y, x] = (norm_x, norm_y),
    i.e. the sampling coordinates of an identity (no-op) transform.
    Used for FIX 2: displacement-based flow application.
    """
    ys = torch.linspace(-1, 1, h, device=device)
    xs = torch.linspace(-1, 1, w, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # (1, H, W, 2)


# ── Flow application ──────────────────────────────────────────────────────────

def _apply_flow(image: np.ndarray, flow: torch.Tensor) -> np.ndarray:
    """
    Remap image pixels according to the model's flow field.

    FIX 2: If USE_DISPLACEMENT=True, the flow is a displacement (delta) from
    the identity grid rather than absolute coordinates. We add the identity
    before calling grid_sample.

    Args:
        image: Original BGR image (any resolution).
        flow:  (1, 2, H_model, W_model) tensor from the model.

    Returns:
        Dewarped BGR image at the original resolution.
    """
    h, w = image.shape[:2]

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)

    # Upsample flow to match original image resolution
    flow_up = F.interpolate(flow, size=(h, w), mode="bilinear", align_corners=True)
    # (1, 2, H, W) → (1, H, W, 2)
    flow_grid = flow_up.permute(0, 2, 3, 1)

    if USE_DISPLACEMENT:
        # FIX 2: add identity grid so that zero-output = no-op
        identity = _make_identity_grid(h, w, flow_grid.device)
        sampling_grid = identity + flow_grid
        # Clamp to valid range so grid_sample doesn't extrapolate wildly
        sampling_grid = sampling_grid.clamp(-1.0, 1.0)
    else:
        # Old mode: flow is absolute coords directly
        sampling_grid = flow_grid

    dewarped = F.grid_sample(
        img_tensor.to(flow_grid.device),
        sampling_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    result = dewarped.squeeze(0).permute(1, 2, 0).cpu().numpy()
    result = (result * 255.0).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)


# ── Tile-based inference for large images ─────────────────────────────────────

def _run_tiled(
    image: np.ndarray,
    model: LightDewarpNet,
    device: torch.device,
    tile_size: int = 448,
    overlap: int = 64,
) -> np.ndarray:
    h, w = image.shape[:2]
    output = np.zeros_like(image, dtype=np.float32)
    weight = np.zeros((h, w, 1), dtype=np.float32)
    stride = tile_size - overlap

    for y0 in range(0, h, stride):
        for x0 in range(0, w, stride):
            y1 = min(y0 + tile_size, h)
            x1 = min(x0 + tile_size, w)
            tile = image[y0:y1, x0:x1]

            pad_b = tile_size - (y1 - y0)
            pad_r = tile_size - (x1 - x0)
            if pad_b > 0 or pad_r > 0:
                tile = cv2.copyMakeBorder(tile, 0, pad_b, 0, pad_r, cv2.BORDER_REFLECT)

            inp = _preprocess(tile, device)
            with torch.no_grad():
                flow = model(inp)

            # Sanity check: if flow std is too high, model output may be unstable
            flow_std = flow.std().item()
            if flow_std > 0.48:
                logger.warning(
                    f"Flow std={flow_std:.3f} very high — model may be undertrained. "
                    "Skipping dewarp, returning original tile."
                )
                dewarped_tile = tile
            else:
                dewarped_tile = _apply_flow(tile, flow.float().cpu())

            th, tw = y1 - y0, x1 - x0
            dewarped_tile = dewarped_tile[:th, :tw]

            wy = np.hanning(th).reshape(-1, 1)
            wx = np.hanning(tw).reshape(1, -1)
            w_mask = (wy * wx)[:, :, np.newaxis].astype(np.float32)

            output[y0:y1, x0:x1] += dewarped_tile.astype(np.float32) * w_mask
            weight[y0:y1, x0:x1] += w_mask

    return (output / (weight + 1e-8)).clip(0, 255).astype(np.uint8)


# ── Public API ────────────────────────────────────────────────────────────────

def dewarp_document(
    image: np.ndarray,
    model: LightDewarpNet | None = None,
    device: torch.device | None = None,
    use_amp: bool = True,
    tiled: bool = False,
) -> np.ndarray:
    """
    Dewarp a document image.

    Args:
        image:   BGR image (any resolution).
        model:   Loaded LightDewarpNet, or None to skip dewarping.
        device:  Torch device (auto-detected if None).
        use_amp: Mixed-precision inference (fp16 on CUDA ≈ 2× speed).
        tiled:   Force tile mode (auto for images > 4 MP).
    """
    if model is None:
        logger.info("No model — returning image unchanged.")
        return image

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    h, w = image.shape[:2]
    if tiled or (h * w) > 4_000_000:
        logger.info(f"Tiled inference for {w}×{h} image.")
        return _run_tiled(image, model, device)

    inp = _preprocess(image, device)
    amp_enabled = use_amp and device.type == "cuda"

    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=amp_enabled):
        flow = model(inp)


    logger.info(
        f"Dewarp done | flow range [{flow.min():.3f}, {flow.max():.3f}] | "
        f"device={device} | amp={amp_enabled}"
    )
    return _apply_flow(image, flow.float().cpu())
