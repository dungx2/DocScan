"""
scripts/evaluate.py
-------------------
Evaluate a trained LightDewarpNet checkpoint on the Doc3D test split.

Metrics reported:
  MS-SSIM    — Multi-Scale Structural Similarity (higher is better; 1.0 = perfect)
  LD         — Local Distortion (lower is better; measures pixel displacement error)
  PSNR       — Peak Signal-to-Noise Ratio in dB (higher is better)
  Li score   — Average normalised flow L1 error (lower is better)

DewarpNet paper benchmarks on Doc3D test set:
  MS-SSIM ≈ 0.46,  LD ≈ 8.2   (traditional / no deep learning)
  MS-SSIM ≈ 0.47,  LD ≈ 7.9   (DewarpNet)
  MS-SSIM ≈ 0.50,  LD ≈ 7.3   (DocTr)

Usage:
  python scripts/evaluate.py --weights models/best_model.pth --data /path/to/doc3d
  python scripts/evaluate.py --weights models/best_model.pth --data /path/to/doc3d --samples 500
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.model_arch import LightDewarpNet
from core.dewarp import load_model, _preprocess, _apply_flow

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── Metrics ───────────────────────────────────────────────────────────────────

def _gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
    x = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g.outer(g)


def ms_ssim(
    pred: torch.Tensor, target: torch.Tensor, levels: int = 5
) -> float:
    """
    Compute Multi-Scale SSIM between two (B, C, H, W) tensors in [0, 1].
    Simplified implementation; for production use pytorch_msssim package.
    """
    weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    mcs_list = []

    x, y = pred.float(), target.float()
    for level in range(levels):
        C1, C2 = 0.01**2, 0.03**2
        mu_x  = F.avg_pool2d(x, 3, 1, 1)
        mu_y  = F.avg_pool2d(y, 3, 1, 1)
        mu_x2, mu_y2, mu_xy = mu_x**2, mu_y**2, mu_x*mu_y
        sig_x  = F.avg_pool2d(x**2, 3, 1, 1)  - mu_x2
        sig_y  = F.avg_pool2d(y**2, 3, 1, 1)  - mu_y2
        sig_xy = F.avg_pool2d(x*y,  3, 1, 1)  - mu_xy

        cs = ((2*sig_xy + C2) / (sig_x + sig_y + C2)).mean()
        mcs_list.append(cs.item())

        if level < levels - 1:
            x = F.avg_pool2d(x, 2)
            y = F.avg_pool2d(y, 2)

    # Final level includes luminance
    C1 = 0.01**2
    mu_x = F.avg_pool2d(x, 3, 1, 1)
    mu_y = F.avg_pool2d(y, 3, 1, 1)
    luminance = ((2*mu_x*mu_y + C1) / (mu_x**2 + mu_y**2 + C1)).mean().item()

    result = luminance
    for i, (w, cs) in enumerate(zip(weights[:-1], mcs_list[:-1])):
        result *= cs ** w
    result *= mcs_list[-1] ** weights[-1]
    return float(result)


def local_distortion(
    pred_flow: torch.Tensor, gt_flow: torch.Tensor, patch_size: int = 7
) -> float:
    """
    Local Distortion (LD): average L2 distance between predicted and
    ground-truth flow vectors, computed in local patches.

    Lower is better; unit is normalised pixels.
    """
    diff  = (pred_flow - gt_flow).pow(2).sum(dim=1, keepdim=True).sqrt()  # (B,1,H,W)
    # Average over patches
    pooled = F.avg_pool2d(diff, patch_size, stride=1, padding=patch_size // 2)
    return pooled.mean().item() * 1000   # scale to match paper values


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred.float(), target.float()).item()
    if mse == 0:
        return float("inf")
    return 10 * np.log10(1.0 / mse)


# ── Evaluator ─────────────────────────────────────────────────────────────────

class Evaluator:

    def __init__(self, model: LightDewarpNet, device: torch.device):
        self.model  = model
        self.device = device

    @torch.no_grad()
    def evaluate_batch(
        self, images: torch.Tensor, gt_flows: torch.Tensor
    ) -> dict[str, float]:
        images   = images.to(self.device)
        gt_flows = gt_flows.to(self.device)

        pred_flows = self.model(images)

        # Reconstruct images from predicted and GT flows to compute MS-SSIM / PSNR
        pred_imgs = F.grid_sample(
            images, pred_flows.permute(0, 2, 3, 1),
            mode="bilinear", padding_mode="border", align_corners=True
        )
        gt_imgs = F.grid_sample(
            images, gt_flows.permute(0, 2, 3, 1),
            mode="bilinear", padding_mode="border", align_corners=True
        )

        return {
            "ms_ssim": ms_ssim(pred_imgs, gt_imgs),
            "ld":      local_distortion(pred_flows, gt_flows),
            "psnr":    psnr(pred_imgs, gt_imgs),
            "l1_flow": F.l1_loss(pred_flows, gt_flows).item(),
        }

    def run(self, loader: DataLoader) -> dict[str, float]:
        totals: dict[str, float] = {}
        n = 0

        for i, batch in enumerate(loader):
            metrics = self.evaluate_batch(batch["image"], batch["flow"])
            for k, v in metrics.items():
                totals[k] = totals.get(k, 0.0) + v
            n += 1
            if (i + 1) % 10 == 0:
                logger.info(f"  Evaluated {n * loader.batch_size} samples…")

        return {k: v / max(n, 1) for k, v in totals.items()}


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Evaluate LightDewarpNet")
    p.add_argument("--weights", required=True, help="Path to .pth checkpoint")
    p.add_argument("--data",    required=True, help="Path to doc3d root")
    p.add_argument("--batch",   type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--samples", type=int, default=None,
                   help="Limit number of test samples (None = full test set)")
    p.add_argument("--cpu",     action="store_true")
    args = p.parse_args()

    device = torch.device("cpu" if args.cpu else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Device: {device}")

    model = load_model(args.weights, device)
    if model is None:
        logger.error("Could not load model. Exiting.")
        sys.exit(1)

    # Use the same dataset class as training but with no augmentation
    from scripts.train import Doc3DDataset
    dataset = Doc3DDataset(Path(args.data), augment=False)

    if args.samples and args.samples < len(dataset):
        indices = torch.randperm(len(dataset))[:args.samples].tolist()
        dataset = Subset(dataset, indices)
        logger.info(f"Evaluating on {args.samples} random samples.")

    loader = DataLoader(dataset, batch_size=args.batch,
                        shuffle=False, num_workers=args.workers)

    evaluator = Evaluator(model, device)
    logger.info(f"Evaluating {len(dataset)} samples…\n")
    metrics = evaluator.run(loader)

    logger.info("\n" + "═" * 50)
    logger.info("  EVALUATION RESULTS")
    logger.info("═" * 50)
    logger.info(f"  MS-SSIM  : {metrics['ms_ssim']:.4f}  (higher is better)")
    logger.info(f"  LD       : {metrics['ld']:.3f}   (lower is better)")
    logger.info(f"  PSNR     : {metrics['psnr']:.2f} dB  (higher is better)")
    logger.info(f"  L1 Flow  : {metrics['l1_flow']:.4f}  (lower is better)")
    logger.info("═" * 50)
    logger.info("\nDewarpNet paper benchmarks (Doc3D):")
    logger.info("  No correction: MS-SSIM=0.46, LD=8.2")
    logger.info("  DewarpNet:     MS-SSIM=0.47, LD=7.9")
    logger.info("  DocTr:         MS-SSIM=0.50, LD=7.3")


if __name__ == "__main__":
    main()
