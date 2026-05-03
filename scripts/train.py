"""
train.py  (FIXED — v4)
--------
Fixes vs original:

  FIX 1 — Normalization: no ImageNet mean/std applied during training.
           Input tensors are [0, 1] float32 — exactly matching _preprocess()
           in dewarp.py. Before this fix, inference used ImageNet normalization
           but training did not → inputs looked completely different to the
           model at inference time.

  FIX 2 — Displacement targets: ground-truth bm is converted from absolute
           coords to displacement from identity grid. This means the model
           learns small deltas (mostly near 0) rather than large spatially-
           varying absolute coordinates. Matches USE_DISPLACEMENT=True in
           dewarp.py. Zero-init of the flow head → identity transform at
           start → physically sensible initialization.

  FIX 3 — Learning rate: restored to 3e-4 (was accidentally changed to 3e-5,
           10× too slow).

  FIX 4 — bm normalization: divide by (size-1) = 447, not 448. Pixel coords
           run 0..447 so dividing by 447 maps max coord to exactly 1.0.

  FIX 5 — Augmentation: flow x-channel flip now works correctly in numpy.

  FIX 6 — Val augmentation bug: random_split returns Subset objects sharing
           the same dataset. Setting val_set.dataset.augment=False disables
           augmentation for BOTH train and val. Fixed by creating separate
           dataset instances with independent augment flags.
"""

import argparse
import copy
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_msssim import ms_ssim
from torch.amp import GradScaler
from torch.utils.data import DataLoader, Dataset, random_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.model_arch import LightDewarpNet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

IMG_SIZE = 448


# ── Identity grid (for computing displacement targets) ────────────────────────

def _identity_grid(size: int) -> np.ndarray:
    """
    Returns (2, size, size) float32 array with values in [-1, 1].
    identity[0] = x coords (increases left→right)
    identity[1] = y coords (increases top→bottom)
    """
    lin = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(lin, lin)   # both (size, size)
    return np.stack([grid_x, grid_y], axis=0)  # (2, size, size)

_IDENTITY = _identity_grid(IMG_SIZE)  # precompute once


# ── Dataset ───────────────────────────────────────────────────────────────────

class Doc3DDataset(Dataset):
    """
    Doc3D dataset reader.

    Each sample:
      image : (3, 448, 448) float32 in [0, 1]   ← FIX 1: no ImageNet norm
      flow  : (2, 448, 448) float32 displacement ← FIX 2: delta from identity
    """

    def __init__(self, root: Path, augment: bool = True):
        self.root    = root
        self.augment = augment
        self.img_dir = root / "img"
        self.bm_dir  = root / "bm"
        self.flat_dir = root / "flat"
        self.has_flat = self.flat_dir.exists()

        self.samples: list[tuple[Path, Path]] = []
        for img_path in sorted(self.img_dir.rglob("*.png")):
            rel     = img_path.relative_to(self.img_dir).with_suffix(".mat")
            bm_path = self.bm_dir / rel
            if bm_path.exists():
                self.samples.append((img_path, bm_path))

        if not self.samples:
            raise FileNotFoundError(
                f"No img/bm pairs found under {root}. "
                "Check that both img/ and bm/ directories exist."
            )
        logger.info(f"Doc3DDataset: {len(self.samples)} samples in {root}")
        if self.has_flat:
            flat_count = len(list(self.flat_dir.glob("*.png")))
            logger.info(f"Doc3DDataset: {flat_count} flat images available")
        else:
            logger.info("Doc3DDataset: no flat images available")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        img_path, bm_path = self.samples[idx]

        # ── Image ─────────────────────────────────────────────────────
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            raise IOError(f"Cannot read: {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # FIX 1: just / 255 — no ImageNet mean/std
        rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0

        # ── Backward map ──────────────────────────────────────────────
        # Doc3D .mat files are HDF5 (MATLAB v7.3 format)
        import h5py
        with h5py.File(str(bm_path), "r") as f:
            # h5py reverses MATLAB dimension order:
            # MATLAB (448, 448, 2) → h5py (2, 448, 448)
            bm = np.array(f["bm"])   # (2, 448, 448) — pixel coords

        # FIX 4: divide by (size-1) so pixel 447 maps to exactly 1.0
        bm_norm = bm.astype(np.float32) / (IMG_SIZE - 1)   # [0, 1]
        bm_norm = bm_norm * 2.0 - 1.0                       # [-1, 1] absolute coords

        # FIX 2: convert absolute → displacement from identity
        # The model learns small corrections, not the full absolute mapping.
        # Zero-output at init → identity → correct starting point.
        flow = bm_norm - _IDENTITY                           # (2, 448, 448)

        # ── Flat image ────────────────────────────────────────────────
        flat = None
        if self.has_flat:
            flat_path = self.flat_dir / img_path.name
            if flat_path.exists():
                flat_bgr = cv2.imread(str(flat_path))
                if flat_bgr is not None:
                    flat_rgb = cv2.cvtColor(flat_bgr, cv2.COLOR_BGR2RGB)
                    flat = cv2.resize(flat_rgb, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
                    flat = torch.from_numpy(np.ascontiguousarray(flat)).permute(2, 0, 1)

        # ── Augmentation ──────────────────────────────────────────────
        if self.augment:
            rgb, flow = self._augment(rgb, flow)

        result = {
            "image": torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1),
            "flow":  torch.from_numpy(np.ascontiguousarray(flow)),
            "flat": flat,
        }
        return result

    @staticmethod
    def _augment(rgb: np.ndarray, flow: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Horizontal flip — FIX 5: use np.ascontiguousarray after flip
        if np.random.rand() < 0.5:
            rgb  = np.ascontiguousarray(np.fliplr(rgb))
            # x-displacement flips sign AND direction; y just flips direction
            flow_x = np.ascontiguousarray(np.fliplr(-flow[0]))
            flow_y = np.ascontiguousarray(np.fliplr( flow[1]))
            flow   = np.stack([flow_x, flow_y])

        # Brightness / contrast jitter
        alpha = np.random.uniform(0.8, 1.2)
        beta  = np.random.uniform(-0.1, 0.1)
        rgb   = np.clip(alpha * rgb + beta, 0.0, 1.0)

        # Colour channel shuffle
        if np.random.rand() < 0.2:
            rgb = rgb[:, :, np.random.permutation(3)]

        # 1. Random background paste (prob=0.3)
        if np.random.rand() < 0.3:
            h, w = rgb.shape[:2]
            if np.random.rand() < 0.5:
                bg = np.random.rand(1, 1, 3) * np.ones((h, w, 3))
            else:
                bg = np.random.rand(h, w, 3) * 0.5 + 0.25
            
            mask = (rgb.sum(axis=2) > 0.05)[..., None]
            rgb = np.where(mask, rgb, bg).astype(np.float32)

        # 2. Gaussian noise (prob=0.4)
        if np.random.rand() < 0.4:
            noise = np.random.normal(0, np.random.uniform(0.005, 0.02), rgb.shape)
            rgb = np.clip(rgb + noise, 0.0, 1.0).astype(np.float32)

        # 3. Random JPEG compression simulation (prob=0.3)
        if np.random.rand() < 0.3:
            rgb_uint8 = (rgb * 255.0).clip(0, 255).astype(np.uint8)
            quality = int(np.random.uniform(40, 85))
            _, encoded = cv2.imencode('.jpg', rgb_uint8, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            rgb = decoded.astype(np.float32) / 255.0

        # 4. Random shadow overlay (prob=0.3)
        if np.random.rand() < 0.3:
            h, w = rgb.shape[:2]
            shadow_mask = np.ones((h, w, 3), dtype=np.float32)
            num_points = np.random.randint(3, 6)
            points = np.random.randint(0, min(h, w), size=(num_points, 2))
            
            hull = cv2.convexHull(points)
            
            alpha_shadow = np.random.uniform(0.3, 0.6)
            cv2.fillConvexPoly(shadow_mask, hull, (1 - alpha_shadow, 1 - alpha_shadow, 1 - alpha_shadow))
            
            rgb = (rgb * shadow_mask).clip(0.0, 1.0).astype(np.float32)

        # 5. Slight random rotation (prob=0.3)
        if np.random.rand() < 0.3:
            h, w = rgb.shape[:2]
            angle = np.random.uniform(-10, 10)
            center = (w / 2, h / 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            rgb = cv2.warpAffine(rgb, M, (w, h), borderMode=cv2.BORDER_REFLECT101)
            rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)
            
            flow_x = cv2.warpAffine(flow[0], M, (w, h), borderMode=cv2.BORDER_REPLICATE)
            flow_y = cv2.warpAffine(flow[1], M, (w, h), borderMode=cv2.BORDER_REPLICATE)
            
            theta = np.deg2rad(angle)
            cos_th = np.cos(theta)
            sin_th = np.sin(theta)
            
            new_flow_x = flow_x * cos_th + flow_y * sin_th
            new_flow_y = -flow_x * sin_th + flow_y * cos_th
            
            flow = np.stack([new_flow_x, new_flow_y])
            flow = np.clip(flow, -1.0, 1.0).astype(np.float32)

        return rgb, flow


class IndexSubset:
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


def collate_fn(batch):
    """
    Custom collate function that handles None flat images.
    If any sample in the batch has flat=None, set the whole batch flat to None,
    otherwise stack normally.
    """
    images = torch.stack([item["image"] for item in batch])
    flows = torch.stack([item["flow"] for item in batch])
    
    flats = [item["flat"] for item in batch]
    if any(flat is None for flat in flats):
        flat_batch = None
    else:
        flat_batch = torch.stack(flats)
    
    return {
        "image": images,
        "flow": flows,
        "flat": flat_batch,
    }


# ── Loss ──────────────────────────────────────────────────────────────────────

class CombinedLoss(nn.Module):
    """
    L1 flow loss + smoothness regularisation + MS-SSIM reconstruction loss.

    Combination of:
    - L1: Direct flow prediction loss (warping accuracy)
    - Smoothness: Total-variation regularisation (natural flow fields)
    - MS-SSIM: Multi-scale structural similarity reconstruction loss
    """

    def __init__(self, smooth_weight: float = 0.1, ssim_weight: float = 0.5, phase: int = 1):
        super().__init__()
        self.l1     = nn.L1Loss()
        self.smooth = smooth_weight
        self.ssim_weight = ssim_weight
        self.phase  = phase

    def _smoothness(self, flow: torch.Tensor) -> torch.Tensor:
        diff_x = flow[:, :, :, 1:] - flow[:, :, :, :-1]
        diff_y = flow[:, :, 1:, :] - flow[:, :, :-1, :]
        return diff_x.abs().mean() + diff_y.abs().mean()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        image: torch.Tensor = None,
        flat: torch.Tensor = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute combined loss.
        
        Args:
            pred:   (B, 2, H, W) predicted flow displacement
            target: (B, 2, H, W) ground-truth flow displacement
            image:  (B, 3, H, W) input images (optional, for MS-SSIM)
            flat:   (B, 3, H, W) flat ground-truth images (optional, for MS-SSIM)

        Returns:
            (total_loss, dict of breakdown)
        """
        l1_loss     = self.l1(pred, target)
        smooth_loss = self._smoothness(pred)

        ssim_loss = 0.0
        if self.phase == 2 and image is not None and flat is not None:
            B, _, H, W = pred.shape
            ys = torch.linspace(-1, 1, H, device=pred.device)
            xs = torch.linspace(-1, 1, W, device=pred.device)
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
            identity = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
            flow_grid = pred.permute(0, 2, 3, 1)  # (B,H,W,2)
            sampling_grid = (identity + flow_grid).clamp(-1, 1)
            warped = F.grid_sample(image, sampling_grid, mode='bilinear', padding_mode='border', align_corners=True)
            
            # DEBUG: Print tensor shapes and properties
            if not hasattr(self, '_debug_printed'):
                self._debug_printed = True
                print(f"DEBUG warped: shape={warped.shape}, min={warped.min():.3f}, max={warped.max():.3f}, requires_grad={warped.requires_grad}")
                print(f"DEBUG flat:   shape={flat.shape},   min={flat.min():.3f},   max={flat.max():.3f}, requires_grad={flat.requires_grad}")
                print(f"DEBUG pred:   shape={pred.shape},   min={pred.min():.3f},   max={pred.max():.3f}, requires_grad={pred.requires_grad}")
                print(f"DEBUG sampling_grid: min={sampling_grid.min():.3f}, max={sampling_grid.max():.3f}, requires_grad={sampling_grid.requires_grad}")
            
            ssim_loss = 1 - ms_ssim(warped, flat, data_range=1.0, size_average=True)
        
        total = l1_loss + self.smooth * smooth_loss + self.ssim_weight * ssim_loss 

        return total, {
            "l1":     l1_loss.item(),
            "smooth": smooth_loss.item(),
            "ssim":   ssim_loss.item() if isinstance(ssim_loss, torch.Tensor) else ssim_loss,
            "total":  total.item(),
        }


# ── Trainer ───────────────────────────────────────────────────────────────────

class Trainer:

    def __init__(self, args: argparse.Namespace):
        self.args   = args
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
        )
        logger.info(f"Training on: {self.device}")

        self.model     = LightDewarpNet().to(self.device)
        self.criterion = CombinedLoss(smooth_weight=args.smooth_weight, ssim_weight=args.ssim_weight, phase=args.phase)
        
        if args.phase == 2:
            # Freeze encoder
            for module in [self.model.enc1, self.model.enc2, self.model.enc3]:
                for param in module.parameters():
                    param.requires_grad = False
            lr_phase = args.lr / 10
        else:
            lr_phase = args.lr
        
        self.optimizer = optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=lr_phase,
            weight_decay=args.weight_decay,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.epochs, eta_min=lr_phase * 0.01
        )
        self.scaler = GradScaler("cuda", enabled=(self.device.type == "cuda"))

        self.out_dir = Path(args.output)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.best_val_loss = float("inf")
        self.start_epoch   = 0

        if args.resume:
            self._load_checkpoint(args.resume)

        logger.info(
            f"Phase {args.phase} | Model: {self.model.count_parameters()/1e6:.1f}M params ({sum(p.numel() for p in self.model.parameters() if p.requires_grad)/1e6:.1f}M trainable) | "
            f"VRAM est: {self.model.estimate_vram_mb(args.batch):.0f} MB"
        )

    def _load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])

        # Fix 2: Try to load optimizer state; skip if parameter groups changed (phase 2 freeze)
        try:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except ValueError as e:
            logger.warning(
                f"Optimizer state skipped — parameter groups changed (phase 2 freeze). "
                f"Starting optimizer fresh."
            )

        # Fix 1: Reset epoch counter and scheduler for continued training
        self.start_epoch   = 0
        if self.args.phase == 2:
            self.best_val_loss = float("inf")
        else:
            self.best_val_loss = ckpt.get("val_loss", float("inf"))

        # Fix 1: Recreate scheduler starting fresh at current LR
        lr_phase = self.optimizer.param_groups[0]['lr']
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.args.epochs, eta_min=lr_phase * 0.01
        )

        logger.info(f"Resumed from {path} (starting from epoch 0 for continued training)")

    def _save_checkpoint(self, epoch: int, val_loss: float, is_best: bool):
        ckpt = {
            "epoch":                epoch,
            "val_loss":             val_loss,
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "args":                 vars(self.args),
            "use_displacement":     True,   # tag so dewarp.py knows which mode
        }
        torch.save(ckpt, self.out_dir / f"checkpoint_epoch{epoch:04d}.pth")
        if is_best:
            best = self.out_dir / "best_model.pth"
            torch.save(ckpt, best)
            logger.info(f"  ★ Best model → {best}")

    def _run_epoch(self, loader: DataLoader, train: bool) -> dict[str, float]:
        self.model.train(train)
        totals = {"l1": 0.0, "smooth": 0.0, "ssim": 0.0, "total": 0.0}
        n = 0

        ctx = torch.enable_grad() if train else torch.no_grad()
        amp = torch.autocast(
            device_type=self.device.type,
            enabled=(self.device.type == "cuda")
        )

        with ctx:
            for batch in loader:
                images = batch["image"].to(self.device)
                flows  = batch["flow"].to(self.device)
                batch_flat = batch.get("flat")
                if batch_flat is not None:
                    batch_flat = batch_flat.to(self.device)

                with amp:
                    pred = self.model(images)
                    loss, breakdown = self.criterion(pred, flows, image=images, flat=batch_flat)

                if train:
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                for k, v in breakdown.items():
                    totals[k] += v
                n += 1

        return {k: v / max(n, 1) for k, v in totals.items()}

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        logger.info(
            f"Training: {self.args.epochs} epochs | "
            f"lr={self.optimizer.param_groups[0]['lr']} | "
            f"train={len(train_loader.dataset)} | val={len(val_loader.dataset)}"
        )
        for epoch in range(self.start_epoch, self.args.epochs):
            t0 = time.time()
            tr  = self._run_epoch(train_loader, train=True)
            val = self._run_epoch(val_loader,   train=False)
            self.scheduler.step()

            val_loss = val["total"]
            is_best  = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            logger.info(
                f"Epoch [{epoch+1:03d}/{self.args.epochs}] "
                f"{time.time()-t0:.0f}s | "
                f"train={tr['total']:.4f} (l1={tr['l1']:.4f} sm={tr['smooth']:.4f} ss={tr['ssim']:.4f}) | "
                f"val={val_loss:.4f} {'★' if is_best else ''}"
            )

            if (epoch + 1) % self.args.save_every == 0 or is_best:
                self._save_checkpoint(epoch, val_loss, is_best)

        logger.info(f"Done. Best val loss: {self.best_val_loss:.4f}")
        logger.info(f"Best model: {self.out_dir / 'best_model.pth'}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LightDewarpNet (fixed)")
    p.add_argument("--data",          required=True)
    p.add_argument("--output",        default="models/training")
    p.add_argument("--epochs",        type=int,   default=50)
    p.add_argument("--batch",         type=int,   default=4)
    p.add_argument("--lr",            type=float, default=3e-4)   # FIX 3
    p.add_argument("--weight-decay",  type=float, default=1e-4)
    p.add_argument("--smooth-weight", type=float, default=0.05)
    p.add_argument("--ssim-weight",   type=float, default=0.5)
    p.add_argument("--val-split",     type=float, default=0.1)
    p.add_argument("--workers",       type=int,   default=4)
    p.add_argument("--save-every",    type=int,   default=5)
    p.add_argument("--resume",        default=None)
    p.add_argument("--cpu",           action="store_true")
    p.add_argument("--phase",         type=int,   default=1, choices=[1, 2])
    return p.parse_args()


def main():
    args = parse_args()

    # FIX 6: Create separate dataset instances for train and val
    # DO NOT use random_split with a shared dataset, as setting
    # augment on one Subset breaks augmentation for both.
    root_path = Path(args.data)
    img_paths = sorted(list((root_path / "img").glob("*.png")))
    n_total = len(img_paths)
    n_val   = int(n_total * args.val_split)
    n_train = n_total - n_val

    # Split the file paths list manually by shuffling with a fixed seed
    np.random.seed(42)
    indices = np.random.permutation(n_total)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    # Create separate dataset instances
    train_dataset = Doc3DDataset(root_path, augment=True)
    val_dataset   = Doc3DDataset(root_path, augment=False)  # FIX 6: no augmentation

    train_set = IndexSubset(train_dataset, train_indices)
    val_set   = IndexSubset(val_dataset, val_indices)

    train_loader = DataLoader(
        train_set, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        collate_fn=collate_fn,
    )

    trainer = Trainer(args)
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
