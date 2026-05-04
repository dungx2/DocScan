# DocScan — Document Flattening Pipeline

A client-server document scanner that detects document corners, removes shadows,
and dewarps curved pages using a custom-trained neural network (LightDewarpNet).

```
Upload photo → Corner Detection & Perspective → Neural Dewarp → Shadow Removal → Download
```

---

## Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.11+, Flask 3 |
| CV Core | OpenCV 4.9+, NumPy |
| AI / Dewarping | PyTorch 2.2+ (CUDA optional), pytorch-msssim |
| Frontend | Vanilla HTML/CSS/JS (no build step) |

---

## Quickstart
Download doc3d-dataset from:
https://github.com/cvlab-stonybrook/doc3D-dataset
### 1. Clone and set up environment

```bash
git clone https://github.com/yourname/docscan.git
cd docscan

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. GPU support (RTX 3050 / CUDA 12.1)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Verify GPU:

```python
import torch
print(torch.cuda.is_available())       # → True
print(torch.cuda.get_device_name(0))   # → NVIDIA GeForce RTX 3050
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env — set DOCSCAN_MODEL_PATH to your trained weights
```

### 4. Run

```bash
python app.py
# Open http://localhost:5000
```

---

## Project Structure

```
docscan/
├── app.py                        # Flask application & API routes
├── config.py                     # All configuration (reads .env)
├── requirements.txt
├── .env.example
│
├── core/                         # AI / CV processing pipeline
│   ├── __init__.py
│   ├── pipeline.py               # Orchestrator — calls all stages
│   ├── corner_detection.py       # Stage 1: Canny → contours → warpPerspective
│   ├── dewarp.py                 # Stage 2: Neural dewarping (LightDewarpNet)
│   ├── shadow_removal.py         # Stage 3: Illumination normalisation + binarization
│   └── model_arch.py             # LightDewarpNet U-Net architecture (~12M params)
│
├── data/
│   └── doc3d_subset/             # Training dataset (Doc3D subset)
│       ├── img/                  # Distorted/curved document images (PNG)
│       ├── bm/                   # Backward maps — ground truth flow (HDF5 .mat)
│       └── flat/                 # Rectified/flat ground truth images (PNG)
│
├── models/
│   └── training/                 # Trained model weights
│       ├── best_model.pth        # Best checkpoint (lowest val loss)
│       └── checkpoint_epoch*.pth # Periodic checkpoints
│
├── scripts/
│   ├── train.py                  # Training script (v3 — all fixes applied)
│   └── generate_synthetic_dataset.py
│
├── uploads/                      # Temporary upload storage (git-ignored)
├── outputs/                      # Processed output images (git-ignored)
│
├── static/                       # Frontend static files (CSS, JS)
└── templates/
    └── index.html                # Single-page frontend
```

---

## API

### `POST /api/process-document`

| Field | Type | Description |
|---|---|---|
| `file` | multipart file | JPG or PNG image |
| `output_mode` | string | `binary` \| `grayscale` \| `color` |
| `binarize_method` | string | `adaptive` \| `sauvola` \| `otsu` |
| `skip_dewarp` | string | `true` \| `false` |

**Response:** `image/png` binary
**Headers:** `X-Processing-Stages` (JSON timing per stage), `X-Warnings` (optional)

### `GET /api/health`

Returns JSON with device info (CPU/GPU) and model load status.

---

## Processing Pipeline — Stage by Stage

### Stage 1 — Corner Detection & Perspective Correction (`core/corner_detection.py`)

1. Convert to grayscale → Gaussian Blur (noise reduction)
2. Canny Edge Detection → find contours
3. Select largest quadrilateral contour → compute Homography Matrix
4. `cv2.warpPerspective()` → output looks as if shot from directly above

### Stage 2 — Neural Dewarping (`core/dewarp.py`, `core/model_arch.py`)

Handles 3D page curl that warpPerspective cannot fix (books, notebooks).

- **Model:** LightDewarpNet — U-Net style encoder-decoder, ~12M parameters
- **Input:** Curved document image (448×448 RGB)
- **Output:** 2-channel displacement flow map (2×448×448)
- **Inference:** Flow map is used to warp the input image back to flat

> If no model weights are present, Stage 2 is skipped automatically. Stage 1 alone handles the majority of real-world cases.

### Stage 3 — Shadow Removal & Binarization (`core/shadow_removal.py`)

**Output modes:**

| Mode | Description | Best for |
|---|---|---|
| `color` | Illumination normalisation only, keeps full color | ID cards, branded documents |
| `grayscale` | Convert to grey + normalise, no binarization | Detailed scans, downstream ML |
| `binary` | Full binarization — white background, black text | OCR pipelines |

**Binarization methods (binary mode only):**

| Method | How it works | Best for |
|---|---|---|
| `otsu` | Single global threshold via histogram analysis | Even lighting, clean white background |
| `adaptive` | Per-block Gaussian weighted threshold | Local shadows, uneven lighting |
| `sauvola` | Per-block threshold using mean + std deviation | Old documents, yellowed/uneven backgrounds |

**Total Loss = L1 + Smoothness + MS-SSIM**, see Training section for formula.

---

## Training the Dewarping Model

### Dataset — Doc3D Subset

| Property | Value |
|---|---|
| Dataset | Doc3D (Stony Brook University) — synthetic 3D renders |
| Subset size | ~1,000 images |
| Train / Val split | 900 / 100 (90% / 10%) |
| Image resolution | 448 × 448 pixels |
| Input format | PNG (RGB, uint8) |
| Ground truth flow | HDF5 `.mat` files, shape `(2, 448, 448)` |
| Ground truth flat | PNG — rectified reference image |

Required folder structure:

```
data/doc3d_subset/
├── img/     ← curved/distorted images
├── bm/      ← backward maps (HDF5 .mat) — 1:1 match with img/
└── flat/    ← flat ground truth images — recommended for MS-SSIM loss
```

> If `flat/` is absent, MS-SSIM loss is skipped. Training still works with L1 + Smoothness only, but output quality will be lower.

### Model Architecture — LightDewarpNet

- **Type:** U-Net style Encoder-Decoder with skip connections
- **Parameters:** ~12 million
- **Input:** RGB image tensor `(B, 3, 448, 448)`
- **Output:** Displacement flow `(B, 2, 448, 448)`
- **File:** `core/model_arch.py`

Skip connections preserve spatial detail from encoder layers — essential for accurate per-pixel displacement prediction.

### Loss Function

```
Total Loss = L1 + (smooth_weight × Smoothness) + (ssim_weight × (1 − MS-SSIM))
           = L1 + 0.1 × Smoothness + 0.5 × (1 − MS-SSIM)
```

| Component | Role |
|---|---|
| **L1 Loss** | Flow prediction accuracy vs ground truth backward map |
| **Smoothness Loss** | TV regularization — prevents noisy/jagged flow fields |
| **MS-SSIM Loss** | Reconstruction quality — compares warped output vs flat GT |

MS-SSIM loss requires the model to produce a visually sharp result, not just a numerically accurate flow field.

### Training Commands

**Recommended (good balance, ~2 hours on RTX 3050):**

```powershell
cd D:\IPR\docscan
python scripts/train.py `
  --data "data/doc3d_subset" `
  --epochs 50 `
  --batch 4 `
  --lr 3e-4 `
  --smooth-weight 0.1 `
  --ssim-weight 0.5
```

**Quick test run (~5 minutes):**

```powershell
python scripts/train.py --data "data/doc3d_subset" --epochs 5 --batch 8
```

**High quality (~4 hours on GPU):**

```powershell
python scripts/train.py `
  --data "data/doc3d_subset" `
  --epochs 100 `
  --batch 4 `
  --lr 3e-4 `
  --smooth-weight 0.1 `
  --ssim-weight 0.5 `
  --save-every 2
```

**Resume from checkpoint:**

```powershell
python scripts/train.py `
  --data "data/doc3d_subset" `
  --resume "models/training/best_model.pth" `
  --epochs 100
```

**CPU only (no GPU):**

```powershell
python scripts/train.py --data "data/doc3d_subset" --batch 2 --cpu
```

### Training Parameters Reference

| Parameter | Default | Description |
|---|---|---|
| `--data` | required | Path to dataset root with `img/`, `bm/`, `flat/` |
| `--output` | `models/training` | Where to save checkpoints |
| `--epochs` | `50` | Number of training passes |
| `--batch` | `4` | Batch size (larger = faster, more VRAM) |
| `--lr` | `3e-4` | Adam learning rate |
| `--weight-decay` | `1e-4` | L2 regularization |
| `--smooth-weight` | `0.1` | Smoothness loss coefficient |
| `--ssim-weight` | `0.5` | MS-SSIM loss coefficient |
| `--val-split` | `0.1` | Fraction of data for validation |
| `--workers` | `4` | DataLoader worker threads |
| `--save-every` | `5` | Save checkpoint every N epochs |
| `--resume` | none | Resume training from checkpoint path |
| `--cpu` | off | Force CPU mode |

### Expected Training Output

```
2026-04-30 14:32:15 [INFO] Training on: cuda
2026-04-30 14:32:15 [INFO] Doc3DDataset: 900 samples (flat images: yes)
2026-04-30 14:32:15 [INFO] Model: 12.0M params | VRAM est: 1250 MB
2026-04-30 14:32:15 [INFO] Training: 50 epochs | lr=0.0003 | train=810 | val=90

Epoch [001/050] 145s | train=0.1524 (l1=0.1089 sm=0.0289 ss=0.0146) | val=0.1342 ★
Epoch [002/050] 142s | train=0.1234 (l1=0.0856 sm=0.0234 ss=0.0144) | val=0.1089 ★
...
Epoch [050/050] 135s | train=0.0456 (l1=0.0312 sm=0.0089 ss=0.0055) | val=0.0512
[INFO] Done. Best val loss: 0.0512
[INFO] Best model: models/training/best_model.pth
```

`★` marks epochs where validation loss improved and `best_model.pth` was saved.

### Approximate Training Times

| Config | Device | Batch | Epochs | Time |
|---|---|---|---|---|
| Quick test | GPU | 8 | 5 | ~5 min |
| Recommended | GPU | 4 | 50 | ~2 hrs |
| High quality | GPU | 4 | 100 | ~4 hrs |
| CPU fallback | CPU | 2 | 10 | ~30 min |

### Saved Model Locations

```
models/training/
├── best_model.pth              ← best checkpoint by val loss → used for inference
├── checkpoint_epoch0000.pth
├── checkpoint_epoch0005.pth
└── ...
```

### Test Inference After Training

```python
from core.dewarp import load_model, dewarp_document
import cv2

model   = load_model("models/training/best_model.pth", device="cuda")
image   = cv2.imread("test.jpg")
result  = dewarp_document(image, model)
cv2.imwrite("output.png", result)
```

---

## Training Fixes — v3 Changelog

### FIX 6 — Validation Augmentation Bug
**Problem:** `random_split()` creates two subsets that share the same dataset object. Setting `val.dataset.augment = False` silently disabled augmentation for **both** train and val.

**Fix:** Create two separate `Doc3DDataset` instances — one with `augment=True`, one with `augment=False` — using custom `IndexSubset` wrappers for proper index mapping.

**Impact:** Validation loss now reflects true model performance without augmentation noise.

### FIX 7 — Flat Ground Truth Images
**Problem:** Dataset only provided distorted images + backward maps. No reference flat image for reconstruction comparison.

**Fix:** Extended `Doc3DDataset.__getitem__()` to load the matching flat image from `root/flat/`. Same augmentation transforms are applied to both input and flat image to keep them in sync.

### FIX 8 — MS-SSIM Reconstruction Loss
**Problem (root cause):** Loss function only evaluated flow prediction accuracy. Model could predict a perfect flow field but still produce blurry or misaligned output — because the loss never inspected the actual output image.

**Fix:** Added a warp step inside `CombinedLoss.forward()`:
1. Apply predicted flow to warp the input image
2. Compute MS-SSIM between warped output and flat ground truth
3. `ssim_loss = (1.0 − MS-SSIM) × ssim_weight`

Added `pytorch-msssim>=0.2.1` as a dependency. Added gradient clipping at 1.0 to stabilise training.

---

## Dependencies

Core:

```
flask>=3.0
opencv-python>=4.9
numpy
torch>=2.2
torchvision
pytorch-msssim>=0.2.1
h5py
scipy
python-dotenv
```

Install all:

```bash
pip install -r requirements.txt
```

GPU (optional, replaces CPU torch):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## Troubleshooting

**Check GPU availability:**

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

**Verify all dependencies:**

```powershell
python -c "import torch, cv2, pytorch_msssim; print('All OK')"
```

**Check training data exists:**

```powershell
Test-Path "data/doc3d_subset/img"
Test-Path "data/doc3d_subset/bm"
Test-Path "data/doc3d_subset/flat"
```

**List trained checkpoints:**

```powershell
ls models/training/
```

---

## Development Roadmap

- [x] **Phase 1** — OpenCV corner detection & perspective correction
- [x] **Phase 2** — LightDewarpNet training pipeline (U-Net, Doc3D, MS-SSIM loss)
- [x] **Phase 3** — Flask REST API
- [x] **Phase 4** — Shadow removal, adaptive thresholding, frontend UI
- [ ] PDF export endpoint (`/api/export-pdf`)
- [ ] Batch processing support
- [ ] Optional authentication (PyJWT)

---

## File Extensions Reference

| Extension | Format | Location | Purpose |
|---|---|---|---|
| `.png` | Image (RGB) | `img/`, `flat/`, `outputs/` | Input distorted / ground truth flat / processed output |
| `.mat` | HDF5 (MATLAB v7.3) | `bm/` | Backward maps — displacement vectors `(2, 448, 448)` |
| `.pth` | PyTorch | `models/training/` | Trained model weights |
| `.py` | Python | `core/`, `scripts/` | Pipeline and training code |
| `.html` | HTML | `templates/` | Web UI |
| `.env` | Environment | root | Runtime configuration (git-ignored) |

---

## Citations

- **Doc3D Dataset** — Sagnik Das et al., [Intrinsic Decomposition of Document Images In-the-Wild](https://github.com/cvlab-stonybrook/doc3D-dataset)
- **DewarpNet** — [CVLab @ Stony Brook](https://github.com/cvlab-stonybrook/DewarpNet)
- **MS-SSIM** — Wang et al., 2003 — [Multi-Scale Structural Similarity](https://ece.uwaterloo.ca/~z70wang/publications/msssim.pdf)
- **pytorch-msssim** — [VainF implementation](https://github.com/VainF/pytorch-msssim)

---

## License

MIT

---

**Last Updated:** 2026-04-30
**Training Script Version:** v3 (FIX 6, 7, 8 applied)
**Status:** Production ready ✓
