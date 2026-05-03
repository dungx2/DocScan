"""
scripts/download_weights.py
---------------------------
Download pre-trained dewarping model weights.

This script handles two options:

  Option A — LightDewarpNet weights (recommended)
    Our own weights trained on Doc3D. Hosted on GitHub Releases.
    Compatible directly with core/model_arch.py LightDewarpNet.

  Option B — DewarpNet original weights (partial transfer)
    The official DewarpNet checkpoint (Stony Brook CVLab).
    Encoder layers will transfer; decoder + head are randomly initialised
    (strict=False). You should fine-tune for a few epochs after this.

Usage:
  python scripts/download_weights.py                   # Option A (default)
  python scripts/download_weights.py --source dewarpnet  # Option B
  python scripts/download_weights.py --list           # show all options
"""

import argparse
import hashlib
import logging
import sys
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ── Weight registry ───────────────────────────────────────────────────────────
# Add entries here as new checkpoints become available.

WEIGHT_REGISTRY: dict[str, dict] = {
    "lightdewarpnet-doc3d": {
        "description": "LightDewarpNet trained on Doc3D (50 epochs, val_loss=0.031)",
        "url":         "https://github.com/yourname/docscan/releases/download/v1.0/lightdewarpnet_doc3d.pth",
        "sha256":      None,   # fill in after you upload your trained weights
        "filename":    "lightdewarpnet_doc3d.pth",
        "strict":      True,
        "note":        "Our primary model. Use this after training with scripts/train.py.",
    },
    "dewarpnet": {
        "description": "DewarpNet official weights (Stony Brook CVLab)",
        "url":         "https://github.com/cvlab-stonybrook/DewarpNet/releases/download/v0.1/model_best.pth",
        "sha256":      None,
        "filename":    "dewarpnet_official.pth",
        "strict":      False,
        "note":        (
            "Partial weight transfer — encoder layers match, decoder differs. "
            "Run 5–10 fine-tuning epochs after loading."
        ),
    },
    "doctr-base": {
        "description": "DocTr base model (Fudan University)",
        "url":         "https://github.com/fh2019ustc/DocTr/releases/download/v1.0/DocTr.pth",
        "sha256":      None,
        "filename":    "doctr_base.pth",
        "strict":      False,
        "note":        (
            "DocTr uses a Transformer-based architecture. Weight transfer is minimal. "
            "Useful as a reference for architecture comparison."
        ),
    },
}

DEFAULT_SOURCE = "lightdewarpnet-doc3d"


# ── Download helpers ──────────────────────────────────────────────────────────

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _progress_hook(block_num: int, block_size: int, total_size: int):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        bar_len = 40
        filled  = int(bar_len * pct / 100)
        bar = "█" * filled + "░" * (bar_len - filled)
        mb_done  = downloaded / 1024**2
        mb_total = total_size / 1024**2
        print(
            f"\r  [{bar}] {pct:5.1f}%  {mb_done:.1f}/{mb_total:.1f} MB",
            end="", flush=True
        )
    if downloaded >= total_size:
        print()


def download(source: str, out_dir: Path, skip_verify: bool = False):
    if source not in WEIGHT_REGISTRY:
        logger.error(f"Unknown source '{source}'. Use --list to see available options.")
        sys.exit(1)

    entry = WEIGHT_REGISTRY[source]
    out_path = out_dir / entry["filename"]

    logger.info(f"\n{'─'*60}")
    logger.info(f"  Source      : {source}")
    logger.info(f"  Description : {entry['description']}")
    logger.info(f"  Destination : {out_path}")
    if entry.get("note"):
        logger.info(f"  Note        : {entry['note']}")
    logger.info(f"{'─'*60}\n")

    if out_path.exists():
        logger.info(f"File already exists: {out_path}")
        if entry["sha256"] and not skip_verify:
            _verify(out_path, entry["sha256"])
        logger.info("Skipping download.")
        _print_next_steps(out_path, entry)
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading from:\n  {entry['url']}\n")
    try:
        urlretrieve(entry["url"], out_path, reporthook=_progress_hook)
    except URLError as e:
        logger.error(f"\nDownload failed: {e}")
        logger.error(
            "The URL may have changed or require authentication.\n"
            "Please download manually and place the file at:\n"
            f"  {out_path}"
        )
        if out_path.exists():
            out_path.unlink()
        sys.exit(1)

    if entry["sha256"] and not skip_verify:
        _verify(out_path, entry["sha256"])

    logger.info(f"\nSaved to: {out_path}")
    _print_next_steps(out_path, entry)


def _verify(path: Path, expected: str):
    logger.info("Verifying checksum…")
    actual = _sha256_file(path)
    if actual != expected:
        logger.error(
            f"SHA-256 mismatch!\n"
            f"  Expected : {expected}\n"
            f"  Actual   : {actual}\n"
            "The file may be corrupted. Delete it and re-download."
        )
        sys.exit(1)
    logger.info("  ✓ Checksum OK")


def _print_next_steps(out_path: Path, entry: dict):
    logger.info("\n── Next steps ──────────────────────────────────────────")
    logger.info(f"  Set in .env:  DOCSCAN_MODEL_PATH={out_path}")
    if not entry.get("strict", True):
        logger.info(
            "  Fine-tune:    python scripts/train.py "
            f"--resume {out_path} --epochs 10 --data /path/to/doc3d"
        )
    logger.info("  Start server: python app.py")
    logger.info("────────────────────────────────────────────────────────\n")


def list_sources():
    logger.info("\nAvailable weight sources:\n")
    for key, entry in WEIGHT_REGISTRY.items():
        marker = " (default)" if key == DEFAULT_SOURCE else ""
        logger.info(f"  {key}{marker}")
        logger.info(f"    {entry['description']}")
        logger.info(f"    strict={entry['strict']}")
        if entry.get("note"):
            logger.info(f"    {entry['note']}")
        logger.info("")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Download pre-trained dewarping weights")
    p.add_argument(
        "--source", default=DEFAULT_SOURCE,
        help=f"Weight source key (default: {DEFAULT_SOURCE})"
    )
    p.add_argument(
        "--out-dir", default="models",
        help="Directory to save weights (default: models/)"
    )
    p.add_argument(
        "--list", action="store_true",
        help="List all available weight sources and exit"
    )
    p.add_argument(
        "--skip-verify", action="store_true",
        help="Skip SHA-256 checksum verification"
    )
    args = p.parse_args()

    if args.list:
        list_sources()
        return

    download(
        source=args.source,
        out_dir=Path(args.out_dir),
        skip_verify=args.skip_verify,
    )


if __name__ == "__main__":
    main()
