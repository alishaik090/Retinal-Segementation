"""Predict vessel masks for unseen images and write original/GT/prediction/overlay comparisons."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from .config import ARTIFACTS_DIR, BEST_MODEL_CHECKPOINT
from .inference import blend_mask
from .models import build_segmenter


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if checkpoint.get("architecture") != "swin_unet":
        raise ValueError("predict_segmentation expects a Swin-UNet best_model.pth checkpoint.")
    model = build_segmenter(architecture="swin_unet", **checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"]); model.eval()
    return model, int(checkpoint["image_size"]), float(checkpoint.get("threshold", 0.5))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Unseen retinal input image.")
    parser.add_argument("--ground-truth", type=Path, default=None, help="Optional matching vessel ground-truth mask.")
    parser.add_argument("--checkpoint", type=Path, default=BEST_MODEL_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS_DIR / "predictions")
    args = parser.parse_args()
    if not args.image.exists() or not args.checkpoint.exists():
        raise FileNotFoundError("Input image or best checkpoint does not exist.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, image_size, threshold = load_model(args.checkpoint, device)
    bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Unable to read {args.image}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized.transpose(2, 0, 1).copy()).float().unsqueeze(0).to(device) / 255.0
    with torch.inference_mode():
        probability = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
    prediction = probability >= threshold
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.image.stem
    Image.fromarray((prediction.astype(np.uint8) * 255)).save(args.output_dir / f"{stem}_prediction.png")
    Image.fromarray(blend_mask(rgb, prediction)).save(args.output_dir / f"{stem}_overlay.jpg", quality=94)
    ground_truth = None
    if args.ground_truth:
        ground_truth = np.asarray(Image.open(args.ground_truth).convert("L")) > 127
        ground_truth = cv2.resize(ground_truth.astype(np.uint8), (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    figure, axes = plt.subplots(1, 4, figsize=(14, 4))
    axes[0].imshow(resized); axes[0].set_title("Original")
    axes[1].imshow(ground_truth if ground_truth is not None else np.zeros_like(prediction), cmap="gray", vmin=0, vmax=1); axes[1].set_title("Ground Truth" if ground_truth is not None else "Ground Truth (not supplied)")
    axes[2].imshow(prediction, cmap="gray", vmin=0, vmax=1); axes[2].set_title("Prediction")
    axes[3].imshow(blend_mask(resized, prediction)); axes[3].set_title("Overlay")
    for axis in axes: axis.axis("off")
    figure.tight_layout(); figure.savefig(args.output_dir / f"{stem}_comparison.png", dpi=180); plt.close(figure)
    print(f"Saved prediction artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
