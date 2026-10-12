"""Regenerate visual QA artifacts from the saved, evaluated segmentation checkpoint."""
from __future__ import annotations

import argparse
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import DRIVE_DIR, REPORTS_DIR, SEGMENTATION_CHECKPOINT, SEGMENTATION_METRICS, ensure_directories
from .data import DriveDataset, find_drive_pairs, split_items
from .models import build_segmenter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    ensure_directories()
    if not SEGMENTATION_CHECKPOINT.exists() or not SEGMENTATION_METRICS.exists():
        raise FileNotFoundError("Train and evaluate segmentation before generating the report.")
    checkpoint = torch.load(SEGMENTATION_CHECKPOINT, map_location="cpu")
    report = json.loads(SEGMENTATION_METRICS.read_text(encoding="utf-8"))
    _, _, test_pairs = split_items(find_drive_pairs(DRIVE_DIR), args.seed)
    dataset = DriveDataset(test_pairs, int(checkpoint["image_size"]))
    model = build_segmenter(int(checkpoint["base_channels"]), checkpoint.get("architecture", "unet"), pretrained=False)
    model.load_state_dict(checkpoint["model_state"]); model.eval()
    figure, axes = plt.subplots(len(dataset), 3, figsize=(9, 3 * len(dataset)))
    threshold = float(report["threshold"])
    with torch.inference_mode():
        for row, (image, truth, field_mask, identifier) in zip(np.atleast_2d(axes), DataLoader(dataset, batch_size=1)):
            probability = torch.sigmoid(model(image))[0, 0].numpy()
            valid = field_mask[0, 0].numpy() > 0.5
            row[0].imshow(image[0].permute(1, 2, 0)); row[0].set_title(f"Fundus {identifier[0]}")
            row[1].imshow(np.ma.masked_where(~valid, truth[0, 0]), cmap="gray", vmin=0, vmax=1); row[1].set_title("Manual annotation (FOV)")
            row[2].imshow(np.ma.masked_where(~valid, probability >= threshold), cmap="gray", vmin=0, vmax=1); row[2].set_title("Prediction (FOV)")
            for axis in row: axis.axis("off")
    figure.suptitle("Held-out DRIVE internal test split (research evaluation)", y=1.01)
    figure.tight_layout(); figure.savefig(REPORTS_DIR / "segmentation_qualitative_test_samples.png", dpi=160, bbox_inches="tight"); plt.close(figure)
    history_path = REPORTS_DIR / "segmentation_training_history.csv"
    with history_path.open(encoding="utf-8") as handle:
        history = list(csv.DictReader(handle))
    if history:
        epoch = [int(row["epoch"]) for row in history]
        loss = [float(row["train_loss"]) for row in history]
        val_dice = [float(row["val_dice_at_0.5"]) for row in history]
        figure, left = plt.subplots(figsize=(8, 4.5)); right = left.twinx()
        left.plot(epoch, loss, color="#bc5090", label="Training loss")
        right.plot(epoch, val_dice, color="#003f5c", label="Validation Dice")
        left.set(xlabel="Epoch", ylabel="Loss", title="Segmentation training history"); right.set_ylabel("Validation Dice")
        figure.tight_layout(); figure.savefig(REPORTS_DIR / "segmentation_training_curve.png", dpi=160); plt.close(figure)
    print(f"Wrote visual evaluation artifacts to {REPORTS_DIR}")


if __name__ == "__main__":
    main()
