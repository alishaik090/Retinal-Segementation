"""Train and evaluate the DRIVE vessel segmenter without test-set leakage."""
from __future__ import annotations

import argparse
import csv
import json
import random
from datetime import datetime, timezone

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from .config import ARTIFACTS_DIR, DRIVE_DIR, SEGMENTATION_CHECKPOINT, SEGMENTATION_METRICS, ensure_directories
from .data import DriveDataset, find_drive_pairs, split_items
from .metrics import binary_metrics, mean_metrics
from .models import build_segmenter


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class DiceBCELoss(nn.Module):
    def __init__(self, positive_weight: float = 3.0) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([positive_weight]))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, target)
        probability = torch.sigmoid(logits)
        smooth = 1.0
        dice = (2.0 * (probability * target).sum((1, 2, 3)) + smooth) / (probability.sum((1, 2, 3)) + target.sum((1, 2, 3)) + smooth)
        return bce + (1.0 - dice.mean())


@torch.inference_mode()
def predict_records(model, loader, device):
    records = []
    model.eval()
    for image, mask, identifiers in loader:
        probabilities = torch.sigmoid(model(image.to(device))).cpu().numpy()[:, 0]
        for probability, truth, identifier in zip(probabilities, mask.numpy()[:, 0], identifiers):
            records.append((str(identifier), probability, truth))
    return records


def score_records(records, threshold: float):
    rows = []
    for identifier, probability, truth in records:
        row = binary_metrics(truth, probability >= threshold)
        row["id"] = identifier
        rows.append(row)
    return mean_metrics([{k: v for k, v in row.items() if k != "id"} for row in rows]), rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--architecture", choices=["unet", "resnet18_unet"], default="unet")
    parser.add_argument("--pretrained-encoder", action="store_true", help="Use official cached/downloaded ImageNet weights for resnet18_unet.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--threads", type=int, default=2, help="CPU worker threads; small models are faster with a modest setting.")
    args = parser.parse_args()

    ensure_directories()
    torch.hub.set_dir(str(ARTIFACTS_DIR / "hub"))
    seed_everything(args.seed)
    torch.set_num_threads(max(1, args.threads))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pairs = find_drive_pairs(DRIVE_DIR)
    train_pairs, val_pairs, test_pairs = split_items(pairs, args.seed)
    train_loader = DataLoader(DriveDataset(train_pairs, args.image_size, augment=True), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(DriveDataset(val_pairs, args.image_size), batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(DriveDataset(test_pairs, args.image_size), batch_size=1, shuffle=False, num_workers=0)

    model = build_segmenter(args.base_channels, args.architecture, args.pretrained_encoder).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=8)
    criterion = DiceBCELoss().to(device)
    best_dice, stale_epochs, history = -1.0, 0, []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, count = 0.0, 0
        for image, mask, _ in train_loader:
            image, mask = image.to(device), mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(image), mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item() * image.size(0)
            count += image.size(0)
        val_records = predict_records(model, val_loader, device)
        val_metrics, _ = score_records(val_records, threshold=0.5)
        scheduler.step(val_metrics["dice"])
        history.append({"epoch": epoch, "train_loss": running_loss / max(count, 1), "val_dice_at_0.5": val_metrics["dice"], "learning_rate": optimizer.param_groups[0]["lr"]})
        print(f"epoch={epoch:03d} train_loss={history[-1]['train_loss']:.4f} val_dice={val_metrics['dice']:.4f}")
        if val_metrics["dice"] > best_dice:
            best_dice, stale_epochs = val_metrics["dice"], 0
            torch.save({"model_state": model.state_dict(), "model": "VesselUNet" if args.architecture == "unet" else "ResNet18UNet", "architecture": args.architecture, "pretrained_encoder": args.pretrained_encoder, "base_channels": args.base_channels, "image_size": args.image_size, "seed": args.seed}, SEGMENTATION_CHECKPOINT)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping after {epoch} epochs.")
                break

    checkpoint = torch.load(SEGMENTATION_CHECKPOINT, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    validation_records = predict_records(model, val_loader, device)
    candidate_thresholds = [round(value, 2) for value in np.arange(0.20, 0.81, 0.05)]
    threshold = max(candidate_thresholds, key=lambda value: score_records(validation_records, value)[0]["dice"])
    test_records = predict_records(model, test_loader, device)
    test_metrics, per_image = score_records(test_records, threshold)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": "DRIVE retinal vessel segmentation",
        "evaluation_protocol": "Deterministic 60/20/20 split of the 20 manually labelled DRIVE training images. Threshold selected only on validation.",
        "warning": "This is a small internal split, not the hidden official DRIVE test benchmark.",
        "model": {"name": checkpoint["model"], "architecture": checkpoint["architecture"], "pretrained_encoder": checkpoint["pretrained_encoder"], "base_channels": args.base_channels, "image_size": args.image_size},
        "seed": args.seed,
        "threshold": threshold,
        "splits": {"train": [p.identifier for p in train_pairs], "validation": [p.identifier for p in val_pairs], "test": [p.identifier for p in test_pairs]},
        "validation_best_dice_at_0_5": best_dice,
        "test_metrics": test_metrics,
        "per_test_image": per_image,
    }
    SEGMENTATION_METRICS.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (SEGMENTATION_METRICS.parent / "segmentation_training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    print("\nMeasured held-out segmentation metrics:")
    print(json.dumps(test_metrics, indent=2))
    print(f"Saved checkpoint: {SEGMENTATION_CHECKPOINT}")
    print(f"Saved report: {SEGMENTATION_METRICS}")


if __name__ == "__main__":
    main()
