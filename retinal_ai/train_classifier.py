"""Fine-tune EfficientNet-B0 when disease-labelled fundus images are supplied."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .config import CLASSIFICATION_DIR, CLASSIFIER_CHECKPOINT, ensure_directories
from .data import ClassificationDataset, discover_classification_samples
from .models import build_classifier


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.inference_mode()
def accuracy(model, loader, device):
    correct = total = 0
    model.eval()
    for image, target, _ in loader:
        prediction = model(image.to(device)).argmax(dim=1).cpu()
        correct += int((prediction == target).sum())
        total += target.numel()
    return correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=CLASSIFICATION_DIR)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", action="store_true", help="Use/download ImageNet weights via torchvision.")
    args = parser.parse_args()
    ensure_directories(); seed_everything(args.seed)
    if not args.data_dir.exists():
        raise FileNotFoundError(f"No classification dataset at {args.data_dir}. See README for required labels/layout.")
    data, labels = discover_classification_samples(args.data_dir)
    if "all" in data:
        samples = data["all"]
        paths, y = zip(*samples)
        train_paths, temp_paths, train_y, temp_y = train_test_split(paths, y, test_size=0.30, stratify=y, random_state=args.seed)
        val_paths, test_paths, val_y, test_y = train_test_split(temp_paths, temp_y, test_size=0.50, stratify=temp_y, random_state=args.seed)
        splits = {"train": list(zip(train_paths, train_y)), "val": list(zip(val_paths, val_y)), "test": list(zip(test_paths, test_y))}
    else:
        splits = data
    train_loader = DataLoader(ClassificationDataset(splits["train"], labels, args.image_size, augment=True), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(ClassificationDataset(splits["val"], labels, args.image_size), batch_size=args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_classifier(len(labels), args.pretrained).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    best_accuracy = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for image, target, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(image.to(device)), target.to(device))
            loss.backward(); optimizer.step()
        val_accuracy = accuracy(model, val_loader, device)
        print(f"epoch={epoch:03d} val_accuracy={val_accuracy:.4f}")
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save({"model_state": model.state_dict(), "model": "EfficientNet-B0", "labels": labels, "image_size": args.image_size, "seed": args.seed, "split_sizes": {key: len(value) for key, value in splits.items()}}, CLASSIFIER_CHECKPOINT)
    (CLASSIFIER_CHECKPOINT.parent / "classification_split.json").write_text(json.dumps({key: [str(path) for path, _ in value] for key, value in splits.items()}, indent=2), encoding="utf-8")
    print(f"Saved checkpoint: {CLASSIFIER_CHECKPOINT}")
    print("Next run: python -m retinal_ai.evaluate_classifier")


if __name__ == "__main__":
    main()

