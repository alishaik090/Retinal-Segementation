"""Dataset discovery and preprocessing with deterministic, auditable splits."""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass(frozen=True)
class DrivePair:
    image: Path
    mask: Path
    identifier: str


def find_drive_pairs(drive_dir: Path) -> list[DrivePair]:
    image_dir = drive_dir / "training" / "images"
    mask_dir = drive_dir / "training" / "1st_manual"
    if not image_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(
            "Expected DRIVE training/images and training/1st_manual. "
            "Run the archive extraction step first."
        )
    pairs: list[DrivePair] = []
    for image in sorted(image_dir.glob("*_training.*")):
        case_id = image.stem.split("_")[0]
        candidates = sorted(mask_dir.glob(f"{case_id}_manual1.*"))
        if not candidates:
            raise FileNotFoundError(f"No first manual annotation for {image.name}")
        pairs.append(DrivePair(image, candidates[0], case_id))
    if len(pairs) < 5:
        raise ValueError(f"Expected at least five labelled DRIVE pairs, found {len(pairs)}")
    return pairs


def split_items(items: Sequence, seed: int, train_ratio: float = 0.6, val_ratio: float = 0.2):
    if not 0 < train_ratio < 1 or not 0 < val_ratio < 1 or train_ratio + val_ratio >= 1:
        raise ValueError("Ratios must be positive and leave a test partition")
    indices = list(range(len(items)))
    random.Random(seed).shuffle(indices)
    n_train = max(1, int(round(len(items) * train_ratio)))
    n_val = max(1, int(round(len(items) * val_ratio)))
    n_train = min(n_train, len(items) - 2)
    n_val = min(n_val, len(items) - n_train - 1)
    return ([items[i] for i in indices[:n_train]], [items[i] for i in indices[n_train:n_train+n_val]], [items[i] for i in indices[n_train+n_val:]])


class DriveDataset(Dataset):
    def __init__(self, pairs: Sequence[DrivePair], image_size: int, augment: bool = False) -> None:
        self.pairs = list(pairs)
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        pair = self.pairs[index]
        image = cv2.imread(str(pair.image), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unable to read image: {pair.image}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = np.asarray(Image.open(pair.mask).convert("L"))
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.float32)
        if self.augment:
            if random.random() < 0.5:
                image, mask = np.fliplr(image).copy(), np.fliplr(mask).copy()
            if random.random() < 0.25:
                image, mask = np.flipud(image).copy(), np.flipud(mask).copy()
            if random.random() < 0.35:
                alpha = random.uniform(0.85, 1.15)
                beta = random.uniform(-10, 10)
                image = np.clip(image * alpha + beta, 0, 255).astype(np.uint8)
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1).copy()).float() / 255.0
        mask_tensor = torch.from_numpy(mask[None, ...].copy())
        return image_tensor, mask_tensor, pair.identifier


def discover_classification_samples(root: Path):
    labels = ["A", "D", "G", "N"]
    split_dirs = [name for name in ("train", "val", "test") if (root / name).is_dir()]
    if split_dirs and set(split_dirs) != {"train", "val", "test"}:
        raise ValueError("Classification data must have all of train, val, test, or none of them.")
    base_splits = split_dirs or [""]
    data = {split: [] for split in (split_dirs or ["all"])}
    for split in base_splits:
        base = root / split
        for label in labels:
            class_dir = base / label
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Missing class directory: {class_dir}")
            files = [p for p in sorted(class_dir.rglob("*")) if p.suffix.lower() in IMAGE_EXTENSIONS]
            data[split or "all"].extend((path, label) for path in files)
    if any(not any(label == wanted for _, label in samples) for wanted in labels for samples in data.values()):
        raise ValueError("Every supplied classification split must contain A, D, G, and N samples.")
    return data, labels


class ClassificationDataset(Dataset):
    def __init__(self, samples, labels: list[str], image_size: int, augment: bool = False) -> None:
        self.samples = list(samples)
        self.labels = labels
        self.image_size = image_size
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB").resize((self.image_size, self.image_size))
        image = np.asarray(image)
        if self.augment and random.random() < 0.5:
            image = np.fliplr(image).copy()
        tensor = torch.from_numpy(image.transpose(2, 0, 1).copy()).float() / 255.0
        return tensor, self.labels.index(label), str(path)

