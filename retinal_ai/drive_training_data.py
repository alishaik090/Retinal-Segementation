"""DRIVE discovery, reproducible splitting, and segmentation augmentation."""
from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .config import DRIVE_DIR
from .data import DrivePair, find_drive_pairs


def resolve_drive_root(data_root: str | Path | None = None) -> Path:
    """Find DRIVE in local or common Google Colab locations."""
    candidates: list[Path] = []
    if data_root:
        requested = Path(data_root).expanduser()
        candidates.extend((requested, requested / "DRIVE"))
    candidates.extend(
        (
            DRIVE_DIR,
            Path("/content/DRIVE"),
            Path("/content/drive/MyDrive/DRIVE"),
            Path("/content/drive/MyDrive/datasets/DRIVE"),
            Path.cwd() / "DRIVE",
        )
    )
    for candidate in candidates:
        if (candidate / "training" / "images").is_dir() and (candidate / "training" / "1st_manual").is_dir():
            return candidate.resolve()
    searched = "\n - ".join(str(item) for item in candidates)
    raise FileNotFoundError(
        "Unable to locate a labelled DRIVE dataset. Expected training/images and "
        f"training/1st_manual. Checked:\n - {searched}"
    )


def split_drive_pairs(pairs: Sequence[DrivePair], validation_fraction: float, test_fraction: float, seed: int) -> tuple[list[DrivePair], list[DrivePair], list[DrivePair]]:
    """Return reproducible non-overlapping train/validation/(optional) test pairs."""
    if not 0.05 <= validation_fraction < 0.5:
        raise ValueError("validation_fraction must be in [0.05, 0.5).")
    if not 0 <= test_fraction < 0.5 or validation_fraction + test_fraction >= 0.9:
        raise ValueError("test_fraction must leave a non-empty training partition.")
    order = list(pairs)
    random.Random(seed).shuffle(order)
    val_count = max(1, round(len(order) * validation_fraction))
    test_count = round(len(order) * test_fraction)
    if len(order) - val_count - test_count < 2:
        raise ValueError("Not enough samples remain for training after requested split.")
    validation = order[:val_count]
    test = order[val_count : val_count + test_count]
    train = order[val_count + test_count :]
    return train, validation, test


@dataclass(frozen=True)
class AugmentationConfig:
    horizontal_flip_probability: float = 0.5
    vertical_flip_probability: float = 0.25
    rotation_degrees: float = 20.0
    brightness_contrast_probability: float = 0.5
    clahe_probability: float = 0.35
    crop_probability: float = 0.5
    crop_scale_min: float = 0.82


class DriveAugmenter:
    """Paired image/mask augmentation using OpenCV interpolation appropriate to each."""

    def __init__(self, config: AugmentationConfig) -> None:
        self.config = config

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if random.random() < self.config.horizontal_flip_probability:
            image, mask = np.fliplr(image).copy(), np.fliplr(mask).copy()
        if random.random() < self.config.vertical_flip_probability:
            image, mask = np.flipud(image).copy(), np.flipud(mask).copy()
        if self.config.rotation_degrees and random.random() < 0.75:
            angle = random.uniform(-self.config.rotation_degrees, self.config.rotation_degrees)
            center = (image.shape[1] / 2, image.shape[0] / 2)
            transform = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, transform, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            mask = cv2.warpAffine(mask, transform, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if random.random() < self.config.crop_probability:
            height, width = image.shape[:2]
            scale = random.uniform(self.config.crop_scale_min, 1.0)
            crop_h, crop_w = max(16, int(height * scale)), max(16, int(width * scale))
            top = random.randint(0, height - crop_h)
            left = random.randint(0, width - crop_w)
            image = image[top : top + crop_h, left : left + crop_w]
            mask = mask[top : top + crop_h, left : left + crop_w]
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        if random.random() < self.config.brightness_contrast_probability:
            alpha = random.uniform(0.82, 1.18)
            beta = random.uniform(-20.0, 20.0)
            image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        if random.random() < self.config.clahe_probability:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return image, mask


class DriveTrainingDataset(Dataset):
    """Dataset used by the Swin-UNet training, validation, and prediction scripts."""

    def __init__(self, pairs: Sequence[DrivePair], image_size: int, augment: bool = False, samples_per_image: int = 1) -> None:
        self.pairs = list(pairs)
        self.image_size = int(image_size)
        self.augmenter = DriveAugmenter(AugmentationConfig()) if augment else None
        self.samples_per_image = max(1, int(samples_per_image)) if augment else 1

    def __len__(self) -> int:
        return len(self.pairs) * self.samples_per_image

    def __getitem__(self, index: int):
        pair = self.pairs[index % len(self.pairs)]
        bgr = cv2.imread(str(pair.image), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Unreadable DRIVE image: {pair.image}")
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mask = np.asarray(Image.open(pair.mask).convert("L"))
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        if self.augmenter:
            image, mask = self.augmenter(image, mask)
        image_tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float() / 255.0
        mask_tensor = torch.from_numpy(np.ascontiguousarray((mask > 127).astype(np.float32)[None, ...]))
        return image_tensor, mask_tensor, pair.identifier


def discover_and_split_drive(data_root: str | Path | None, validation_fraction: float, test_fraction: float, seed: int):
    root = resolve_drive_root(data_root)
    pairs = find_drive_pairs(root)
    return root, split_drive_pairs(pairs, validation_fraction, test_fraction, seed)


def seed_worker(worker_id: int) -> None:
    """Seed NumPy/OpenCV-style randomness per DataLoader worker."""
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed + worker_id)
    np.random.seed(worker_seed + worker_id)
