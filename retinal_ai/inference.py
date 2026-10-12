"""Checkpoint-backed inference and explainability helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from .config import BEST_MODEL_CHECKPOINT, CLASSIFIER_CHECKPOINT, CLASSIFICATION_METRICS, SEGMENTATION_CHECKPOINT, SEGMENTATION_METRICS, TRAINING_SUMMARY
from .models import build_classifier, build_segmenter


def decode_rgb(content: bytes) -> np.ndarray:
    encoded = np.frombuffer(content, dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("The uploaded file is not a readable image.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def write_rgb(path: Path, image: np.ndarray) -> None:
    Image.fromarray(image).save(path, quality=94)


def blend_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    base = cv2.resize(image, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_AREA)
    overlay = base.copy()
    overlay[mask.astype(bool)] = np.array([0, 235, 120], dtype=np.uint8)
    return cv2.addWeighted(base, 0.62, overlay, 0.38, 0)


def blend_heatmap(image: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    base = cv2.resize(image, (heatmap.shape[1], heatmap.shape[0]), interpolation=cv2.INTER_AREA)
    colored = cv2.applyColorMap(np.uint8(np.clip(heatmap, 0, 1) * 255), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(base, 0.55, colored, 0.45, 0)


class SegmentationService:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.image_size = None
        self.threshold = 0.5
        self.measured_metrics: dict[str, Any] | None = None
        checkpoint_path = SEGMENTATION_CHECKPOINT if SEGMENTATION_CHECKPOINT.exists() else BEST_MODEL_CHECKPOINT
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.image_size = int(checkpoint["image_size"])
            if checkpoint.get("architecture") == "swin_unet":
                self.model = build_segmenter(architecture="swin_unet", **checkpoint["model_config"]).to(self.device)
            else:
                self.model = build_segmenter(int(checkpoint["base_channels"]), checkpoint.get("architecture", "unet"), pretrained=False).to(self.device)
            self.model.load_state_dict(checkpoint["model_state"])
            self.model.eval()
            if checkpoint.get("architecture") == "swin_unet" and TRAINING_SUMMARY.exists():
                self.measured_metrics = json.loads(TRAINING_SUMMARY.read_text(encoding="utf-8"))
                self.threshold = float(checkpoint.get("threshold", self.threshold))
            elif SEGMENTATION_METRICS.exists():
                self.measured_metrics = json.loads(SEGMENTATION_METRICS.read_text(encoding="utf-8"))
                self.threshold = float(self.measured_metrics.get("threshold", self.threshold))

    @property
    def available(self) -> bool:
        return self.model is not None

    @torch.inference_mode()
    def predict(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.model or not self.image_size:
            raise RuntimeError("Segmentation model unavailable. Train it with python -m retinal_ai.train_segmentation first.")
        resized = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(resized.transpose(2, 0, 1).copy()).float().unsqueeze(0).to(self.device) / 255.0
        probability = torch.sigmoid(self.model(tensor))[0, 0].cpu().numpy()
        mask = probability >= self.threshold
        return mask, probability


class ClassificationService:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.labels: list[str] = []
        self.image_size = 224
        self.measured_metrics: dict[str, Any] | None = None
        if CLASSIFIER_CHECKPOINT.exists():
            checkpoint = torch.load(CLASSIFIER_CHECKPOINT, map_location=self.device)
            self.labels = list(checkpoint["labels"])
            self.image_size = int(checkpoint["image_size"])
            self.model = build_classifier(len(self.labels), pretrained=False).to(self.device)
            self.model.load_state_dict(checkpoint["model_state"])
            self.model.eval()
            if CLASSIFICATION_METRICS.exists():
                self.measured_metrics = json.loads(CLASSIFICATION_METRICS.read_text(encoding="utf-8"))

    @property
    def available(self) -> bool:
        return self.model is not None

    def predict_with_gradcam(self, image: np.ndarray) -> tuple[str, dict[str, float], np.ndarray]:
        if not self.model:
            raise RuntimeError("Classification model unavailable. Supply disease-labelled A/D/G/N data and train it first.")
        resized = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(resized.transpose(2, 0, 1).copy()).float().unsqueeze(0).to(self.device) / 255.0
        captured: dict[str, torch.Tensor] = {}

        def forward_hook(_, __, output):
            captured["activation"] = output
            output.register_hook(lambda gradient: captured.__setitem__("gradient", gradient))

        hook = self.model.features[-1].register_forward_hook(forward_hook)
        try:
            self.model.zero_grad(set_to_none=True)
            logits = self.model(tensor)
            prediction_index = int(logits.argmax(dim=1).item())
            logits[0, prediction_index].backward()
            activation = captured["activation"][0]
            gradient = captured["gradient"][0]
            weights = gradient.mean(dim=(1, 2), keepdim=True)
            heatmap = torch.relu((weights * activation).sum(dim=0))
            heatmap = heatmap / (heatmap.max() + 1e-8)
            heatmap = torch.nn.functional.interpolate(heatmap[None, None], size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)[0, 0]
            probability = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
        finally:
            hook.remove()
        return self.labels[prediction_index], {label: float(value) for label, value in zip(self.labels, probability)}, heatmap.detach().cpu().numpy()
