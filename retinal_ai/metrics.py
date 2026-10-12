from __future__ import annotations

import numpy as np


def binary_metrics(target: np.ndarray, prediction: np.ndarray, valid_mask: np.ndarray | None = None) -> dict[str, float]:
    """Pixel-level binary metrics for pre-thresholded masks."""
    target = target.astype(bool).ravel()
    prediction = prediction.astype(bool).ravel()
    if valid_mask is not None:
        valid = valid_mask.astype(bool).ravel()
        target = target[valid]
        prediction = prediction[valid]
    tp = int(np.logical_and(target, prediction).sum())
    fp = int(np.logical_and(~target, prediction).sum())
    fn = int(np.logical_and(target, ~prediction).sum())
    tn = int(np.logical_and(~target, ~prediction).sum())
    eps = 1e-8
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    specificity = (tn + eps) / (tn + fp + eps)
    accuracy = (tp + tn + eps) / (tp + tn + fp + fn + eps)
    return {"dice": float(dice), "iou": float(iou), "precision": float(precision), "recall": float(recall), "specificity": float(specificity), "pixel_accuracy": float(accuracy)}


def mean_metrics(metric_rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([row[key] for row in metric_rows])) for key in metric_rows[0]}

