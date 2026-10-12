"""Losses, metric accumulation, and LR scheduling for segmentation training."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as functional
from torch import nn


class HybridSegmentationLoss(nn.Module):
    """Dice + BCE, with optional focal BCE to improve highly imbalanced masks."""

    def __init__(self, dice_weight: float = 0.6, bce_weight: float = 0.4, positive_weight: float = 4.0, focal_gamma: float = 0.0) -> None:
        super().__init__()
        if dice_weight < 0 or bce_weight < 0 or dice_weight + bce_weight <= 0:
            raise ValueError("Dice and BCE loss weights must be non-negative and have a positive sum.")
        self.dice_weight = float(dice_weight)
        self.bce_weight = float(bce_weight)
        self.focal_gamma = float(focal_gamma)
        self.register_buffer("positive_weight", torch.tensor([positive_weight], dtype=torch.float32))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probability = torch.sigmoid(logits)
        smooth = 1.0
        intersection = (probability * target).sum(dim=(1, 2, 3))
        denominator = probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice_loss = 1.0 - ((2 * intersection + smooth) / (denominator + smooth)).mean()
        bce = functional.binary_cross_entropy_with_logits(logits, target, pos_weight=self.positive_weight, reduction="none")
        if self.focal_gamma > 0:
            p_t = probability * target + (1 - probability) * (1 - target)
            bce = ((1 - p_t) ** self.focal_gamma) * bce
        return self.dice_weight * dice_loss + self.bce_weight * bce.mean()


@dataclass
class MetricAccumulator:
    loss_total: float = 0.0
    examples: int = 0
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    true_negative: int = 0

    def update(self, logits: torch.Tensor, target: torch.Tensor, loss: torch.Tensor) -> None:
        prediction = torch.sigmoid(logits).detach() >= 0.5
        truth = target.detach() >= 0.5
        self.true_positive += int((prediction & truth).sum().item())
        self.false_positive += int((prediction & ~truth).sum().item())
        self.false_negative += int((~prediction & truth).sum().item())
        self.true_negative += int((~prediction & ~truth).sum().item())
        self.examples += int(target.shape[0])
        self.loss_total += float(loss.detach().item()) * int(target.shape[0])

    def values(self) -> dict[str, float]:
        epsilon = 1e-8
        tp, fp, fn, tn = self.true_positive, self.false_positive, self.false_negative, self.true_negative
        precision = (tp + epsilon) / (tp + fp + epsilon)
        recall = (tp + epsilon) / (tp + fn + epsilon)
        dice = (2 * tp + epsilon) / (2 * tp + fp + fn + epsilon)
        return {
            "loss": self.loss_total / max(1, self.examples),
            "dice": dice,
            "iou": (tp + epsilon) / (tp + fp + fn + epsilon),
            "precision": precision,
            "recall": recall,
            "f1": (2 * precision * recall) / (precision + recall + epsilon),
            "accuracy": (tp + tn + epsilon) / (tp + fp + fn + tn + epsilon),
        }


@dataclass
class WarmupCosinePlateau:
    """Warmup + cosine annealing with validation-Dice plateau reductions.

    A single stateful scheduler avoids fighting scheduler implementations that
    both write to the same optimizer LR. The LR is warm-started, annealed with
    cosine decay, and multiplicatively reduced whenever validation Dice plateaus.
    """

    optimizer: torch.optim.Optimizer
    base_learning_rate: float
    total_epochs: int
    warmup_epochs: int
    minimum_learning_rate: float
    plateau_patience: int
    plateau_factor: float
    best_metric: float = field(default=-float("inf"))
    plateau_wait: int = field(default=0)
    reduction_multiplier: float = field(default=1.0)

    def learning_rate_for_epoch(self, epoch: int) -> float:
        if epoch <= self.warmup_epochs:
            warmup = epoch / max(1, self.warmup_epochs)
            return max(self.minimum_learning_rate, self.base_learning_rate * warmup)
        progress = min(1.0, (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(self.minimum_learning_rate, self.base_learning_rate * self.reduction_multiplier * cosine)

    def set_epoch(self, epoch: int) -> float:
        learning_rate = self.learning_rate_for_epoch(epoch)
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
        return learning_rate

    def step(self, validation_dice: float) -> bool:
        """Update plateau state; return True when an automatic reduction happened."""
        if validation_dice > self.best_metric + 1e-6:
            self.best_metric = validation_dice
            self.plateau_wait = 0
            return False
        self.plateau_wait += 1
        if self.plateau_wait < self.plateau_patience:
            return False
        self.reduction_multiplier *= self.plateau_factor
        self.plateau_wait = 0
        return True

    def state_dict(self) -> dict:
        return {"best_metric": self.best_metric, "plateau_wait": self.plateau_wait, "reduction_multiplier": self.reduction_multiplier}

    def load_state_dict(self, state: dict) -> None:
        self.best_metric = float(state.get("best_metric", self.best_metric))
        self.plateau_wait = int(state.get("plateau_wait", self.plateau_wait))
        self.reduction_multiplier = float(state.get("reduction_multiplier", self.reduction_multiplier))
