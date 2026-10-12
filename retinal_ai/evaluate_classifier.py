"""Evaluate the saved classifier only on its held-out test partition."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, log_loss, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader

from .config import CLASSIFIER_CHECKPOINT, CLASSIFICATION_METRICS, MODELS_DIR, REPORTS_DIR, ensure_directories
from .data import ClassificationDataset
from .models import build_classifier


def main():
    ensure_directories()
    if not CLASSIFIER_CHECKPOINT.exists():
        raise FileNotFoundError("Classifier checkpoint is missing. Run train_classifier first.")
    split_file = MODELS_DIR / "classification_split.json"
    if not split_file.exists():
        raise FileNotFoundError("classification_split.json is missing. Re-run train_classifier to create the auditable split manifest.")
    checkpoint = torch.load(CLASSIFIER_CHECKPOINT, map_location="cpu")
    labels = list(checkpoint["labels"])
    split_paths = json.loads(split_file.read_text(encoding="utf-8"))
    test_samples = [(path, next(label for label in labels if Path(path).parent.name == label)) for path in split_paths["test"]]
    loader = DataLoader(ClassificationDataset(test_samples, labels, int(checkpoint["image_size"])), batch_size=16)
    model = build_classifier(len(labels))
    model.load_state_dict(checkpoint["model_state"]); model.eval()
    y_true, y_pred, probabilities = [], [], []
    with torch.inference_mode():
        for image, target, _ in loader:
            output = torch.softmax(model(image), dim=1).numpy()
            probabilities.extend(output); y_true.extend(target.numpy()); y_pred.extend(output.argmax(axis=1))
    probabilities = np.asarray(probabilities)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=list(range(len(labels))), zero_division=0)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": "Four-class fundus disease classification",
        "evaluation_protocol": "Held-out test split created before training.",
        "labels": labels,
        "test_metrics": {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
            "log_loss": float(log_loss(y_true, probabilities, labels=list(range(len(labels))))),
            "roc_auc_ovr_macro": float(roc_auc_score(y_true, probabilities, multi_class="ovr", average="macro")),
        },
        "per_class": {
            label: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i])}
            for i, label in enumerate(labels)
        },
        "classification_report": classification_report(y_true, y_pred, target_names=labels, output_dict=True, zero_division=0),
    }
    CLASSIFICATION_METRICS.write_text(json.dumps(report, indent=2), encoding="utf-8")
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis); axis.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels, yticklabels=labels, xlabel="Predicted", ylabel="True", title="Classification confusion matrix")
    for row in range(len(labels)):
        for col in range(len(labels)):
            axis.text(col, row, str(matrix[row, col]), ha="center", va="center")
    figure.tight_layout(); figure.savefig(REPORTS_DIR / "classification_confusion_matrix.png", dpi=160); plt.close(figure)
    print(json.dumps(report["test_metrics"], indent=2))


if __name__ == "__main__":
    main()
