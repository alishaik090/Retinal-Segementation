"""Production-style Swin-UNet training entry point for DRIVE vessel segmentation.

Examples
--------
Local::

    python -m retinal_ai.train_swin_unet --data-root data/raw/DRIVE --epochs 120

Google Colab::

    python -m retinal_ai.train_swin_unet --data-root /content/DRIVE --epochs 120 --workers 2
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .config import ARTIFACTS_DIR, BEST_MODEL_CHECKPOINT, LAST_MODEL_CHECKPOINT, PLOTS_DIR, TRAINING_METRICS, TRAINING_REPORT, TRAINING_SUMMARY, ensure_directories
from .drive_training_data import DriveTrainingDataset, discover_and_split_drive, seed_worker
from .models import build_segmenter
from .reporting import generate_training_report
from .training_utils import HybridSegmentationLoss, MetricAccumulator, WarmupCosinePlateau


METRIC_FIELDS = [
    "epoch", "learning_rate", "train_loss", "train_dice", "train_iou", "train_precision", "train_recall", "train_f1", "train_accuracy",
    "val_loss", "val_dice", "val_iou", "val_precision", "val_recall", "val_f1", "val_accuracy", "epoch_seconds",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def parse_integer_tuple(values: list[str], expected: int, name: str) -> tuple[int, ...]:
    parsed = tuple(int(value) for value in values)
    if len(parsed) != expected:
        raise argparse.ArgumentTypeError(f"{name} requires exactly {expected} integers.")
    return parsed


def run_epoch(model, loader, loss_function, device, optimizer=None, scaler=None, amp_enabled: bool = False, gradient_clip: float = 1.0) -> dict[str, float]:
    is_training = optimizer is not None
    model.train(is_training)
    metrics = MetricAccumulator()
    for image, mask, _ in loader:
        image, mask = image.to(device, non_blocking=True), mask.to(device, non_blocking=True)
        if is_training:
            optimizer.zero_grad(set_to_none=True)
        context = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True) if amp_enabled else nullcontext()
        with context:
            logits = model(image)
            loss = loss_function(logits, mask)
        if is_training:
            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
        metrics.update(logits, mask, loss)
    return metrics.values()


@torch.inference_mode()
def evaluate(model, loader, loss_function, device, amp_enabled: bool) -> dict[str, float]:
    model.eval()
    metrics = MetricAccumulator()
    for image, mask, _ in loader:
        image, mask = image.to(device, non_blocking=True), mask.to(device, non_blocking=True)
        context = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True) if amp_enabled else nullcontext()
        with context:
            logits = model(image)
            loss = loss_function(logits, mask)
        metrics.update(logits, mask, loss)
    return metrics.values()


def write_metrics(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_metrics(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [{key: float(value) if key != "epoch" else int(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def save_curves(rows: list[dict[str, float]], plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    epochs = [int(row["epoch"]) for row in rows]
    for metric in ("dice", "iou", "loss", "precision", "recall"):
        figure, axis = plt.subplots(figsize=(8, 4.5))
        axis.plot(epochs, [row[f"train_{metric}"] for row in rows], label=f"Training {metric.title()}", color="#0B6477", linewidth=2)
        axis.plot(epochs, [row[f"val_{metric}"] for row in rows], label=f"Validation {metric.title()}", color="#D35400", linewidth=2)
        axis.set(xlabel="Epoch", ylabel=metric.title(), title=f"{metric.title()} curve")
        axis.grid(alpha=0.25); axis.legend(); figure.tight_layout()
        figure.savefig(plots_dir / f"{metric}_curve.png", dpi=170)
        plt.close(figure)


def checkpoint_payload(model, optimizer, scaler, scheduler, epoch: int, best_epoch: int, best_metrics: dict[str, float], args, model_config, split_ids, elapsed_seconds: float) -> dict:
    return {
        "format_version": 2,
        "architecture": "swin_unet",
        "model_config": model_config,
        "image_size": args.image_size,
        "threshold": 0.5,
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_validation_metrics": best_metrics,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "seed": args.seed,
        "splits": split_ids,
        "elapsed_seconds": elapsed_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-root", default=None, help="DRIVE root. Optional: local and common Colab paths are auto-discovered.")
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--embed-dim", type=int, default=24)
    parser.add_argument("--depths", nargs="+", default=["1", "1", "2", "1"])
    parser.add_argument("--num-heads", nargs="+", default=["3", "6", "12", "24"])
    parser.add_argument("--window-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--minimum-learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--plateau-patience", type=int, default=12)
    parser.add_argument("--plateau-factor", type=float, default=0.5)
    parser.add_argument("--early-stopping-patience", type=int, default=30)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--positive-weight", type=float, default=4.0)
    parser.add_argument("--dice-weight", type=float, default=0.6)
    parser.add_argument("--bce-weight", type=float, default=0.4)
    parser.add_argument("--focal-gamma", type=float, default=0.0, help="0 uses Dice+BCE; >0 uses Dice+focal BCE.")
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--test-fraction", type=float, default=0.0, help="Optional internal labelled test fraction; it is never used for early stopping.")
    parser.add_argument("--train-samples-per-image", type=int, default=3, help="Augmented random views per labelled image per epoch.")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", nargs="?", const="last_model.pth", default=None, help="Resume a checkpoint; no value resumes artifacts/last_model.pth.")
    args = parser.parse_args()
    args.depths = parse_integer_tuple(args.depths, 4, "--depths")
    args.num_heads = parse_integer_tuple(args.num_heads, 4, "--num-heads")
    if args.image_size < 64 or args.image_size % 4:
        parser.error("--image-size must be at least 64 and divisible by 4.")
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive.")

    set_seed(args.seed)
    ensure_directories()
    artifacts_dir = args.artifacts_dir.resolve(); artifacts_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = artifacts_dir / "plots"; plots_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = artifacts_dir / "metrics.csv"
    best_path = artifacts_dir / "best_model.pth"
    last_path = artifacts_dir / "last_model.pth"
    summary_path = artifacts_dir / "training_summary.json"
    report_path = artifacts_dir / "training_report.pdf"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(args.amp and device.type == "cuda")
    if device.type == "cpu":
        torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

    data_root, (train_pairs, validation_pairs, test_pairs) = discover_and_split_drive(args.data_root, args.validation_fraction, args.test_fraction, args.seed)
    split_ids = {"train": [pair.identifier for pair in train_pairs], "validation": [pair.identifier for pair in validation_pairs], "test": [pair.identifier for pair in test_pairs]}
    (artifacts_dir / "data_splits.json").write_text(json.dumps({"data_root": str(data_root), "seed": args.seed, **split_ids}, indent=2), encoding="utf-8")
    data_generator = torch.Generator().manual_seed(args.seed)
    common_loader = {"num_workers": max(0, args.workers), "pin_memory": device.type == "cuda", "worker_init_fn": seed_worker, "generator": data_generator}
    if args.workers > 0:
        common_loader["persistent_workers"] = True
    train_loader = DataLoader(DriveTrainingDataset(train_pairs, args.image_size, augment=True, samples_per_image=args.train_samples_per_image), batch_size=args.batch_size, shuffle=True, **common_loader)
    validation_loader = DataLoader(DriveTrainingDataset(validation_pairs, args.image_size), batch_size=args.batch_size, shuffle=False, **common_loader)
    test_loader = DataLoader(DriveTrainingDataset(test_pairs, args.image_size), batch_size=args.batch_size, shuffle=False, **common_loader) if test_pairs else None

    model_config = {"in_channels": 3, "embed_dim": args.embed_dim, "depths": list(args.depths), "num_heads": list(args.num_heads), "window_size": args.window_size, "patch_size": 4}
    model = build_segmenter(architecture="swin_unet", **model_config).to(device)
    loss_function = HybridSegmentationLoss(dice_weight=args.dice_weight, bce_weight=args.bce_weight, positive_weight=args.positive_weight, focal_gamma=args.focal_gamma).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    scheduler = WarmupCosinePlateau(optimizer, args.learning_rate, args.epochs, args.warmup_epochs, args.minimum_learning_rate, args.plateau_patience, args.plateau_factor)
    start_epoch, best_epoch, best_metrics, stale_epochs, metrics_rows, prior_elapsed = 1, 0, {}, 0, [], 0.0

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_absolute():
            resume_path = artifacts_dir / resume_path
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        if checkpoint.get("architecture") != "swin_unet":
            raise ValueError("Resume checkpoint is not a Swin-UNet checkpoint.")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scaler.load_state_dict(checkpoint.get("scaler_state", {}))
        scheduler.load_state_dict(checkpoint.get("scheduler_state", {}))
        start_epoch = int(checkpoint["epoch"]) + 1
        best_epoch = int(checkpoint.get("best_epoch", 0))
        best_metrics = dict(checkpoint.get("best_validation_metrics", {}))
        prior_elapsed = float(checkpoint.get("elapsed_seconds", 0.0))
        metrics_rows = read_metrics(metrics_path)
        print(f"Resuming from epoch {start_epoch} using {resume_path}")

    training_started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_started = time.perf_counter()
        learning_rate = scheduler.set_epoch(epoch)
        train_metrics = run_epoch(model, train_loader, loss_function, device, optimizer, scaler, amp_enabled, args.gradient_clip)
        validation_metrics = evaluate(model, validation_loader, loss_function, device, amp_enabled)
        improved = not best_metrics or validation_metrics["dice"] > best_metrics["dice"] + 1e-6
        if improved:
            best_epoch, best_metrics, stale_epochs = epoch, validation_metrics, 0
        else:
            stale_epochs += 1
        plateau_reduced = scheduler.step(validation_metrics["dice"])
        elapsed = prior_elapsed + time.perf_counter() - training_started
        payload = checkpoint_payload(model, optimizer, scaler, scheduler, epoch, best_epoch, best_metrics, args, model_config, split_ids, elapsed)
        torch.save(payload, last_path)
        if improved:
            torch.save(payload, best_path)
        row = {"epoch": epoch, "learning_rate": learning_rate, **{f"train_{key}": value for key, value in train_metrics.items()}, **{f"val_{key}": value for key, value in validation_metrics.items()}, "epoch_seconds": time.perf_counter() - epoch_started}
        metrics_rows = [existing for existing in metrics_rows if int(existing["epoch"]) != epoch] + [row]
        metrics_rows.sort(key=lambda item: int(item["epoch"]))
        write_metrics(metrics_path, metrics_rows)
        save_curves(metrics_rows, plots_dir)
        status = "best" if improved else "-"
        reduction_note = " plateau-lr-reduced" if plateau_reduced else ""
        print(f"epoch={epoch:03d} lr={learning_rate:.2e} train_loss={train_metrics['loss']:.4f} val_loss={validation_metrics['loss']:.4f} val_dice={validation_metrics['dice']:.4f} {status}{reduction_note}")
        if stale_epochs >= args.early_stopping_patience:
            print(f"Early stopping at epoch {epoch}; no validation Dice improvement for {stale_epochs} epochs.")
            break

    if not best_path.exists():
        raise RuntimeError("No best_model.pth was produced.")
    best_checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state"])
    test_metrics = evaluate(model, test_loader, loss_function, device, amp_enabled) if test_loader else None
    training_time = prior_elapsed + time.perf_counter() - training_started
    summary = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "task": "DRIVE retinal vessel segmentation",
        "data_root": str(data_root),
        "device": str(device),
        "amp_enabled": amp_enabled,
        "completed_epochs": int(metrics_rows[-1]["epoch"]),
        "best_epoch": best_epoch,
        "best_validation_metrics": best_metrics,
        "test_metrics": test_metrics,
        "training_time_seconds": training_time,
        "model_config": {"architecture": "swin_unet", **model_config},
        "hyperparameters": {
            "batch_size": args.batch_size, "image_size": args.image_size, "learning_rate": args.learning_rate,
            "minimum_learning_rate": args.minimum_learning_rate, "weight_decay": args.weight_decay, "warmup_epochs": args.warmup_epochs,
            "plateau_patience": args.plateau_patience, "plateau_factor": args.plateau_factor, "early_stopping_patience": args.early_stopping_patience,
            "gradient_clip": args.gradient_clip, "positive_weight": args.positive_weight, "dice_weight": args.dice_weight, "bce_weight": args.bce_weight, "focal_gamma": args.focal_gamma,
            "workers": args.workers, "seed": args.seed, "train_samples_per_image": args.train_samples_per_image,
        },
        "splits": split_ids,
        "warning": "Validation metrics are from a small automatically split DRIVE training subset. The official DRIVE test field-of-view masks are not vessel ground truth.",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    generate_training_report(report_path, summary, plots_dir)
    print(json.dumps({"best_validation_metrics": best_metrics, "test_metrics": test_metrics, "best_model": str(best_path), "last_model": str(last_path), "metrics": str(metrics_path), "report": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
