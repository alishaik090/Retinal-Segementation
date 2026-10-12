from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DRIVE_DIR = DATA_DIR / "raw" / "DRIVE"
CLASSIFICATION_DIR = DATA_DIR / "classification"
ARTIFACTS_DIR = ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
UPLOADS_DIR = ARTIFACTS_DIR / "uploads"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
BEST_MODEL_CHECKPOINT = ARTIFACTS_DIR / "best_model.pth"
LAST_MODEL_CHECKPOINT = ARTIFACTS_DIR / "last_model.pth"
TRAINING_METRICS = ARTIFACTS_DIR / "metrics.csv"
TRAINING_SUMMARY = ARTIFACTS_DIR / "training_summary.json"
TRAINING_REPORT = ARTIFACTS_DIR / "training_report.pdf"

SEGMENTATION_CHECKPOINT = MODELS_DIR / "vessel_unet_best.pt"
CLASSIFIER_CHECKPOINT = MODELS_DIR / "fundus_efficientnet_b0_best.pt"
SEGMENTATION_METRICS = REPORTS_DIR / "segmentation_metrics.json"
CLASSIFICATION_METRICS = REPORTS_DIR / "classification_metrics.json"


def ensure_directories() -> None:
    for directory in (DATA_DIR, ARTIFACTS_DIR, MODELS_DIR, REPORTS_DIR, UPLOADS_DIR, PLOTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
