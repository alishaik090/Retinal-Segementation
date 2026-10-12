# Handover Report — Retinal AI Suite

**Project path:** `C:\Users\skmda\Desktop\sih26\SIH26`  
**Handover date:** 12 October 2026  
**Delivery archive:** `Retinal_AI_Suite_Complete.zip`

## 1. Current project status

| Area | Status | Evidence |
|---|---|---|
| DRIVE vessel-segmentation dataset | Complete locally | Extracted under `data/raw/DRIVE/`; excluded from Git. |
| Vessel segmentation model | Trained locally | `artifacts/models/vessel_unet_best.pt`; checkpoint excluded from Git. |
| Segmentation evaluation | Complete locally | `artifacts/reports/segmentation_metrics.json` and visual QA PNGs; generated reports excluded from Git. |
| FastAPI inference service | Complete and smoke-tested | `retinal_ai/api.py`; `POST /analyze` accepts a DRIVE TIFF and creates a mask/overlay |
| Streamlit UI | Complete | `ui/app.py`; upload image → vessel mask/overlay; shows classifier state |
| Classification + Grad-CAM code path | Implemented, not trained | `retinal_ai/train_classifier.py`, `evaluate_classifier.py`, and `inference.py` |
| Classification evaluation | Pending data | No supplied disease-labelled `A/D/G/N` dataset |
| Automated checks | Passing | `python -m pytest -q` → 6 passed after FOV-aware pipeline changes |
| Package validation | Passing | ZIP was extracted into a clean folder; expected files and all four tests passed |

### Measured segmentation result

The saved model is a `ResNet18UNet` trained at 256×256 under a deterministic 60/20/20 split of the **20 manually labelled DRIVE training images**. Loss and metrics are restricted to the DRIVE field of view. Threshold 0.65 was selected on validation only, then frozen for the held-out test split.

| Metric | Held-out result |
|---|---:|
| Dice | **0.7217** |
| IoU | 0.5646 |
| Precision | 0.7541 |
| Recall | 0.6957 |
| Specificity | 0.9652 |
| Pixel accuracy | 0.9277 |

This **does not meet the requested Dice ≥ 0.80** and is not directly comparable to the earlier full-frame result because scoring now excludes pixels outside the field of view. It is not an official DRIVE benchmark. See `MODEL_CARD.md` and `artifacts/reports/segmentation_metrics.json` for the exact protocol, split IDs, threshold, and per-image results.

## 2. Folder structure

```text
SIH26/
├── README.md                         # Installation and use guide
├── HANDOVER.md                       # This report
├── MODEL_CARD.md                     # Data/model limitations and safe-use notes
├── requirements.txt                  # Python dependencies
├── run_app.bat                       # Starts the Streamlit UI
├── train_segmentation.bat            # Retrains segmentation and writes reports
├── retinal_ai/
│   ├── api.py                        # FastAPI service: /health and /analyze
│   ├── config.py                     # Project paths and artifact locations
│   ├── data.py                       # Dataset discovery, preprocessing, splits
│   ├── models.py                     # ResNet18 U-Net and EfficientNet-B0 models
│   ├── train_segmentation.py         # Train/evaluate vessel segmentation
│   ├── report_segmentation.py        # Training curve and held-out visual QA
│   ├── train_classifier.py           # Train four-class EfficientNet-B0
│   ├── evaluate_classifier.py        # Held-out classifier metrics/CM
│   ├── inference.py                  # Segmentation, classifier, Grad-CAM helpers
│   └── metrics.py                    # Dice, IoU and pixel metric functions
├── ui/
│   └── app.py                        # Streamlit upload UI
├── tests/
│   ├── test_data.py                  # DRIVE discovery/split tests
│   └── test_metrics.py               # Metric correctness tests
├── data/
│   ├── raw/DRIVE/                    # Supplied DRIVE archive, extracted
│   └── classification/               # Create/populate with A/D/G/N dataset
└── artifacts/
    ├── models/vessel_unet_best.pt    # Saved trained segmentation checkpoint
    ├── reports/                      # Metrics, training curve, qualitative QA
    ├── uploads/                      # Generated API/UI results; runtime-only
    └── hub/checkpoints/              # Cached official ResNet-18 ImageNet weights
```

## 3. Completed features

### Vessel segmentation

- Pairs DRIVE training fundus images with first-manual vessel annotations.
- Uses deterministic, non-overlapping train/validation/test partitioning.
- Trains Dice + weighted binary-cross-entropy objective with early stopping and best-validation checkpointing.
- Uses an ImageNet-pretrained ResNet-18 encoder with a U-Net-style decoder.
- Selects the binary threshold on validation data only, then reports held-out metrics.
- Saves metrics JSON, training history CSV, training curve, and visual comparisons of original / manual mask / predicted mask.

### API and UI

- `GET /health` returns model availability and saved metrics.
- `POST /analyze` accepts uploaded fundus images, returns vessel mask/overlay URLs and a research-use disclaimer.
- Streamlit UI automatically starts the local API if needed, accepts common image types, displays results, and shows model availability.
- Artifacts are saved locally in `artifacts/uploads/<request-id>/`.

### Disease classification and XAI implementation

- EfficientNet-B0 four-class head for `A` (AMD), `D` (diabetic retinopathy), `G` (glaucoma), `N` (normal).
- Training script supports supplied train/validation/test folders or deterministic stratified splitting of a class-folder dataset.
- Held-out accuracy, weighted F1, log loss, ROC-AUC, per-class results, classification report, and confusion matrix are implemented.
- Grad-CAM is generated for a trained classifier and is returned by the API/UI.
- In the absence of a trained classifier checkpoint, the API/UI correctly return **no disease prediction**.

## 4. Dataset paths

### Supplied data (available)

```text
data/raw/DRIVE/training/images/       # 20 labelled-training fundus TIFF images
data/raw/DRIVE/training/1st_manual/   # 20 corresponding vessel masks
data/raw/DRIVE/training/mask/         # field-of-view masks
data/raw/DRIVE/test/images/           # 20 fundus TIFF images
data/raw/DRIVE/test/mask/             # field-of-view masks; not vessel ground truth
```

### Required classification data (not supplied)

Use one of these layouts:

```text
data/classification/
├── train/{A,D,G,N}/*.{png,jpg,jpeg,tif,tiff}
├── val/{A,D,G,N}/*.{png,jpg,jpeg,tif,tiff}
└── test/{A,D,G,N}/*.{png,jpg,jpeg,tif,tiff}
```

or:

```text
data/classification/{A,D,G,N}/*.{png,jpg,jpeg,tif,tiff}
```

The latter is split stratifiably 70/15/15 by `train_classifier.py`. Use properly licensed data and keep patient-level duplicates in the same split.

## 5. Model architecture

### Trained vessel segmenter

```text
RGB fundus (256 × 256)
  → ImageNet-normalized ResNet-18 encoder
  → skip connections from encoder stages
  → transposed-convolution U-Net decoder
  → 1-channel logits
  → sigmoid + validation-selected 0.50 threshold
  → binary vessel mask
```

- Class: `ResNet18UNet` in `retinal_ai/models.py`
- Encoder: pretrained ResNet-18 from official torchvision weights
- Decoder: U-Net-style transposed convolutions and convolution blocks
- Loss: weighted `BCEWithLogitsLoss` + Dice loss
- Optimizer: AdamW; learning-rate reduction on validation Dice
- Input: 256 × 256 RGB; output: 256 × 256 binary mask

### Pending disease classifier

```text
RGB fundus (224 × 224)
  → EfficientNet-B0
  → 4-unit linear classification head
  → softmax probabilities for A / D / G / N
  → Grad-CAM from the final feature layer
```

## 6. Training and run commands

```powershell
cd C:\Users\skmda\Desktop\sih26\SIH26
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Run the UI

```powershell
streamlit run ui\app.py
# or double-click run_app.bat
```

Open `http://localhost:8501`. API documentation is at `http://127.0.0.1:8000/docs`.

### Retrain segmentation

```powershell
python -m retinal_ai.train_segmentation --architecture resnet18_unet --pretrained-encoder --epochs 100 --image-size 256 --batch-size 2
python -m retinal_ai.report_segmentation
# or double-click train_segmentation.bat
```

### Train and evaluate classification after adding data

```powershell
python -m retinal_ai.train_classifier --epochs 30 --batch-size 16 --pretrained
python -m retinal_ai.evaluate_classifier
```

### Verify the project

```powershell
python -m pytest -q
python -m compileall retinal_ai ui tests
```

## 7. Pending tasks / next steps

1. Obtain and place an appropriately licensed `A/D/G/N` disease-labelled fundus dataset under `data/classification/`.
2. Train the EfficientNet-B0 classifier, run its held-out evaluation, and inspect class balance, ROC-AUC, calibration, and confusion matrix.
3. Improve vessel Dice using more manually annotated images, higher-resolution training, reproducible cross-validation, and GPU training. Do not tune on the held-out test data.
4. Obtain official DRIVE vessel labels/test protocol (or a different licensed segmentation benchmark) before reporting benchmark-level performance.
5. Add patient-level split metadata, input-quality checks, and calibration/uncertainty reporting before any serious research demonstration.
6. Add authentication, persistent job tracking, rate limiting, audit logs, and HTTPS only if deploying beyond localhost.
7. Seek clinical/ethical review and external validation before any clinical-facing use. This project is not a medical device.

## 8. Known bugs and limitations

- **Dice target is unmet:** current held-out FOV Dice is 0.7217, not ≥0.80. Pixel accuracy must not be substituted for Dice.
- **No disease dataset was provided:** classification and Grad-CAM code exists but there is no trained disease classifier or classification metric report.
- **Small segmentation data:** only 20 manually labelled source images are available for the internal split; results have high variance and are not an official DRIVE score.
- **DRIVE `test/mask` files are field-of-view masks, not vessel annotations:** they cannot be used to calculate vessel Dice.
- **CPU training is slow:** use a CUDA-capable environment for higher-resolution experiments and larger models.
- **Network is needed only when requesting uncached pretrained torchvision weights:** this delivery already includes the cached ResNet-18 weights under `artifacts/hub/checkpoints/`.
- **The Streamlit launcher starts a local child API process:** closing the Streamlit window may not stop a manually started API; stop it with the terminal process if required.
- **Generated uploads grow over time:** `artifacts/uploads/` has no retention job and should be cleaned according to data-governance policy.

## 9. Handover acceptance checklist

- [x] Source code present
- [x] Supplied DRIVE dataset included
- [x] Trained segmentation checkpoint included
- [x] Actual metrics and visual evaluation included
- [x] API and UI source included
- [x] Classification/XAI implementation included
- [x] Test suite passes
- [x] Portable ZIP tested by clean extraction
- [ ] Dice ≥ 0.80 achieved — **not achieved with supplied data**
- [ ] Disease classifier trained/evaluated — **awaiting labelled A/D/G/N data**
