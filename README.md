# Retinal AI Suite

An end-to-end, local fundus-image application for two independent tasks:

1. **Retinal vessel segmentation** — train and evaluate a U-Net on the supplied DRIVE archive, then generate a vessel-mask overlay for an uploaded fundus image.
2. **Four-class fundus classification with XAI** — fine-tune EfficientNet-B0 for `A` (AMD), `D` (diabetic retinopathy), `G` (glaucoma), and `N` (normal), then generate a Grad-CAM explanation.

> **Clinical safety:** this is a research/demo project, not a diagnostic device. It must not be used to make, delay, or replace clinical decisions.

## What is included

- Reproducible training, evaluation, API, and Streamlit UI.
- DRIVE data, trained checkpoints, generated reports, and the delivery ZIP are excluded from Git. Place the DRIVE dataset under `data/raw/DRIVE` before training; checkpoints and reports are generated locally.
- A working DRIVE segmentation training route using an internal, patient-level held-out split. The archive has no public vessel annotations for its `test` images, so those images cannot be used for supervised test Dice.
- Complete but deliberately untrained classification/XAI route. The supplied archive contains only DRIVE vessel labels; it has no disease-class labels. The application clearly reports this rather than inventing a medical prediction.

## Quick start (Windows PowerShell)

```powershell
cd C:\Users\skmda\Desktop\sih26\SIH26
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Train the vessel segmenter and create the measured evaluation report
python -m retinal_ai.train_segmentation --epochs 160 --image-size 256

# Start the one-command UI (it starts the API itself)
streamlit run ui\app.py
```

Open the URL Streamlit prints (normally `http://localhost:8501`), upload a fundus image, and choose **Analyze**. The API is also available directly at `http://127.0.0.1:8000/docs`.

## Classification dataset layout

Place a legally obtained disease-labelled fundus dataset in one of these layouts before training:

```text
data/classification/
  train/{A,D,G,N}/*.png|jpg|jpeg|tif
  val/{A,D,G,N}/*.png|jpg|jpeg|tif
  test/{A,D,G,N}/*.png|jpg|jpeg|tif
```

or a single class-folder layout:

```text
data/classification/{A,D,G,N}/*.png|jpg|jpeg|tif
```

For the second form, the training command creates deterministic stratified 70/15/15 train/validation/test partitions. Then run:

```powershell
python -m retinal_ai.train_classifier --epochs 30 --batch-size 16
python -m retinal_ai.evaluate_classifier
```

The resulting `artifacts/reports/classification_metrics.json`, ROC-AUC, classification report, and confusion-matrix PNG are actual held-out-test results. Do not claim a score before this command completes.

## API

- `GET /health` – model availability.
- `POST /analyze` – multipart `image`; returns classification probabilities, Grad-CAM location, segmentation-mask location, and a non-diagnostic report.
- `GET /docs` – interactive OpenAPI documentation.

Run independently with `uvicorn retinal_ai.api:app --host 127.0.0.1 --port 8000`.

## Evaluation integrity

The training scripts separate train, validation, and held-out test data. The segmentation threshold is selected on validation only, then frozen for test evaluation. `metrics.json` records split IDs, model configuration, threshold, Dice, IoU, precision, recall, specificity, and pixel accuracy. The UI never displays a claimed Dice score; it displays only the saved measured value from the evaluated checkpoint.

The original notebooks rely on machine-specific `D:\Data` / Kaggle paths and reuse the test set as validation in places. This project replaces that with deterministic paths and non-leaking splits.

For a small training set, the optional transfer-learning route is available after the one-time official ResNet-18 ImageNet weight download:

```powershell
python -m retinal_ai.train_segmentation --architecture resnet18_unet --pretrained-encoder --epochs 100 --image-size 256
```
