# Model card and delivery status

## Scope

This package combines the two requested notebook workflows into one local application:

- **Vessel segmentation:** a trained ResNet-18 U-Net producing a binary retinal-vessel mask.
- **Disease classification:** an EfficientNet-B0 training/inference/XAI implementation for AMD (`A`), diabetic retinopathy (`D`), glaucoma (`G`), and normal (`N`).

The application explicitly separates these tasks. A vessel mask is not a disease diagnosis.

## Data audit

The provided `archive (6).zip` expands to DRIVE images and masks only:

- 20 training images with first manual vessel annotations.
- 20 DRIVE test images with field-of-view masks, **not** vessel ground truth.
- No `A`, `D`, `G`, `N` disease classes and no FIVES classification images.

The original classification notebook documents FIVES class labels and expects folders on `D:\Data\fundus`; those files are not present in this delivery. The original segmentation notebook refers to Kaggle paths and uses its test set again for validation. Neither machine-specific paths nor that data leakage are used here.

## Trained segmentation checkpoint

`artifacts/models/vessel_unet_best.pt` was trained from the supplied labelled DRIVE training images using a deterministic 60/20/20 split. The ResNet-18 encoder begins with official ImageNet weights and was fine-tuned with a Dice+BCE objective.

The checkpoint and DRIVE data are local-only and are not included in this Git repository. The metrics below document the latest measured local run; reproduce it after placing the dataset under `data/raw/DRIVE` and running the training script. It uses 256×256 inputs, field-of-view-masked loss and evaluation, and a threshold selected on validation only.

| Held-out metric | Measured value |
|---|---:|
| Dice | **0.7217** |
| IoU | 0.5646 |
| Precision | 0.7541 |
| Recall | 0.6957 |
| Specificity | 0.9652 |
| Pixel accuracy | 0.9277 |

The exact split IDs, threshold, per-image scores, and configuration are in `artifacts/reports/segmentation_metrics.json`. `segmentation_qualitative_test_samples.png` is visual QA of those same held-out samples.

## Important limitation

The requested **Dice ≥ 0.80 has not been reached** on the supplied data under the non-leaking protocol. This FOV-restricted score must not be compared directly with the earlier full-frame score. This package does not select its threshold on the test set or report pixel accuracy as Dice. A stronger attempt at the target needs more labelled vessel images (for example, a suitable licensed retinal-vessel dataset), a predeclared validation/test protocol, and preferably GPU training.

The classifier has **not** been trained or tested because its required labelled disease dataset was not provided. Its API/UI status is intentionally `unavailable` and it emits no disease prediction until a valid `data/classification/{train,val,test}/{A,D,G,N}` dataset is supplied and the documented training/evaluation commands run.

## Safe use

Outputs are research/demo artifacts only. They are not validated for clinical care, triage, diagnosis, or treatment and must not replace assessment by a qualified clinician.

