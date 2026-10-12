"""FastAPI interface used by the local Streamlit UI and external clients."""
from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import ARTIFACTS_DIR, UPLOADS_DIR, ensure_directories
from .inference import ClassificationService, SegmentationService, blend_heatmap, blend_mask, decode_rgb, write_rgb
from .reporting import generate_medical_report

ensure_directories()
app = FastAPI(title="Retinal AI Suite API", version="1.0.0", description="Local research API for vessel segmentation and, when trained, disease classification/XAI.")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"], allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["*"])
app.mount("/artifacts", StaticFiles(directory=str(ARTIFACTS_DIR)), name="artifacts")
segmenter = SegmentationService()
classifier = ClassificationService()


def public_artifact(path: Path) -> str:
    return "/artifacts/" + path.relative_to(ARTIFACTS_DIR).as_posix()


@app.get("/")
def root():
    return {"service": "Retinal AI Suite", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    segmentation = {"available": segmenter.available, "metrics": (segmenter.measured_metrics.get("test_metrics") or segmenter.measured_metrics.get("best_validation_metrics")) if segmenter.measured_metrics else None}
    classification = {"available": classifier.available, "metrics": classifier.measured_metrics.get("test_metrics") if classifier.measured_metrics else None}
    return {"status": "ok", "segmentation": segmentation, "classification": classification}


@app.post("/analyze")
async def analyze(image: UploadFile = File(...)):
    content = await image.read()
    if not content or len(content) > 20 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image must be between 1 byte and 20 MB.")
    try:
        rgb = decode_rgb(content)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    if not segmenter.available:
        raise HTTPException(status_code=503, detail="The vessel segmentation checkpoint is missing. Train the model before using the analysis endpoint.")

    request_id = uuid.uuid4().hex
    output_dir = UPLOADS_DIR / request_id
    output_dir.mkdir(parents=True, exist_ok=False)
    original_path = output_dir / "original.jpg"
    write_rgb(original_path, rgb)
    mask, probability = segmenter.predict(rgb)
    mask_path = output_dir / "vessel_mask.png"
    overlay_path = output_dir / "vessel_overlay.jpg"
    write_rgb(mask_path, (mask.astype("uint8") * 255))
    write_rgb(overlay_path, blend_mask(rgb, mask))
    response = {
        "request_id": request_id,
        "disclaimer": "Research/demo output only. It is not a diagnosis and must not be used for clinical decisions.",
        "input": {"filename": image.filename, "width": int(rgb.shape[1]), "height": int(rgb.shape[0]), "original_url": public_artifact(original_path)},
        "segmentation": {"available": True, "threshold": segmenter.threshold, "vessel_pixel_fraction": float(mask.mean()), "mask_url": public_artifact(mask_path), "overlay_url": public_artifact(overlay_path), "measured_test_metrics": segmenter.measured_metrics.get("test_metrics") if segmenter.measured_metrics else None},
        "classification": {"available": False, "reason": "No disease-labelled A/D/G/N training data/checkpoint is present. No disease prediction has been made."},
    }
    if classifier.available:
        label, probabilities, heatmap = classifier.predict_with_gradcam(rgb)
        heatmap_path = output_dir / "gradcam.jpg"
        write_rgb(heatmap_path, blend_heatmap(rgb, heatmap))
        response["classification"] = {"available": True, "predicted_class": label, "probabilities": probabilities, "gradcam_url": public_artifact(heatmap_path), "explanation": "Grad-CAM shows image regions that most increased this model's predicted class score; it is not evidence of disease."}
    report_path = output_dir / "medical_report.pdf"
    try:
        generate_medical_report(
            report_path,
            response,
            {"original": original_path, "overlay": overlay_path, "gradcam": output_dir / "gradcam.jpg"},
        )
        response["medical_report"] = {"available": True, "url": public_artifact(report_path)}
    except Exception as error:  # Keep a successful image analysis usable if report rendering fails.
        response["medical_report"] = {"available": False, "reason": f"PDF report generation failed: {error}"}
    return response
