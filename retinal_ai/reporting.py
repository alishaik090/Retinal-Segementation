"""PDF reports for model training and one-image research analysis."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image as ReportImage
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def _styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], alignment=TA_CENTER, textColor=colors.HexColor("#123047"), spaceAfter=12))
    styles.add(ParagraphStyle(name="Section", parent=styles["Heading2"], textColor=colors.HexColor("#0B6477"), spaceBefore=10, spaceAfter=6))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8.5, leading=11))
    return styles


def _page_number(canvas, document) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#566573"))
    canvas.drawString(40, 24, "Retinal AI Suite - research use only")
    canvas.drawRightString(A4[0] - 40, 24, f"Page {document.page}")
    canvas.restoreState()


def _table(rows: list[list[str]], widths: list[float] | None = None) -> Table:
    table = Table(rows, colWidths=widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#123047")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D5DBDB")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), (colors.white, colors.HexColor("#F7F9F9"))),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


def generate_training_report(output_path: Path, summary: dict[str, Any], plots_dir: Path) -> Path:
    """Create a compact, self-contained PDF for a completed training run."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    styles = _styles()
    document = SimpleDocTemplate(str(output_path), pagesize=A4, rightMargin=40, leftMargin=40, topMargin=42, bottomMargin=38)
    story = [
        Paragraph("Swin-UNet DRIVE Training Report", styles["ReportTitle"]),
        Paragraph("Automatically generated research training report. Metrics are measured values from the configured validation partition; they are not a clinical or benchmark claim.", styles["BodyText"]),
        Spacer(1, 8),
        Paragraph("Run summary", styles["Section"]),
    ]
    best = summary["best_validation_metrics"]
    story.append(_table([
        ["Metric", "Value"],
        ["Best validation Dice", f"{best['dice']:.4f}"],
        ["Best validation IoU", f"{best['iou']:.4f}"],
        ["Best validation precision", f"{best['precision']:.4f}"],
        ["Best validation recall", f"{best['recall']:.4f}"],
        ["Epoch of best checkpoint", str(summary["best_epoch"])],
        ["Completed epochs", str(summary["completed_epochs"])],
        ["Total training time", f"{summary['training_time_seconds']:.1f} seconds"],
        ["Device", str(summary["device"])],
    ], widths=[2.8 * inch, 3.7 * inch]))
    story += [Spacer(1, 8), Paragraph("Model configuration", styles["Section"])]
    model = summary["model_config"]
    story.append(_table([["Setting", "Value"]] + [[key.replace("_", " ").title(), str(value)] for key, value in model.items()], widths=[2.2 * inch, 4.3 * inch]))
    story += [PageBreak(), Paragraph("Training hyperparameters", styles["Section"])]
    hyperparameters = summary["hyperparameters"]
    story.append(_table([["Setting", "Value"]] + [[key.replace("_", " ").title(), str(value)] for key, value in hyperparameters.items()], widths=[2.8 * inch, 3.7 * inch]))
    story += [Spacer(1, 8), Paragraph("Data protocol", styles["Section"])]
    split = summary["splits"]
    story.append(Paragraph(
        f"DRIVE root: {summary['data_root']}<br/>"
        f"Training images: {len(split['train'])}; validation images: {len(split['validation'])}; test images: {len(split['test'])}.<br/>"
        "The validation partition is generated deterministically from the DRIVE manually labelled training images using the recorded seed. "
        "The official DRIVE test field-of-view masks are not used as vessel ground truth.", styles["Small"]))
    story.append(PageBreak())
    story.append(Paragraph("Learning curves", styles["Section"]))
    for title, filename in (("Dice", "dice_curve.png"), ("IoU", "iou_curve.png"), ("Loss", "loss_curve.png"), ("Precision", "precision_curve.png"), ("Recall", "recall_curve.png")):
        image_path = plots_dir / filename
        if image_path.exists():
            story += [Paragraph(title, styles["Heading3"]), ReportImage(str(image_path), width=6.6 * inch, height=3.25 * inch), Spacer(1, 6)]
    document.build(story, onFirstPage=_page_number, onLaterPages=_page_number)
    return output_path


def generate_medical_report(output_path: Path, analysis: dict[str, Any], image_paths: dict[str, Path]) -> Path:
    """Create a non-diagnostic single-image research report for the UI/API."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    styles = _styles()
    document = SimpleDocTemplate(str(output_path), pagesize=A4, rightMargin=40, leftMargin=40, topMargin=42, bottomMargin=38)
    classification = analysis["classification"]
    segmentation = analysis["segmentation"]
    story = [
        Paragraph("Retinal Image Research Report", styles["ReportTitle"]),
        Paragraph("Important: this output is generated by a research/demo model. It is not a diagnosis, clinical decision, screening clearance, or treatment recommendation.", styles["BodyText"]),
        Spacer(1, 8),
        Paragraph("Image analysis", styles["Section"]),
        _table([
            ["Field", "Result"],
            ["Input", analysis["input"].get("filename") or "Uploaded retinal image"],
            ["Vessel pixel fraction", f"{100 * segmentation['vessel_pixel_fraction']:.2f}%"],
            ["Segmentation threshold", f"{segmentation['threshold']:.2f}"],
            ["Disease prediction", classification.get("predicted_class", "Unavailable - classifier has not been trained")],
            ["Confidence", f"{100 * max(classification.get('probabilities', {'unavailable': 0}).values()):.2f}%" if classification.get("available") else "Unavailable"],
            ["XAI visualization", "Grad-CAM generated" if classification.get("available") else "Unavailable - no trained classification checkpoint"],
            ["Risk category", "Model-estimated only; not clinically validated" if classification.get("available") else "Indeterminate"],
        ], widths=[2.0 * inch, 4.5 * inch]),
        Spacer(1, 8),
        Paragraph("Clinical interpretation and recommendation", styles["Section"]),
        Paragraph(
            "The vessel mask highlights structures that the segmentation model classified as retinal vessels. "
            "It does not establish the presence or absence of retinal disease. "
            + ("A classifier result and its Grad-CAM are included as model evidence only, not as a medical finding. " if classification.get("available") else "No disease-classification result is provided because a disease-labelled classifier checkpoint is unavailable. ")
            + "If this image is clinically relevant, it should be reviewed by a qualified eye-care professional using validated clinical workflows.",
            styles["BodyText"],
        ),
        Spacer(1, 8),
        Paragraph("Visual outputs", styles["Section"]),
    ]
    cells = []
    for label, key in (("Original", "original"), ("Vessel overlay", "overlay"), ("Grad-CAM", "gradcam")):
        path = image_paths.get(key)
        if path and path.exists():
            cells.append([Paragraph(label, styles["Small"]), ReportImage(str(path), width=2.05 * inch, height=2.05 * inch)])
    if cells:
        row_one = [[cell[0] for cell in cells], [cell[1] for cell in cells]]
        visuals = Table(row_one, hAlign="LEFT")
        visuals.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]))
        story.append(visuals)
    story += [Spacer(1, 8), Paragraph(f"Generated {datetime.now(timezone.utc).isoformat()}", styles["Small"])]
    document.build(story, onFirstPage=_page_number, onLaterPages=_page_number)
    return output_path
