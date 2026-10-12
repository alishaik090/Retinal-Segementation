"""Streamlit interface for the local Retinal AI Suite API."""
from __future__ import annotations

import os
import subprocess
import sys
import time

import requests
import streamlit as st


API_URL = os.environ.get("RETINAL_API_URL", "http://127.0.0.1:8000")


def api_request(method: str, path: str, **kwargs):
    timeout = kwargs.pop("timeout", 90)
    return requests.request(method, f"{API_URL}{path}", timeout=timeout, **kwargs)


def ensure_api() -> bool:
    try:
        return api_request("GET", "/health", timeout=2).ok
    except requests.RequestException:
        subprocess.Popen([sys.executable, "-m", "uvicorn", "retinal_ai.api:app", "--host", "127.0.0.1", "--port", "8000"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        for _ in range(20):
            time.sleep(0.5)
            try:
                if api_request("GET", "/health", timeout=2).ok:
                    return True
            except requests.RequestException:
                pass
    return False


st.set_page_config(page_title="Retinal AI Suite", page_icon="👁️", layout="wide")
st.title("Retinal AI Suite")
st.caption("Local research tool · vessel segmentation · optional disease classification with Grad-CAM")
st.warning("Research/demo use only — the results are not a medical diagnosis and must not guide clinical decisions.")

if not ensure_api():
    st.error("The local API could not be started. Install requirements and run: uvicorn retinal_ai.api:app --host 127.0.0.1 --port 8000")
    st.stop()

health = api_request("GET", "/health").json()
with st.sidebar:
    st.subheader("Model status")
    if health["segmentation"]["available"]:
        st.success("Vessel segmentation ready")
    else:
        st.error("Vessel model not trained")
    if health["classification"]["available"]:
        st.success("Disease classifier ready")
    else:
        st.info("Disease classifier requires labelled A/D/G/N data")
    if health["segmentation"].get("metrics"):
        st.caption("Saved held-out segmentation metrics")
        st.json(health["segmentation"]["metrics"])

upload = st.file_uploader("Upload a fundus image", type=["jpg", "jpeg", "png", "tif", "tiff", "bmp"])
if upload:
    st.image(upload, caption="Uploaded image", width=400)
    if st.button("Analyze image", type="primary"):
        with st.spinner("Generating vessel segmentation and XAI report…"):
            response = api_request("POST", "/analyze", files={"image": (upload.name, upload.getvalue(), upload.type or "image/jpeg")})
        if not response.ok:
            st.error(response.json().get("detail", "Analysis failed."))
            st.stop()
        report = response.json()
        st.success("Analysis complete")
        st.caption(report["disclaimer"])
        seg = report["segmentation"]
        left, right = st.columns(2)
        with left:
            st.image(API_URL + seg["mask_url"], caption="Predicted vessel mask")
        with right:
            st.image(API_URL + seg["overlay_url"], caption="Vessel overlay")
        st.metric("Predicted vessel pixel fraction", f"{100 * seg['vessel_pixel_fraction']:.2f}%")
        classification = report["classification"]
        st.subheader("Disease classification & explanation")
        if not classification["available"]:
            st.info(classification["reason"])
        else:
            st.metric("Model-predicted class", classification["predicted_class"])
            st.bar_chart(classification["probabilities"])
            st.image(API_URL + classification["gradcam_url"], caption="Grad-CAM explanation")
            st.caption(classification["explanation"])
        medical_report = report.get("medical_report", {})
        if medical_report.get("available"):
            st.link_button("Download research report (PDF)", API_URL + medical_report["url"])
        else:
            st.warning(medical_report.get("reason", "Medical PDF report was not generated."))

st.divider()
st.caption("API documentation: http://127.0.0.1:8000/docs · Outputs are stored locally under artifacts/uploads.")
