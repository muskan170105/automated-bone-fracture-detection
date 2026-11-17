"""Visual processing workflow."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from src.utils import html, page_header, render_disclaimer, setup_page

setup_page("How It Works", "⚙️")
page_header("How It Works", "From an uploaded radiograph to an annotated result.")

steps = [("X-ray Image", "JPG / JPEG / PNG upload, validated and kept in memory."),
         ("Preprocessing", "8-bit RGB conversion, orientation fix, optional CLAHE."),
         ("YOLOv8-Nano", "Fine-tuned detector runs on the (optionally tiled) image."),
         ("Fracture Detection", "Candidate regions with confidence scores; NMS removes duplicates."),
         ("Fracture Localization", "Bounding-box coordinates in original pixel space."),
         ("Annotated Result", "Boxes and labels drawn; image available to download.")]
html('<div class="pipeline">' + '<div class="arrow">→</div>'.join(
    f'<div class="step"><div class="n">{i}</div><h4>{t}</h4><p>{d}</p></div>' for i, (t, d) in enumerate(steps, 1)) + "</div>")

st.markdown("### Under the hood")
st.markdown("""
- **Confidence threshold** keeps detections the model is sufficiently sure about. It was tuned on validation images, not test images.
- **NMS IoU** merges overlapping boxes that describe the same region.
- **Tiled inference** (optional) slices large images into overlapping tiles so small regions keep more pixels, then merges results.
- **Privacy:** uploads are never written to disk by the app.
""")
render_disclaimer()
