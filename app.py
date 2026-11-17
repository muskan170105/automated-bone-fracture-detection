"""FractureAI – Home page (Streamlit entry point)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from src.utils import html, load_config, model_path, render_disclaimer, setup_page

setup_page("Home")

html("""
<div class="hero">
<div class="badge">🩻 AI-assisted musculoskeletal radiograph research</div>
<h1>FractureAI</h1>
<div class="tag">Automated bone fracture detection and localization in X-rays</div>
<p>A YOLOv8-Nano research prototype that scans limb radiographs, localizes suspected fracture regions with bounding boxes
and reports confidence scores, with an honest, transparent evaluation of what it can and cannot do.</p>
</div>
""")

cfg = load_config()
weights = model_path(cfg)
c1, c2 = st.columns([1, 3])
with c1:
    st.page_link("pages/1_Fracture_Detection.py", label="Get Started", icon="🚀")
with c2:
    if weights.is_file():
        html('<div class="status-ok">✅ Trained model found, detection mode is active.</div>')
    else:
        html('<div class="status-warn">⚙️ Setup mode: no trained model (models/best.pt) yet. Run the training section of the notebook.</div>')

st.markdown("### Key features")
features = [
    ("🎯", "Fracture localization", "Bounding boxes highlight regions the model considers suspicious."),
    ("📊", "Confidence scores", "Every detection includes its model confidence."),
    ("⚡", "Fast inference", "YOLOv8-Nano is lightweight; latency is measured and reported."),
    ("🔍", "Small-region aware", "Resolution options and optional tiled inference for small findings."),
    ("📈", "Transparent metrics", "Sensitivity, specificity, mAP, IoU computed on unseen test images."),
    ("🔒", "Privacy by default", "Uploads are processed in memory and never stored."),
]
html('<div class="grid">' + "".join(
    f'<div class="card"><div class="icon">{i}</div><h3>{t}</h3><p>{d}</p></div>' for i, t, d in features) + "</div>")

st.markdown("### Project objectives")
html("""
<div class="card"><p>
1) Detect possible fractures in X-ray images &nbsp;·&nbsp; 2) Localize suspected regions &nbsp;·&nbsp;
3) Support small-region detection where data permits &nbsp;·&nbsp; 4) Display confidence &nbsp;·&nbsp;
5) Produce annotated images &nbsp;·&nbsp; 6) Evaluate sensitivity &amp; specificity &nbsp;·&nbsp;
7) Measure precision, recall, mAP, IoU and latency &nbsp;·&nbsp; 8) Provide an interactive website.
</p></div>
""")

st.markdown("### How it works")
steps = ["X-ray Image", "Preprocessing", "YOLOv8-Nano", "Detection", "Localization", "Annotated Result"]
html('<div class="pipeline">' + '<div class="arrow">→</div>'.join(
    f'<div class="step"><div class="n">{n}</div><h4>{s}</h4></div>' for n, s in enumerate(steps, 1)) + "</div>")

st.markdown("### Technology stack")
html('<div class="chips">' + "".join(f"<span>{t}</span>" for t in [
    "Python", "PyTorch", "Ultralytics YOLOv8n", "OpenCV", "NumPy", "Pandas", "scikit-learn",
    "Streamlit", "Plotly", "Kaggle API", "Google Colab"]) + "</div>")

render_disclaimer()
