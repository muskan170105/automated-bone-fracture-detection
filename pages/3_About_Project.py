
"""About the project."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from src.utils import (
    html,
    page_header,
    render_disclaimer,
    setup_page
)

setup_page("About Project", "ℹ️")

st.markdown("""
<style>
[data-testid="stMetricValue"] {
    color: #0875d1 !important;
}
[data-testid="stMetricLabel"] {
    color: #52647a !important;
}
</style>
""", unsafe_allow_html=True)

page_header(
    "About the Project",
    "Problem, approach, data, limitations and future scope."
)


def card(icon, title, body):
    html(
        f'<div class="card"><div class="icon">{icon}</div>'
        f'<h3>{title}</h3><p>{body}</p></div>'
    )


# Problem and solution
a, b = st.columns(2)

with a:
    card(
        "🩺",
        "Problem statement",
        "Fractures, especially small or hairline ones, can be missed "
        "on busy shifts. A second-reader tool that flags suspicious "
        "regions could support radiologists and emergency clinicians."
    )

with b:
    card(
        "💡",
        "Proposed solution",
        "Fine-tune a lightweight YOLOv8-Nano object detector on "
        "annotated X-rays and expose it through an easy web interface "
        "with transparent evaluation."
    )

st.write("")

# Methodology
st.markdown("### Methodology")

st.markdown("""
1. Verify that the dataset has real bounding-box annotations.
2. Validate images and labels and convert annotations to YOLO format.
3. Split train/validation/test by group to avoid data leakage.
4. Fine-tune pretrained YOLOv8-Nano with X-ray-appropriate augmentation.
5. Tune confidence and NMS-IoU on validation data.
6. Evaluate on the unseen test dataset using precision, recall and mAP.
""")

# Architecture
st.markdown("### YOLOv8-Nano Architecture")

st.markdown("""
YOLOv8 is a single-stage, anchor-free object detector.

- **Backbone:** Extracts features from the X-ray image.
- **Neck:** Combines features at multiple scales.
- **Detection head:** Predicts bounding boxes and class probabilities.

The Nano variant is lightweight and suitable for relatively fast inference.
Small fractures can be difficult to detect because they occupy fewer pixels.
""")

# Dataset information
st.markdown("### Dataset Information")

dataset_report = {
    "images_found": 4148,
    "images_used": 4148,
    "classes": [
        "elbow positive",
        "fingers positive",
        "forearm fracture",
        "humerus fracture",
        "humerus",
        "shoulder fracture",
        "wrist positive"
    ],
    "splits": {
        "train": {
            "images": 2904,
            "positive": 1473,
            "negative": 1431,
            "boxes": 1698
        },
        "val": {
            "images": 623,
            "positive": 290,
            "negative": 333,
            "boxes": 343
        },
        "test": {
            "images": 621,
            "positive": 297,
            "negative": 324,
            "boxes": 347
        }
    },
    "split_method": "Grouped augmentation-source split, seed=42"
}

st.json(dataset_report)

st.caption(
    "Dataset: Bone Fracture Detection (Roboflow). "
    "Segmentation annotations were converted into YOLO bounding boxes."
)

# Model performance
st.markdown("### Model Performance")

cols = st.columns(4)

for col, label, value in zip(
    cols,
    ["Precision", "Recall", "mAP50", "mAP50-95"],
    ["32.0%", "28.1%", "20.5%", "5.6%"]
):
    col.metric(label, value)

st.caption(
    "Evaluation on 621 held-out test images. "
    "Confidence=0.001, IoU=0.60, image size=640."
)

# Technology stack
st.markdown("### Technology Stack")

html(
    '<div class="chips">' +
    "".join(
        f"<span>{tech}</span>"
        for tech in [
            "Python",
            "PyTorch",
            "Ultralytics YOLOv8n",
            "OpenCV",
            "NumPy",
            "Pandas",
            "scikit-learn",
            "Matplotlib",
            "Streamlit",
            "Plotly",
            "Google Colab"
        ]
    ) +
    "</div>"
)

# Limitations
st.markdown("### Limitations")

st.markdown("""
- Not clinically validated; trained on a public dataset.
- Dataset annotations may contain noise or duplicates.
- Some predictions may be incorrect or poorly localized.
- Rare classes have insufficient training examples.
- Hairline fracture detection is not claimed.
- A no-detection result does not exclude a fracture.
""")

# Future scope
st.markdown("### Future Scope")

st.markdown("""
- Expert-reviewed annotations and improved dataset quality.
- Better class balancing and fracture localization.
- External dataset validation.
- Higher-resolution training and tiled inference.
- Body-part classification and uncertainty estimation.
- DICOM image support and permanent hosting.
""")

render_disclaimer()
