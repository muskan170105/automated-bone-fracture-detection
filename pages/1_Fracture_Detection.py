
"""Fracture detection page: upload, run real YOLOv8 inference, display + download."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from src.detector import FractureDetector, ModelNotAvailableError
from src.image_processor import (
    ImageValidationError,
    enhance_contrast,
    to_png_bytes,
    validate_image,
)
from src.utils import (
    load_config,
    model_path,
    page_header,
    read_json,
    render_disclaimer,
    results_dir,
    setup_page,
)

setup_page("Fracture Detection")

page_header(
    "Fracture Detection",
    "Upload a limb X-ray. The trained YOLOv8-Nano model proposes possible fracture regions."
)

cfg = load_config()
weights = model_path(cfg)

tuned = read_json(results_dir() / "thresholds.json") or {}

default_conf = float(
    tuned.get("conf", cfg["model"].get("conf", 0.25))
)
default_iou = float(
    tuned.get("iou", cfg["model"].get("iou", 0.45))
)


@st.cache_resource(show_spinner="Loading model…")
def get_detector(path: str, mtime: float, imgsz: int):
    detector = FractureDetector(
        path,
        imgsz=imgsz
    )
    detector.load()
    return detector


# --------------------------------------------------
# MODEL AVAILABILITY
# --------------------------------------------------

if not weights.is_file() or weights.stat().st_size == 0:
    st.warning("**Model weights unavailable.**")

    st.markdown(
        f"""
**Setup mode.** No trained model was found at `{weights}`.

1. Open the Colab notebook and run the training section.
2. Copy the trained `best.pt` file into the project's `models` folder.
3. Reload this page.

No placeholder predictions are displayed.
"""
    )

    render_disclaimer()
    st.stop()


# --------------------------------------------------
# SIDEBAR SETTINGS
# --------------------------------------------------

with st.sidebar:
    st.markdown("### Detection settings")

    conf = st.slider(
        "Confidence threshold",
        min_value=0.01,
        max_value=0.95,
        value=0.10,
        step=0.01,
        help="Predictions below this confidence are excluded."
    )

    iou = st.slider(
        "NMS IoU threshold",
        min_value=0.10,
        max_value=0.90,
        value=float(min(max(default_iou, 0.10), 0.90)),
        step=0.05,
        help="Controls suppression of overlapping predictions."
    )

    with st.expander("Advanced settings"):
        tiled = st.checkbox(
            "Tiled inference (experimental)",
            value=bool(
                cfg["app"].get("tiled_inference_default", False)
            ),
            help="Runs inference on image tiles. Experimental."
        )

        clahe = st.checkbox(
            "CLAHE contrast enhancement",
            value=bool(
                cfg["app"].get("enhance_contrast_default", False)
            )
        )

    if tuned:
        st.caption(
            f"Validation defaults: "
            f"conf={tuned.get('conf')}, "
            f"iou={tuned.get('iou')}"
        )


# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

left, right = st.columns([4, 1])

with left:
    upload = st.file_uploader(
        "Upload an X-ray image (JPG, JPEG or PNG)",
        type=["jpg", "jpeg", "png"],
        key=f"uploader_{st.session_state.uploader_key}"
    )

with right:
    st.write("")
    st.write("")

    if st.button("🗑️ Reset / clear"):
        st.session_state.uploader_key += 1
        st.rerun()


if upload is None:
    st.info("Upload an image to begin.")
    render_disclaimer()
    st.stop()


# --------------------------------------------------
# IMAGE VALIDATION
# --------------------------------------------------

try:
    image = validate_image(
        upload.getvalue(),
        max_mb=float(cfg["app"].get("max_upload_mb", 10))
    )

except ImageValidationError as exc:
    st.error(f"Invalid image: {exc}")
    st.stop()

if clahe:
    try:
        image = enhance_contrast(image)
    except Exception as exc:
        st.error(f"Image enhancement failed: {exc}")
        st.stop()


# --------------------------------------------------
# MODEL INFERENCE
# --------------------------------------------------

try:
    detector = get_detector(
        str(weights),
        weights.stat().st_mtime,
        int(cfg["model"].get("imgsz", 640))
    )

    with st.spinner("Running YOLOv8-Nano inference…"):
        result = detector.predict(
            image,
            conf=conf,
            iou=iou,
            tiled=tiled
        )

except ModelNotAvailableError:
    st.warning("Model weights unavailable.")
    st.stop()

except Exception as exc:
    st.error(f"Prediction error: {exc}")
    st.stop()


# --------------------------------------------------
# PREDICTION STATUS
# --------------------------------------------------

st.markdown("### Prediction result")

if result.detections:
    st.warning(
        f"**{len(result.detections)} possible region(s) flagged.** "
        "These are experimental model predictions, not a medical diagnosis."
    )
else:
    st.info(
        "No regions were detected at the selected confidence threshold. "
        "This does not rule out a fracture."
    )


# --------------------------------------------------
# ORIGINAL AND ANNOTATED IMAGE
# --------------------------------------------------

c1, c2 = st.columns(2)

with c1:
    st.markdown("**Original X-ray**")
    st.image(image, width="stretch")

with c2:
    st.markdown("**Model prediction**")
    st.image(result.annotated, width="stretch")


# --------------------------------------------------
# PERFORMANCE METRICS
# --------------------------------------------------

st.markdown("### Detection statistics")

m1, m2, m3 = st.columns(3)

m1.markdown(
    f"""
    <div class="metric">
        <div class="k">Inference time</div>
        <div class="v">{result.inference_ms:.1f} ms</div>
    </div>
    """,
    unsafe_allow_html=True
)

m2.markdown(
    f"""
    <div class="metric">
        <div class="k">Detections</div>
        <div class="v">{len(result.detections)}</div>
    </div>
    """,
    unsafe_allow_html=True
)

top_confidence = (
    f"{result.detections[0].confidence:.1%}"
    if result.detections
    else "–"
)

m3.markdown(
    f"""
    <div class="metric">
        <div class="k">Top confidence</div>
        <div class="v">{top_confidence}</div>
    </div>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# DETECTION DETAILS
# --------------------------------------------------

if result.detections:
    st.markdown("### Detected regions")

    detection_data = []

    for index, detection in enumerate(result.detections, start=1):
        detection_data.append({
            "No.": index,
            "Predicted class": detection.label,
            "Confidence": f"{detection.confidence:.2%}",
            "X1": round(detection.box[0]),
            "Y1": round(detection.box[1]),
            "X2": round(detection.box[2]),
            "Y2": round(detection.box[3]),
        })

    st.dataframe(
        pd.DataFrame(detection_data),
        width="stretch",
        hide_index=True
    )


# --------------------------------------------------
# DOWNLOAD ANNOTATED IMAGE
# --------------------------------------------------

st.download_button(
    "⬇️ Download annotated image",
    data=to_png_bytes(result.annotated),
    file_name="annotated_xray.png",
    mime="image/png"
)


# --------------------------------------------------
# MEDICAL DISCLAIMER
# --------------------------------------------------

render_disclaimer()
