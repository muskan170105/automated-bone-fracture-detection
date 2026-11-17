# 🩻 Automated Bone Fracture Detection and Localization in Musculoskeletal X-Rays

> **Research / educational prototype. Not clinically validated, not a medical device. Results require review by a qualified radiologist.**

## Introduction & problem statement
Fractures (particularly small or hairline ones) can be overlooked in busy clinical settings. This project fine-tunes **YOLOv8-Nano** to propose
possible fracture regions in limb radiographs and serves it through a **Streamlit** website.

## Objectives
Detect and localize possible fractures, show confidence, produce annotated images, evaluate sensitivity/specificity/precision/recall/mAP/IoU/latency, and provide an interactive website.

## Technology stack
Python · PyTorch · Ultralytics YOLOv8n · OpenCV · NumPy · Pandas · Matplotlib · scikit-learn · Streamlit · Plotly · Kaggle API · Google Colab (free tier).

## Dataset
A Kaggle bone-fracture dataset **with bounding-box annotations** chosen by you in the notebook (Section 3). The notebook verifies annotations before use.
MURA has normal/abnormal study labels but no fracture boxes, so it is not used to train the detector. Check each dataset's licence.

## Architecture
```
app.py, pages/      Streamlit website (Home, Detection, Performance, About, How It Works)
src/                detector.py (inference + drawing + tiling), image_processor.py, evaluation.py, utils.py
training/           convert_annotations.py, validate_dataset.py, train.py
config/config.yaml  model path & defaults        assets/  CSS + logo
models/best.pt      trained weights               results/ metrics, curves, reports
tests/              pytest suite
```

## Colab setup (Colab is the only environment needed)
1. Open `Automated_Bone_Fracture_Detection.ipynb` in Google Colab, choose **Runtime → Change runtime type → T4 GPU**.
2. Run cells top-to-bottom.

## Kaggle setup
Kaggle → Settings → API → *Create New Token* → upload `kaggle.json` when prompted. Search the listed datasets, set `DATASET_SLUG`, and run the verification cell.

## Training procedure
Group-aware 70/15/15 split → YOLOv8n fine-tuning (`imgsz`, `batch`, `epochs`, `lr0` configurable; seed 42, deterministic) → best checkpoint saved as `models/best.pt`.
Command-line equivalent: `python training/train.py --data <data.yaml> --epochs 50 --imgsz 640`.

## Evaluation methodology
Confidence/NMS-IoU tuned on validation only; single evaluation on the held-out test split. Image-level ground truth: positive ⇔ ≥ 1 GT box. Sensitivity = TP/(TP+FN),
specificity = TN/(TN+FP) (reported N/A if no negatives exist). Box-level matching at IoU ≥ 0.5; mAP50/mAP50-95 via Ultralytics; latency = batch-1 `predict()` after 10 warm-up runs (hardware recorded in `results/metrics.json`).
All metrics are generated, never hand-written.

## Website features
Upload JPG/JPEG/PNG → validation → YOLOv8 inference → boxes, labels, confidences, inference time, side-by-side view, PNG download, reset. Setup mode when no weights exist. Uploads are not stored.

## Running the website from Colab
Run the final "Launch website" cell. It starts Streamlit on port 8501 and opens a free **Cloudflare quick tunnel**, printing a temporary `https://…trycloudflare.com` URL.
The link works only while the Colab runtime is alive; if the runtime disconnects, re-run the setup cells (or restore from Google Drive) and the launch cell. It is **not permanent hosting**.

## Model integration
`config/config.yaml → model.weights` points to `models/best.pt`. `src/detector.py::FractureDetector` loads it lazily; the pages never fabricate predictions.
To use your own model: copy it to `models/best.pt` (class names are read from the weights).

## Troubleshooting
- *No GPU*: Runtime → Change runtime type → T4 GPU.  - *Kaggle 403*: accept the dataset's rules on kaggle.com.
- *No boxes found*: choose another dataset or annotate manually (Roboflow/CVAT/LabelImg → export YOLO).
- *Out of memory*: lower `BATCH` or `IMGSZ`.  - *Tunnel URL not shown*: re-run the launch cell.
- *Uploads fail via tunnel*: the demo launch disables XSRF protection for the temporary tunnel only; keep it enabled for real deployments.

## Limitations
Single-dataset training; no external validation; annotation noise; hairline fissure detection not claimed; "no detection" ≠ "no fracture".

## Future scope & GitHub / permanent deployment
Expert-reviewed labels, external validation, larger models, DICOM, uncertainty estimates.
GitHub: `git init && git add . && git commit -m "Initial commit"` then push to your repo (do this from a Colab cell or upload the ZIP via the GitHub web UI).
Deploy free via **Streamlit Community Cloud** (connect the repo, main file `app.py`) or **Hugging Face Spaces** (Streamlit SDK). Mind the AGPL-3.0 licence of Ultralytics and your dataset licence; large weights may need Git LFS.

## Medical disclaimer
This software is for education and research only. It has not been clinically validated and must not be used for diagnosis, treatment or any clinical decision. A qualified radiologist must review every image.
