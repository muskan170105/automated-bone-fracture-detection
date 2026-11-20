# 🩻 Automated Bone Fracture Detection and Localization in Musculoskeletal X-Rays

> **Research and educational prototype. Not clinically validated or intended for medical diagnosis. All results require review by a qualified radiologist.**

## Overview

FractAI is a computer vision project that uses a fine-tuned **YOLOv8-Nano** model to detect and localize potential fractures in musculoskeletal X-ray images.

The project provides an interactive Streamlit application for image upload, fracture localization, confidence visualization, and model performance analysis.

## Features

- Upload JPG, JPEG, and PNG X-ray images.
- Detect and localize potential fracture regions.
- Display bounding boxes, class labels, and confidence scores.
- Generate annotated images for download.
- View model performance metrics and evaluation visualizations.
- Explore the detection workflow through the application.

## Technology Stack

- **Language:** Python
- **Deep Learning:** PyTorch, Ultralytics YOLOv8n
- **Image Processing:** OpenCV, Pillow, NumPy
- **Data Analysis:** Pandas, scikit-learn
- **Visualization:** Matplotlib, Plotly
- **Frontend:** Streamlit
- **Training Environment:** Google Colab

## Project Structure

```text
├── app.py
├── pages/
├── src/
├── training/
│   └── notebooks/
├── models/
│   └── best.pt
├── results/
├── config/
├── assets/
├── tests/
├── requirements.txt
├── conftest.py
└── README.md
```

## Model Training

The YOLOv8-Nano model was trained using Google Colab with GPU acceleration.

The training workflow includes:

1. Dataset extraction and preparation.
2. Annotation verification and preprocessing.
3. Train, validation, and test split preparation.
4. YOLOv8-Nano fine-tuning.
5. Model evaluation and metric generation.
6. Saving the best checkpoint as `models/best.pt`.

The training notebook is available in:

`training/notebooks/`

## Model Evaluation

The project evaluates model performance using:

- Precision and Recall
- Sensitivity and Specificity
- mAP@50 and mAP@50-95
- Intersection over Union (IoU)
- Inference Latency

Evaluation results and generated visualizations are stored in `results/`.

## Run Locally

**1. Clone the repository**

```bash
git clone https://github.com/muskan170105/automated-bone-fracture-detection.git
cd automated-bone-fracture-detection
```

**2. Create and activate a virtual environment**

```bash
python -m venv venv
```

Windows:

```powershell
venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Run the application**

```bash
streamlit run app.py
```

The application opens at `http://localhost:8501`.

## Trained Model

The trained weights are stored at:

`models/best.pt`

The model path and inference defaults are configured in `config/config.yaml`.

## Deployment

The application is deployed using **Streamlit Community Cloud**.

The deployed application uses the repository's `app.py` entry point and the trained model weights.

## Limitations

- The model is trained on a limited dataset.
- External clinical validation has not been performed.
- Annotation quality can affect detection performance.
- Hairline fracture detection is not guaranteed.
- Absence of a predicted fracture does not mean the X-ray is fracture-free.

## Future Scope

- Expand and improve the training dataset.
- Perform external validation.
- Improve detection of subtle fractures.
- Explore larger detection architectures.
- Support DICOM medical imaging.

## Medical Disclaimer

FractAI is intended strictly for research and educational purposes. It is not a medical device and must not be used for diagnosis, treatment, or clinical decision-making. Every image must be reviewed by a qualified radiologist.