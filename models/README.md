# Models

This folder contains the trained YOLOv8n fracture detection model used by FractAI.

## Model Weights

Place the trained detector here as **`best.pt`**.

The application loads the model from this location using the weights path configured in `config/config.yaml`.

## Training

The model was trained using Google Colab.

The cleaned training notebook is available at:

`training/notebooks/FractAI_YOLOv8n_Training_Clean.ipynb`

The notebook preserves the essential workflow:
- Dataset extraction and preparation
- Dataset validation
- YOLOv8n training
- Model verification
- Test-set evaluation
- Exporting the trained weights

After training, ensure the resulting `best.pt` file is placed in this folder.

If you trained the model elsewhere, upload your weights here using the filename `best.pt`, or update `model.weights` in `config/config.yaml`.

## Setup Mode

Until `best.pt` exists, the website runs in **setup mode** and never displays fake predictions.