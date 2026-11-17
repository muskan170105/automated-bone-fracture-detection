# models/

Place the trained detector here as **`best.pt`**.

The Colab notebook creates and copies it automatically after training (Section 5). If you trained elsewhere,
upload your file to this folder (keeping the name `best.pt`) or change `model.weights` in `config/config.yaml`.

Until `best.pt` exists, the website runs in *setup mode* and never shows fake predictions.
