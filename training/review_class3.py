
from pathlib import Path
from PIL import Image, ImageDraw
import math

root = Path("raw_dataset/BoneFractureYolo8")
label_dir = root / "train" / "labels"
image_dir = root / "train" / "images"

output = Path("training/class3_review.png")

images = []

for label_file in label_dir.glob("*.txt"):
    lines = label_file.read_text(encoding="utf-8").splitlines()

    for line in lines:
        parts = line.split()

        if not parts or parts[0] != "3":
            continue

        image_path = next(
            (image_dir / f"{label_file.stem}{ext}"
             for ext in [".jpg", ".jpeg", ".png", ".JPG", ".PNG"]
             if (image_dir / f"{label_file.stem}{ext}").exists()),
            None
        )

        if image_path is None:
            print("Image not found:", label_file.stem)
            continue

        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        coords = list(map(float, parts[1:]))
        points = [
            (coords[i] * img.width, coords[i + 1] * img.height)
            for i in range(0, len(coords), 2)
        ]

        draw.polygon(points, outline="red", width=3)
        img.thumbnail((450, 350))

        images.append((label_file.name, img.copy()))

if not images:
    print("No class 3 images found.")
    raise SystemExit

cell_w, cell_h = 480, 400
cols = 2
rows = math.ceil(len(images) / cols)

sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
draw = ImageDraw.Draw(sheet)

for i, (name, img) in enumerate(images):
    x = (i % cols) * cell_w
    y = (i // cols) * cell_h

    sheet.paste(img, (x, y + 35))
    draw.text((x + 5, y + 5), name[:55], fill="black")

sheet.save(output)
print("Saved:", output.resolve())
