from pathlib import Path
import shutil
import random

BASE = Path("data/classification")       
ORIGINAL_LABELS = Path("data/train/label") # where masks actually are

# Folders
train_crack = BASE / "train" / "1_crack"
train_noncrack = BASE / "train" / "0_no_crack"

val_crack = BASE / "val" / "1_crack"
val_noncrack = BASE / "val" / "0_no_crack"

# Create val folders
val_crack.mkdir(parents=True, exist_ok=True)
val_noncrack.mkdir(parents=True, exist_ok=True)

# ---- SELECT RANDOM IMAGES ----

crack_imgs = sorted(train_crack.glob("*.jpg")) #sorted so its reproducible
non_imgs = sorted(train_noncrack.glob("*.jpg"))

selected_crack = random.sample(crack_imgs, 130)
selected_non = random.sample(non_imgs, 70)

# ---- MOVE FUNCTION ----

def move_with_mask(img_path: Path, out_folder: Path):
    name = img_path.name
    stem = img_path.stem  # example: "0152"

    # mask stored as data/train/label/0152.jpg
    mask_path = ORIGINAL_LABELS / f"{stem}.jpg"

    # move image
    shutil.move(str(img_path), out_folder / name)

    # ---- MOVE MASK IF FOUND ----
    if mask_path.exists():
        # create val label folder on first use
        val_label = Path("data/val/label")
        val_label.mkdir(parents=True, exist_ok=True)

        shutil.move(str(mask_path), val_label / f"{stem}.jpg")

# ---- MOVE CRACK IMAGES ----
print("Moving 130 CRACK images...")
for img in selected_crack:
    move_with_mask(img, val_crack)

# ---- MOVE NON-CRACK IMAGES ----
print("Moving 70 NON-CRACK images...")
for img in selected_non:
    move_with_mask(img, val_noncrack)

print("Done!")
