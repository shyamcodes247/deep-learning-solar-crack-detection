from pathlib import Path
import numpy as np
import cv2
import shutil

# Path to the dataset you downloaded from Kaggle, SolarCells and its subsets
DATASET_ROOT = Path("data")  

#simplify the path, this:
#"C:\\Users\\HP\\Desktop\\project\\data"
#becomes this:
#Path("data")


# Where to save classification dataset
OUT = Path("data")

#can sub in split as train or test
def prepare_split(split):
    img_dir = DATASET_ROOT / split / "defect"
    mask_dir = DATASET_ROOT / split / "label"

    print("Checking split:", split)
    print("Looking in:", img_dir)

    #for each image->create mask path->
    for ext in ["*.jpg", "*.jpeg", "*.png"]:

        for img_path in img_dir.glob(ext):
            print("Found image:", img_path.name)

            # --------------------------------------
            # FIXED: handle same-name masks (1571.jpg)
            # --------------------------------------
            # FIRST try matching exactly the same name (e.g. 1571.jpg)
            mask_candidates = list(mask_dir.glob(img_path.stem + ".*"))

            # ensure mask file matches SAME name and valid extension
            mask_candidates = [
                m for m in mask_candidates
                if m.suffix.lower() in [".jpg", ".jpeg", ".png"] 
                and m.name == img_path.name
            ]

            if len(mask_candidates) == 0:
                #print("WARNING: No mask found for", img_path.name)
                continue

            mask_path = mask_candidates[0]
            # --------------------------------------

            # Load mask, read mask, load in grayscale
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

            # Decide classification label, below 10 might be noise
            label = "1_crack" if np.any(mask > 10) else "0_no_crack"

            # Create output folder
            out_dir = OUT / split / label
            out_dir.mkdir(parents=True, exist_ok=True)

            # Copy image
            shutil.copy(img_path, out_dir / img_path.name)

def main():
    prepare_split("train")
    prepare_split("test")

if __name__ == "__main__":
    main()
