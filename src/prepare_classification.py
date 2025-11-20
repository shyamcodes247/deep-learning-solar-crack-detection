from pathlib import Path
import numpy as np
import cv2
import shutil

# Path to the dataset you downloaded from Kaggle, SolarCells and its subsets
DATASET_ROOT = Path("data/SolarCells")   # change to SolarCells, SolarCells-S, PVEL-S

#simplify the path, this:
#"C:\\Users\\HP\\Desktop\\project\\data\\SolarCells"
#becomes this:
#Path("data/SolarCells")


# Where to save classification dataset
OUT = Path("data/classification")

#can sub in split as train or test
def prepare_split(split):
    img_dir = DATASET_ROOT / split / "defect"
    mask_dir = DATASET_ROOT / split / "label"

    #for each image->create mask path->
    for img_path in img_dir.glob("*.jpg"):
        #mask_path = mask_dir / img_path.name, this is if image name is the same

        #Find mask with pattern "<name>_mask.*"
        mask_candidates = list(mask_dir.glob(img_path.stem + "_mask.*"))
        if len(mask_candidates) == 0:
            print("WARNING: No mask found for", img_path.name)
            continue

        mask_path = mask_candidates[0]

        # Load mask, read mask, load in grayscale
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        # Decide classification label, below 10 might be noise
        label = "crack" if np.any(mask > 10) else "no_crack"

        # Create output folder
        out_dir = OUT / split / label
        out_dir.mkdir(parents=True, exist_ok=True) #create parent directories if they don't exist (for first image0, if exist then continue)

        # Copy image
        shutil.copy(img_path, out_dir / img_path.name)

def main():
    prepare_split("train")
    prepare_split("test")

if __name__ == "__main__":
    main()
