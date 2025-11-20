from pathlib import Path
import shutil

# Paths
test_defect = Path("data/test/defect")
test_label = Path("data/test/label")

# Starting number for new names
START = 2000  

# Get sorted list of defect images
images = sorted(test_defect.glob("*.jpg"))

for i, img_path in enumerate(images):
    stem = f"{START + i}"
    new_img_name = stem + ".jpg"
    new_mask_name = stem + ".jpg"

    # old label path (same filename as image)
    old_mask_path = test_label / img_path.name

    # new paths
    new_img_path = test_defect / new_img_name
    new_mask_path = test_label / new_mask_name

    # rename image
    img_path.rename(new_img_path)


print("Renaming complete!")
