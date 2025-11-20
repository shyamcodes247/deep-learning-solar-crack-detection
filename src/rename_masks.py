from pathlib import Path

# Folders
test_defect = Path("data/test/defect")
test_label  = Path("data/test/label")

# Get sorted lists
defect_files = sorted(test_defect.glob("*.jpg"))
label_files  = sorted(test_label.glob("*.jpg"))

# Safety check
if len(defect_files) != len(label_files):
    print("ERROR: Number of defect and label files DOES NOT MATCH.")
    print("Defects:", len(defect_files))
    print("Labels:", len(label_files))
    raise SystemExit

# Rename
for defect, mask in zip(defect_files, label_files):
    new_name = defect.name  # match defect file name
    new_path = test_label / new_name
    print("Renaming", mask.name, "to", new_name)
    mask.rename(new_path)

print("DONE! All masks renamed to match defect images.")
