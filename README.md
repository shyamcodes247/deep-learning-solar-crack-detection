# Solar Cell Crack Detection — Image Segmentation with DeepLabv3

This project implements **image segmentation** to detect micro‑cracks in solar cells using a **DeepLabv3‑ResNet50** model.  
It replaces the earlier **binary image classification** approach with a **pixel‑wise segmentation model**, providing fine‑grained crack localization and far more meaningful evaluation.

---

## 🔄 Shift from Classification → Segmentation

Originally, this repository attempted to classify solar cell images as *cracked* or *not cracked*.  
This approach had major limitations:

- It could **not show where cracks occur**
- Subtle micro‑cracks often went undetected
- Labels were too coarse for real analysis

The project was rebuilt around **semantic segmentation**, allowing:

- Pixel‑accurate crack localization  
- Better interpretability and visualization  
- Stronger evaluation metrics (IoU, Dice, F1, etc.)  
- Improved real‑world applicability  

---


Each mask is a **binary image** (`0 = background`, `1 = crack`).

---

## 🧠 Model Architecture

The model is based on:

- **DeepLabv3 (ResNet‑50 backbone)**
- Single‑channel sigmoid output for binary segmentation
- Loss function: **BCE + Dice Loss**
- Pixel‑level thresholding at `0.5`
- Albumentations for preprocessing and augmentation

Training uses:

- **70%** training  
- **15%** validation  
- **15%** testing  

Splits are performed via **ID lists**, not separate folders.

---

## ⚙️ Training Pipeline

Key components of the final training setup:

- Consistent IoU computation using:

  ```python
  preds = (torch.sigmoid(logits) >= 0.5)

## FINAL METRICS
- IoU:        0.6003
- Dice:       0.5681
- Precision:  0.5482
- Recall:     0.6308
- F1 Score:   0.5681

# 🚀 Future Work
Potential extensions:
- Replace DeepLabv3 with U‑Net++, DeepLabv3+, or HRNet
- Add post‑processing (morphology, CRFs, contour smoothing)
- Introduce crack classification or severity scoring
- Use semi‑supervised learning to expand dataset
