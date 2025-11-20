# deep-learning-solar-crack-detection
# 🪐 Solar Cell Crack Detection using Deep Learning (ResNet18 + PyTorch)

> Automatically detect micro-cracks in solar cells using AI.

This project uses a Convolutional Neural Network (CNN) built with **PyTorch** to classify electroluminescence (EL) solar cell images as either:

- 🟢 **No Defect**
- 🔴 **Contains Crack**

The model is trained using **transfer learning (ResNet18)** and evaluated using accuracy, sensitivity, specificity, confusion matrix, and ROC–AUC.

---

## 🔍 Motivation

Solar cracks are often subtle and hard to identify manually in EL images. Automated detection:

- Reduces inspection time  
- Helps maintain manufacturing quality  
- Detects hidden micro-fractures before panels are deployed  
- Prevents long-term efficiency loss in photovoltaic systems  

This project demonstrates how deep learning can support scalable, automated inspection.

---

## 📁 Dataset

Dataset: **Dataset of Solar Cells Defect Segmentation**  
Source: https://www.kaggle.com/datasets/yaozhang01182010/dataset-of-solar-cells-defect-segmentation

Directory structure:

data/
├── train/
│ ├── crack/
│ └── no_crack/
├── val/
│ ├── crack/
│ └── no_crack/
└── test/
├── crack/
└── no_crack/


---

## 🧠 Model Architecture

- ResNet-18 backbone (ImageNet pretrained)
- Final fully connected layer replaced for binary classification
- Loss: **Binary Cross Entropy with Logits (BCEWithLogitsLoss)**
- Optimizer: **Adam**
- Augmentations: resize, normalization, random horizontal flips

---

## ⚙️ Installation

Install dependencies:

pip install -r requirements.txt

python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

## 🚀 Training

Run the training script:

python main_train.py

The best-performing model checkpoint will be saved automatically as:

best_model.pt

## 🧪 Evaluation

Evaluate model performance on the unseen test dataset:

python eval.py

This command prints key metrics, including:

# Accuracy

# Sensitivity (recall for crack detection)

# Specificity

# ROC–AUC

Confusion matrix values

Example output:
Accuracy: 92.54%
Sensitivity: 89.73%
Specificity: 94.10%
AUC: 0.96

📊 Results
Metric	Score
Accuracy	0.92
Sensitivity	0.89
Specificity	0.94
AUC	0.96
ROC Curve
