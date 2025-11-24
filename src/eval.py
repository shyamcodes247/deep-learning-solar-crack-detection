import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

def evaluate_model(model, test_loader, device):
    # Move model to the correct device (CPU or GPU)
    model.to(device)
    # Put the model into evaluation mode:
    # - turns off dropout
    # - uses running stats for batch norm instead of updating them
    model.eval()

    # Lists to store all true labels and predicted probabilities
    all_labels = []
    all_probs  = []

    # Disable gradient calculation since we're only doing inference (no training)
    with torch.no_grad():
        # Loop over all batches in the test set
        for images, labels in test_loader:
            # Move images to the device (CPU/GPU)
            images = images.to(device)

            # Forward pass: get raw outputs (logits) from the model
            logits = model(images)

            # Apply sigmoid to convert logits → probabilities between 0 and 1
            # Then move to CPU, convert to NumPy array, and flatten to 1D
            probs = torch.sigmoid(logits).cpu().numpy().ravel()

            # Store predicted probabilities for this batch
            all_probs.extend(probs)
            # Store true labels (still as NumPy, from CPU)
            all_labels.extend(labels.numpy())

    # Convert lists into NumPy arrays for easier processing with sklearn
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)
    

    # ------------------------------
    # Thresholding to get class labels
    # ------------------------------
    # Use default threshold = 0.5:
    # prob >= 0.5 → class 1, prob < 0.5 → class 0
    preds = (all_probs >= 0.5).astype(int)

    print(all_labels)
    print(preds)
    
    # ------------------------------
    # Confusion matrix and metrics
    # ------------------------------
    # confusion_matrix returns a 2x2 matrix for binary classification:
    # [[tn, fp],
    #  [fn, tp]]
    tn, fp, fn, tp = confusion_matrix(all_labels, preds).ravel()

    # Overall accuracy: (correct predictions / total samples)
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # Sensitivity (a.k.a. recall for the positive class):
    # fraction of actual positives (cracks) that were correctly predicted
    sensitivity = tp / (tp + fn)

    # Specificity: fraction of actual negatives (no-cracks)
    # that were correctly predicted
    specificity = tn / (tn + fp)

    # Print metrics as percentages where appropriate
    print(f"Accuracy   : {accuracy*100:.2f}%")
    print(f"Sensitivity: {sensitivity*100:.2f}%")
    print(f"Specificity: {specificity*100:.2f}%")

    # ------------------------------
    # ROC curve and AUC
    # ------------------------------
    # ROC curve: show tradeoff between TPR and FPR at different thresholds
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    # AUC: area under the ROC curve (higher is better, max = 1)
    roc_auc = auc(fpr, tpr)
    print(f"AUC: {roc_auc:.3f}")

    # ------------------------------
    # Plot ROC curve and save as image
    # ------------------------------
    plt.figure()
    # Plot the ROC curve
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
    # Plot a diagonal line representing random guessing
    plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Solar Cell Crack Classification")
    plt.legend(loc="lower right")
    # Save ROC plot to a PNG file
    plt.savefig("roc_curve.png", dpi=200)
    plt.close()

    # Return metrics in a dictionary for later use
    return {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "auc": roc_auc,
        "confusion": (tn, fp, fn, tp)
    }