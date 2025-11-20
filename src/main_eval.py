import torch
from pre_process import get_dataloaders
from model import build_model
from eval import evaluate_model
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"

def main():
    # Select the best available compute device:
    # - GPU ("cuda") if available 
    # - otherwise fallback to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load ONLY the test dataloader because training and validation aren't needed anymore.
    # The underscores (`_`) indicate that train_loader and val_loader are intentionally ignored.
    _, _, test_loader, class_to_idx = get_dataloaders(
        data_root=DATA_ROOT,      # Root folder where dataset is stored
        batch_size=32,         # Number of images passed at once during evaluation
        num_workers=2,         # Parallel workers for faster data loading
    )

    # Create the model architecture (same as used during training).
    # pretrained=False means we do NOT load ImageNet weights here, 
    # because we will immediately load our trained weights from file.
    model = build_model(pretrained=False)

    # Load the trained model weights from disk (best checkpoint saved during training).
    # map_location=device ensures model loads on the correct hardware (CPU/GPU)
    model.load_state_dict(torch.load("best_model.pt", map_location=device))

    # Run full evaluation on the test dataset:
    # - computes accuracy, specificity, sensitivity, ROC, and confusion matrix
    metrics = evaluate_model(model, test_loader, device)

    # Print the metrics dictionary for inspection or logging
    print(metrics)


# This ensures main() only runs if the script is executed directly,
# not when it is imported in another module.
if __name__ == "__main__":
    main()
