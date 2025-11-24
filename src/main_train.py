import torch
from pre_process import get_dataloaders
from model import build_model
from train import train_model
from pathlib import Path 

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"

def main():
    # Decide whether to use GPU (cuda) or CPU for training.
    # If a CUDA-capable GPU is available, we use it; otherwise we fall back to CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Create DataLoaders for training, validation, and test sets.
    # - data_root points to the folder containing 'train/', 'val/', 'test/' subfolders.
    # - batch_size is how many images per batch.
    # - num_workers is how many subprocesses to use for data loading.
    train_loader, val_loader, test_loader, class_to_idx = get_dataloaders(
        data_root=DATA_ROOT,
        batch_size=32,
        num_workers=2,
    )
    
    # Print the mapping from class names to numeric labels, e.g. {'crack': 0, 'no_crack': 1}
    print("Class mapping:", class_to_idx)

    # Build the ResNet18 model (optionally with pretrained ImageNet weights).
    # The build_model function will also replace the final layer for binary classification.
    model = build_model(pretrained=False)
    

    # Train the model using the training and validation DataLoaders.
    # - device: CPU or GPU to run on
    # - num_epochs: how many passes over the training dataset
    # - lr: learning rate for the optimizer
    # 'history' typically stores training/validation losses across epochs.
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=10,
        lr=0.035,  # chosen to match the paper's experimental setup
    )

    # (Optional) you could later evaluate on test_loader here, or save history/model, etc.


# This ensures that main() only runs when this script is executed directly,
# and not when it is imported as a module from somewhere else.
if __name__ == "__main__":
    main()