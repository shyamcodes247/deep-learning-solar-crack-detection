"""
============================================================
TRAINING SCRIPT FOR SOLAR CELL CRACK CLASSIFICATION MODEL
------------------------------------------------------------
This file defines a function `train_model()` which trains a
binary classification model (crack / no crack) using:

- BCEWithLogitsLoss (best for binary classification)
- Adam optimizer
- Train and validation dataloaders
- GPU or CPU (device)

It also:
  - Tracks train and validation loss
  - Saves the best model (lowest val loss) to best_model.pt
  - Returns a history dictionary for plotting loss curves

This script is designed to be extremely easy to understand,
even for beginners or new collaborators.

============================================================
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm


def train_model(model, train_loader, val_loader, device,
                num_epochs=35, lr=0.035):
    """
    Trains a binary classification model using PyTorch.

    Parameters:
    - model:        The neural network (e.g., ResNet18 with 1 output neuron)
    - train_loader: DataLoader for training data
    - val_loader:   DataLoader for validation data
    - device:       "cuda" or "cpu"
    - num_epochs:   How many passes through the full dataset
    - lr:           Learning rate for Adam optimizer

    Returns:
    - history: A dictionary containing training/validation loss history
    """

    # Loss function for binary classification.
    # BCEWithLogitsLoss = Sigmoid + BCE combined (more stable).
    criterion = nn.BCEWithLogitsLoss()

    # Adam optimizer updates model weights using gradients
    optimizer = Adam(model.parameters(), lr=lr)

    # Move the model to GPU or CPU
    model.to(device)

    # Used to track the best validation loss so we can save the best model
    best_val_loss = float("inf")

    # Dictionary for plotting loss curves later
    history = {"train_loss": [], "val_loss": []}

    # -------------------------------
    #         TRAINING LOOP
    # -------------------------------
    for epoch in range(1, num_epochs + 1):

        # --- TRAINING MODE ---
        model.train()                      # enables dropout, batchnorm updates, etc.
        running_train_loss = 0.0           # accumulate loss across batches

        # tqdm = progress bar for cleaner terminal output
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [train]"):

            # Move data to GPU/CPU
            images = images.to(device)
            labels = labels.float().to(device)   # labels must be float for BCE
            labels = labels.unsqueeze(1)         # reshape (batch,) → (batch,1)

            optimizer.zero_grad()                # clear previous gradients

            logits = model(images)               # model outputs raw scores (logits)
            loss = criterion(logits, labels)     # compute loss

            loss.backward()                      # compute gradients
            optimizer.step()                     # update weights

            # Add batch loss → total epoch loss
            running_train_loss += loss.item() * images.size(0)

        # Average training loss for the entire epoch
        epoch_train_loss = running_train_loss / len(train_loader.dataset)

        # --- VALIDATION MODE ---
        model.eval()                             # disables dropout, etc.
        running_val_loss = 0.0

        # No gradient computation during validation (saves memory + speed)
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [val]"):

                images = images.to(device)
                labels = labels.float().to(device).unsqueeze(1)

                logits = model(images)
                loss = criterion(logits, labels)

                running_val_loss += loss.item() * images.size(0)

        # Average validation loss
        epoch_val_loss = running_val_loss / len(val_loader.dataset)

        # Save loss history for graphing later
        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)

        # Print progress for this epoch
        print(f"Epoch {epoch}: train_loss={epoch_train_loss:.4f}, val_loss={epoch_val_loss:.4f}")

        # --- SAVE BEST MODEL ---
        # We save the model with the LOWEST validation loss.
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), "best_model.pt")
            print("  → Saved new best model")

    return history
