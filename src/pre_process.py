from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path

# For resizing image to 300 pixels
IMG_SIZE = 300

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"

# Custom transform class for per-image normalization (can be pickled)
class PerImageNormalize:
    """Normalize each image independently based on its own statistics.
    This removes brightness bias and forces the model to learn patterns."""
    def __call__(self, tensor):
        mean = tensor.mean()
        std = tensor.std()
        # Normalize: (x - mean) / std
        # Add small epsilon to avoid division by zero
        return (tensor - mean) / (std + 1e-8)

# Function for transforming/preparing data for training
def transform_data(): 
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        # Add brightness/contrast augmentation to prevent brightness shortcut
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2),
        transforms.ToTensor(),
        # Per-image normalization removes brightness bias
        PerImageNormalize(),
    ])
    
    eval_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        # Use same per-image normalization for evaluation
        PerImageNormalize(),
    ])

    return train_transform, eval_transform

# Function for getting and loading data for training
def get_dataloaders(data_root=DATA_ROOT, batch_size=32, num_workers=2):
    print("Using data_root:", data_root)  # debug print
    
    train_transform, eval_transform = transform_data()

    train_dataset = datasets.ImageFolder(f"{data_root}/train", transform=train_transform)
    val_dataset   = datasets.ImageFolder(f"{data_root}/val",   transform=eval_transform)
    test_dataset  = datasets.ImageFolder(f"{data_root}/test",  transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)

    return train_loader, val_loader, test_loader, train_dataset.class_to_idx
