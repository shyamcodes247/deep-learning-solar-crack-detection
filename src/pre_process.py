from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# For resizing image to 300 pixels
IMG_SIZE = 300

# Function for transforming/preparing data for training
def transform_data(): 
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        # optional augmentation:
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])
    
    eval_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    return train_transform, eval_transform

# Function for getting and loading data for training
def get_dataloaders(data_root="data/classification", batch_size=32, num_workers=2):
    train_transform, eval_transform = transform_data()

    train_dataset = datasets.ImageFolder(f"{data_root}/train", transform=train_transform)
    val_dataset   = datasets.ImageFolder(f"{data_root}/val",   transform=eval_transform)
    test_dataset  = datasets.ImageFolder(f"{data_root}/test",  transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, train_dataset.class_to_idx
