import os
import random
from pathlib import Path
from PIL import Image

import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, img_size=(300,300), augment=False):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.ids = sorted([f for f in os.listdir(self.images_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))])
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        fname = self.ids[idx]
        img = Image.open(self.images_dir / fname).convert("RGB").resize(self.img_size)
        mask = Image.open(self.masks_dir / fname).convert("L").resize(self.img_size)
        image = TF.to_tensor(img)
        image = TF.normalize(image, mean=[0.5]*3, std=[0.5]*3)
        mask = TF.to_tensor(mask)
        mask = (mask > 0.5).float()
        if self.augment and random.random() > 0.5:
            image = TF.hflip(image); mask = TF.hflip(mask)
        return image, mask