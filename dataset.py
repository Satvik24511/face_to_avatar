from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np

class ImageDataset(Dataset):
    def __init__(self, root_A, root_B, transform=None):
        self.root_A = root_A
        self.root_B = root_B
        self.transform = transform

        self.images_A = os.listdir(root_A)
        self.images_B = os.listdir(root_B)
        self.length_dataset = max(len(self.images_A), len(self.images_B))
        self.len_A = len(self.images_A)
        self.len_B = len(self.images_B)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, idx):
        idx_A = idx % self.len_A
        idx_B = idx % self.len_B

        image_path_A = os.path.join(self.root_A, self.images_A[idx_A])
        image_path_B = os.path.join(self.root_B, self.images_B[idx_B])

        image_A = np.array(Image.open(image_path_A).convert("RGB"))
        image_B = np.array(Image.open(image_path_B).convert("RGB"))

        if self.transform:
            augmentation = self.transform(image=image_A, image0=image_B)
            image_A = augmentation["image"]
            image_B = augmentation["image0"]

        return image_A, image_B
        