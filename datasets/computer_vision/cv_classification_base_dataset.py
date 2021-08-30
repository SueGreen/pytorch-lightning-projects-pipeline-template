import copy
import glob
import random

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class ComputerVisionCLassificationBaseDataset(Dataset):
    def __init__(self, image_paths, class2idx, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        self.class2idx = class2idx

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = image_filepath.split('/')[-2]
        label = self.class2idx[label]
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image, label














