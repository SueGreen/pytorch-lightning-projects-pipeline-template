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
from skimage import io

class ComputerVisionCLassificationBaseDataset(Dataset):
    def __init__(self, images_path, df, class2idx, transform=None):
        self.images_path = images_path
        self.df = df
        self.class2idx = class2idx
        self.transform = transform

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        image_filepath = str(self.images_path / self.df.iloc[idx]['filename'])
        # image = cv2.imread(image_filepath)
        image = io.imread(image_filepath)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = self.df.iloc[idx]['label']
        label = self.class2idx[label]
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image, label














