import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
import argparse
import json
import os
from pathlib import Path
import numpy as np
import math
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, random_split
# from torchvision.data import MNIST
import os
from torchvision import datasets, transforms
import torchvision.models as models
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
from datasets.computer_vision.cv_classification_base_dataset import ComputerVisionCLassificationBaseDataset

class BaseComputerVisionClassificationDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=4, train_val_ratio=0.8,
                 img_dim=(3, 256, 256), num_classes=None):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.train_val_ratio = train_val_ratio
        # todo: handle different-size input images with a better approach.
        #  some ideas present here: https://ai.stackexchange.com/a/14364
        self.img_dim = img_dim # channels, height, width
        if num_classes:
            self.num_classes = num_classes

        # todo: set augmentation parameters in the config file
        self.train_transforms = A.Compose([
            A.Resize(height=img_dim[1], width=img_dim[2]),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=30, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
        self.test_transforms = A.Compose([
            A.Resize(height=img_dim[1], width=img_dim[2]),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    def flatten_list(self, nested_list):
        flat_list = [item for sublist in nested_list for item in sublist]
        return flat_list

    def prepare_data(self):
        train_data_path = os.path.join(self.data_dir, 'train')
        test_data_path = os.path.join(self.data_dir, 'test')

        # todo: redo with pandas dataframes
        # todo: handle case when some classes are present only in test:
        #  (make classes[] a union of what is present in train and test)
        train_image_paths = []
        classes = []
        for data_path in glob.glob(train_data_path + '/*'):
            classes.append(data_path.split('/')[-1])
            train_image_paths.append(glob.glob(data_path + '/*'))

        train_image_paths = list(self.flatten_list(train_image_paths))
        random.shuffle(train_image_paths)
        self.train_image_paths = \
            train_image_paths[:int(self.train_val_ratio * len(train_image_paths))]
        self.valid_image_paths = \
            train_image_paths[int(self.train_val_ratio * len(train_image_paths)):]

        test_image_paths = []
        for data_path in glob.glob(test_data_path + '/*'):
            test_image_paths.append(glob.glob(data_path + '/*'))
        self.test_image_paths = list(self.flatten_list(test_image_paths))

        print(f'classes: {classes}')
        self.num_classes = len(classes)
        print(f'Train size: {len(self.train_image_paths)} '
              f'\nVal size: {len(self.valid_image_paths)} '
              f'\nTest size: {len(self.test_image_paths)}')

        self.idx2class = {i: j for i, j in enumerate(classes)}
        self.class2idx = {value: key for key, value in self.idx2class.items()}

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = ComputerVisionCLassificationBaseDataset(self.train_image_paths, self.class2idx, self.train_transforms)
            self.valid_dataset = ComputerVisionCLassificationBaseDataset(self.valid_image_paths, self.class2idx, self.test_transforms)
            self.test_dataset = ComputerVisionCLassificationBaseDataset(self.test_image_paths, self.class2idx, self.test_transforms)
        if stage == 'test' or stage == 'inference' or stage is None:
            self.test_dataset = ComputerVisionCLassificationBaseDataset(self.test_image_paths, self.class2idx, self.test_transforms)

    def visualize_augmentations(self, dataset, idx=0, n_samples=10, cols=5, random_img=False):
        """
        Function for visual check of applied augmentations
        :param dataset: The dataset to be shown. Dataset has a predefined set of augmentations
        :param idx: Index of an image in the dataset (for non-random image display)
        :param n_samples: Number of grid cells for images
        :param cols: Number of grid columns
        :param random_img: If True, display random augmented images.
                If False, display different augmentations of the same image specified by the idx
        """
        dataset_for_display = copy.deepcopy(dataset)
        dataset_for_display.transform = A.Compose(
            [t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))])

        rows = n_samples // cols
        figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 8))
        for i in range(n_samples):
            if random_img:
                idx = np.random.randint(1, len(dataset_for_display))
            image, label = dataset_for_display[idx]
            ax.ravel()[i].imshow(image)
            ax.ravel()[i].set_axis_off()
            ax.ravel()[i].set_title(self.idx2class[label])
        plt.tight_layout(pad=1)
        plt.show()


    def visualize_classification_results(self, predictions, dataset=None, init_idx=0, n_samples=10, cols=5, random_img=False):
        """
        Function for visual check of results
        :param dataset: The dataset to be shown. Dataset has a predefined set of augmentations
        :param idx: Index of an image in the dataset (for non-random image display)
        :param n_samples: Number of grid cells for images
        :param cols: Number of grid columns
        :param random_img: If True, display random augmented images.
                If False, display different augmentations of the same image specified by the idx
        """
        predictions = list(self.flatten_list(predictions))
        n_samples = min(n_samples, len(predictions))
        if dataset is None:
            dataset = self.train_dataset
        dataset_for_display = copy.deepcopy(dataset)
        dataset_for_display.transform = A.Compose(
            [t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))])

        rows = n_samples // cols
        figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 8))
        idx = init_idx
        for i in range(rows * cols):
            if random_img:
                idx = np.random.randint(1, len(dataset_for_display))
            image, true_label = dataset_for_display[idx]
            predicted_label = predictions[idx]
            ax.ravel()[i].imshow(image)
            ax.ravel()[i].set_axis_off()
            ax.ravel()[i].set_title(f'{predicted_label} ({self.idx2class[true_label]})')
            idx = init_idx + i
        plt.tight_layout(pad=1)
        plt.show()


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=4, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4)