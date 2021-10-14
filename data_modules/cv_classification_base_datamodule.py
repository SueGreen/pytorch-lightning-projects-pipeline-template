import copy
from pathlib import Path

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from datasets.computer_vision.cv_classification_base_dataset import ComputerVisionCLassificationBaseDataset


class BaseComputerVisionClassificationDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, dataframe_filename, images_dir, batch_size=4, train_val_ratio=0.8,
                 img_dim=(3, 256, 256), num_classes=None):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = Path(data_dir)
        self.dataframe_path = Path(data_dir) / dataframe_filename
        self.images_path = self.data_dir / images_dir
        self.train_val_ratio = train_val_ratio
        self.img_dim = img_dim  # channels, height, width
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
        self.data_csv = pd.read_csv(self.dataframe_path)
        self.classes = self.data_csv['label'].unique()

        self.train_df, self.test_df = train_test_split(self.data_csv, test_size=0.2)
        self.train_df, self.val_df = train_test_split(self.train_df, test_size=0.2)

        # print(f'Classes: {self.classes}')
        self.num_classes = len(self.classes)
        # print(f'Train size: {len(self.train_df)} '
        #       f'\nVal size: {len(self.val_df)} '
        #       f'\nTest size: {len(self.test_df)}')

        # integer encoding
        self.idx2class = {i: j for i, j in enumerate(self.classes)}
        self.class2idx = {value: key for key, value in self.idx2class.items()}
        # print(f'Labels encoded: {self.class2idx}')

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = ComputerVisionCLassificationBaseDataset(self.images_path, self.train_df,
                                                                         self.class2idx, self.train_transforms)
            self.valid_dataset = ComputerVisionCLassificationBaseDataset(self.images_path, self.val_df, self.class2idx,
                                                                         self.test_transforms)
            self.test_dataset = ComputerVisionCLassificationBaseDataset(self.images_path, self.test_df, self.class2idx,
                                                                        self.test_transforms)
        if stage == 'test' or stage == 'inference' or stage is None:
            self.test_dataset = ComputerVisionCLassificationBaseDataset(self.images_path, self.test_df, self.class2idx,
                                                                        self.test_transforms)

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
        # todo: add possibility to run only on several images instead of the whole copy.deepcopy(dataset)
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

    def visualize_classification_results(self, predictions, dataset=None, init_idx=0, n_samples=10, cols=5,
                                         random_img=False):
        """
        Function for visual check of results
        :param dataset: The dataset to be shown. Dataset has a predefined set of augmentations
        :param idx: Index of an image in the dataset (for non-random image display)
        :param n_samples: Number of grid cells for images
        :param cols: Number of grid columns
        :param random_img: If True, display random augmented images.
                If False, display different augmentations of the same image specified by the idx
        """
        # todo: add possibility to run only on several images instead of the whole copy.deepcopy(dataset)
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
