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
from data_modules.cv_classification_base_datamodule import BaseComputerVisionClassificationDataModule

class BaseComputerVisionClassificationModel(pl.LightningModule):
    def __init__(self, num_classes, img_dim=(3, 256, 256),
                 learning_rate=0.001, batch_size=4,
                 data_module=None):
        super().__init__()
        self.model_name = 'base_computer_vision_classification'
        self.img_dim = img_dim # channels, height, width
        self.num_classes = num_classes
        self.set_architecture()

        self.train_accuracy = pl.metrics.Accuracy()
        self.val_accuracy = pl.metrics.Accuracy()
        self.test_accuracy = pl.metrics.Accuracy()
        self.test_predictions = []

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.save_hyperparameters()
        self.data_module = data_module

        self.cur_stage = 'train'

    def set_architecture(self):
        self.layer_1 = nn.Linear(self.img_dim[0] * self.img_dim[1] * self.img_dim[2], 10)
        self.layer_2 = nn.Linear(10, 10)
        self.layer_3 = nn.Linear(10, self.num_classes)


    def forward(self, x):
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, -1)

        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        x = F.relu(x)
        x = self.layer_3(x)

        x = F.log_softmax(x, dim=1)
        return x

    def on_predict_start(self) -> None:
        self.cur_stage = 'predict'

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        if isinstance(batch, list) and len(batch) == 2:
            x, y = batch
        else:
            x = batch
        logits = self(x)
        pred_classes_ids = torch.argmax(logits, dim=1)
        pred_classes = [self.data_module.idx2class.get(int(p.cpu().numpy()))
                        for p in pred_classes_ids]

        self.test_predictions.append(pred_classes)
        return pred_classes

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.calculate_loss(logits, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.calculate_loss(logits, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.calculate_loss(logits, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def calculate_loss(self, logits, labels):
        return F.nll_loss(logits, labels)



















