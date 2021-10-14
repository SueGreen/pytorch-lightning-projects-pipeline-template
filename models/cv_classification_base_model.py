import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F


class BaseComputerVisionClassificationModel(pl.LightningModule):
    def __init__(self, num_classes, img_dim=(3, 256, 256),
                 learning_rate=0.001, batch_size=4,
                 data_module=None):
        super().__init__()
        self.model_name = 'base_computer_vision_classification'
        self.img_dim = img_dim  # channels, height, width
        self.num_classes = num_classes
        self.set_architecture(from_scratch=True)

        self.loss = nn.CrossEntropyLoss()
        self.train_accuracy = pl.metrics.Accuracy()
        self.val_accuracy = pl.metrics.Accuracy()
        self.test_accuracy = pl.metrics.Accuracy()
        self.test_predictions = []

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.save_hyperparameters()
        self.data_module = data_module

        self.cur_stage = 'train'

    def set_architecture(self, from_scratch=False):
        if from_scratch:
            self.layer_1 = nn.Linear(self.img_dim[0] * self.img_dim[1] * self.img_dim[2], 10)
            self.layer_2 = nn.Linear(10, 10)
            self.layer_3 = nn.Linear(10, self.num_classes)

            self.layers = nn.Sequential(
                self.layer_1,
                nn.ReLU(),
                self.layer_2,
                nn.ReLU(),
                self.layer_3
            )
        else:
            pass  # todo

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, -1)
        logits = self.layers(x)
        return logits

    def step(self, batch, batch_idx, step_name):
        x, y = batch
        logits = self(x)
        loss = self.calculate_loss(logits, y)
        self.log(f'{step_name}_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, step_name='train')

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, step_name='val')

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, step_name='test')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def calculate_loss(self, logits, labels):
        return self.loss(logits, labels)

    def on_predict_start(self) -> None:
        self.cur_stage = 'predict'

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        if isinstance(batch, list) and len(batch) == 2:
            x, y = batch
        else:
            x = batch
        logits = self(x)
        probs = F.log_softmax(logits, dim=1)
        pred_classes_ids = torch.argmax(probs, dim=1)
        pred_classes = [self.data_module.idx2class.get(int(p.cpu().numpy()))
                        for p in pred_classes_ids]

        self.test_predictions.append(pred_classes)
        return pred_classes
