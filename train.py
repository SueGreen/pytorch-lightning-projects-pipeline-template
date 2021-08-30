import argparse
import json
from pathlib import Path

import pytorch_lightning as pl

from models.cv_classification_base_model import BaseComputerVisionClassificationModel, \
    BaseComputerVisionClassificationDataModule


class PipelineTrainer():
    def __init__(self, experiment_id=None, experiment_name=None):
        self.experiment_id = experiment_id if experiment_id else 0
        self.experiment_name = experiment_name

    def parse_config(self, config_filename):
        config_file = Path(config_filename).open(mode='r')
        # todo: modify for yaml file instead of json
        config_content = json.loads(config_file.read())

        data_params = config_content.get('data_params')
        self.data_dir = data_params.get('data_dir')
        self.img_dim = data_params.get('img_dim')
        self.batch_size = data_params.get('batch_size')

        train_params = config_content.get('train_params')
        self.random_seed = train_params.get('random_seed')
        self.gpus = train_params.get('gpus')
        self.max_epochs = train_params.get('max_epochs')
        self.limit_train_batches = train_params.get('limit_train_batches')
        self.limit_val_batches = train_params.get('limit_val_batches')
        self.log_every_n_steps = train_params.get('log_every_n_steps')

    def train(self, config_filename):
        self.parse_config(config_filename)
        pl.seed_everything(self.random_seed)

        # todo: set customizable model and data module from config
        base_classification_data_module = BaseComputerVisionClassificationDataModule(
            data_dir=self.data_dir, batch_size=self.batch_size, img_dim=self.img_dim)
        base_classification_data_module.prepare_data()
        base_classification_data_module.setup()
        base_classification_data_module.visualize_augmentations(
            base_classification_data_module.train_dataset)
        base_classifier = BaseComputerVisionClassificationModel(
            base_classification_data_module.num_classes,
            img_dim=self.img_dim,
            data_module=base_classification_data_module)
        model_name = base_classifier.model_name
        trainer = pl.Trainer(gpus=self.gpus,
                             log_every_n_steps=self.log_every_n_steps,
                             limit_train_batches=self.limit_train_batches,
                             limit_val_batches=self.limit_val_batches,
                             max_epochs=self.max_epochs,
                             default_root_dir=
                             f'checkpoints/{model_name}/{self.experiment_id}/lightning_logs')

        trainer.fit(base_classifier, base_classification_data_module)
        test_metrics = trainer.test(base_classifier, base_classification_data_module)
        preds = trainer.predict(model=base_classifier,
                                datamodule=base_classification_data_module,
                                return_predictions=True)
        base_classification_data_module.visualize_classification_results(preds)


if __name__ == "__main__":
    # Parse command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", dest="config",
                        help="Absolute path to configuration file.")
    args = parser.parse_args()

    # Ensure a config was passed to the script.
    if not args.config:
        print("No configuration file provided.")
        exit()
    else:
        PipelineTrainer().train(args.config)
