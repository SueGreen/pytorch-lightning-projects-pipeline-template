import argparse
from pathlib import Path

import pytorch_lightning as pl
from yaml import load

from models.cv_classification_base_model import BaseComputerVisionClassificationModel
from data_modules.cv_classification_base_datamodule import BaseComputerVisionClassificationDataModule


# from models.nlp_categorization_base_model_old_boy import BaseNLPClassificationModel, BaseNLPClassificationDataModule, \
#     NLPCLassificationBaseEmbeddingsDataset


class PipelineTrainer():
    def __init__(self, experiment_id=None, experiment_name=None):
        self.experiment_id = experiment_id if experiment_id else 0
        self.experiment_name = experiment_name
        self.available_model_names = ['nlp_classification_base', 'cv_classification_base']
        # self.models = [BaseComputerVisionClassificationModel, BaseNLPClassificationModel]
        self.models = [BaseComputerVisionClassificationModel]

    def parse_config(self, config_filename):
        config_path = Path(config_filename)
        config_content = load(open(config_path))

        model_params = config_content.get('model')
        self.model_name = model_params.get('name')

        data_params = config_content.get('data_params')
        self.data_dir = data_params.get('data_dir')
        self.dataframe_filename = data_params.get('dataframe_filename')
        self.images_dir = data_params.get('images_dir')
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
        data_module = BaseComputerVisionClassificationDataModule(
            data_dir=self.data_dir, dataframe_filename=self.dataframe_filename, images_dir=self.images_dir,
            batch_size=self.batch_size, img_dim=self.img_dim)
        data_module.prepare_data()
        data_module.setup()

        data_module.visualize_augmentations(
            data_module.train_dataset)

        model = BaseComputerVisionClassificationModel(
            data_module.num_classes,
            img_dim=self.img_dim,
            data_module=data_module)
        trainer = pl.Trainer(gpus=self.gpus,
                             log_every_n_steps=self.log_every_n_steps,
                             limit_train_batches=self.limit_train_batches,
                             limit_val_batches=self.limit_val_batches,
                             max_epochs=self.max_epochs,
                             default_root_dir=
                             f'checkpoints/{self.model_name}/')

        trainer.fit(model, data_module)
        test_metrics = trainer.test(model, data_module)
        predictions = trainer.predict(model=model,
                                      datamodule=data_module,
                                      return_predictions=True)
        data_module.visualize_classification_results(predictions)


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
