import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset

import Define
from tts.collates import LanguageCollate, TextCollate
from tts.datasets.FastSpeech2Dataset import FastSpeech2Dataset
from tts.datasets.TextDataset import TextDataset
from .utils import EpisodicInfiniteWrapper


class FastSpeech2DataModule(pl.LightningDataModule):
    """
    Train: FastSpeech2Dataset + LanguageCollate.
    Val: FastSpeech2Dataset + LanguageCollate.
    Test: TextDataset + TextCollate.
    """
    def __init__(self, data_configs, train_config, algorithm_config, log_dir, result_dir):
        super().__init__()
        self.data_configs = data_configs
        self.train_config = train_config
        self.algorithm_config = algorithm_config

        self.log_dir = log_dir
        self.result_dir = result_dir
        self.val_step = self.train_config["step"]["val_step"]

        self.re_id = True  # TODO: should be controlled by client directly
        self.collate = LanguageCollate()
        self.text_collate = TextCollate()

    def setup(self, stage=None):

        if stage in (None, 'fit', 'validate'):
            self.train_datasets = [
                FastSpeech2Dataset(
                    data_config['subsets']['train'],
                    Define.DATAPARSERS[data_config["name"]],
                    data_config
                ) for data_config in self.data_configs if 'train' in data_config['subsets']
            ]
            self.val_datasets = [
                FastSpeech2Dataset(
                    data_config['subsets']['val'],
                    Define.DATAPARSERS[data_config["name"]],
                    data_config
                ) for data_config in self.data_configs if 'val' in data_config['subsets']
            ]
            self.train_dataset = ConcatDataset(self.train_datasets)
            self.val_dataset = ConcatDataset(self.val_datasets)
            self._train_setup()
            self._validation_setup()

        if stage in (None, 'test', 'predict'):
            self.test_datasets = [
                TextDataset(
                    data_config['subsets']['test'],
                    Define.DATAPARSERS[data_config["name"]],
                    data_config
                ) for data_config in self.data_configs if 'test' in data_config['subsets']
            ]
            self.test_dataset = ConcatDataset(self.test_datasets)
            self._test_setup()

    def _train_setup(self):
        if not isinstance(self.train_dataset, EpisodicInfiniteWrapper):
            self.batch_size = self.train_config["optimizer"]["batch_size"]
            self.train_dataset = EpisodicInfiniteWrapper(self.train_dataset, self.val_step*self.batch_size)

    def _validation_setup(self):
        pass

    def _test_setup(self):
        pass

    def train_dataloader(self):
        """Training dataloader"""
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=Define.MAX_WORKERS,
            collate_fn=self.collate.collate_fn(sort=False)
        )
        return self.train_loader

    def val_dataloader(self):
        """Validation dataloader"""
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            collate_fn=self.collate.collate_fn(sort=False),
        )
        return self.val_loader

    def test_dataloader(self):
        """Test dataloader"""
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.text_collate.collate_fn(sort=False),
        )
        return self.test_loader
