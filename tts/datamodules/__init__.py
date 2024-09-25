import pytorch_lightning as pl

from . import FastSpeech2DataModule
from . import FastSpeech2UnitDataModule
from . import Tacotron2DataModule


DATA_MODULE = {
    "fastspeech2-unit": FastSpeech2UnitDataModule.FastSpeech2DataModule,
    "fastspeech2": FastSpeech2DataModule.FastSpeech2DataModule,
    "tacotron2": Tacotron2DataModule.Tacotron2DataModule,
}

def get_datamodule(system_type: str) -> pl.LightningDataModule:
    return DATA_MODULE[system_type]
