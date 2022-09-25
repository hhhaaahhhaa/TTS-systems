import pytorch_lightning as pl

from . import FastSpeech2DataModule


DATA_MODULE = {
    "fastspeech2": FastSpeech2DataModule.FastSpeech2DataModule,
}

def get_datamodule(system_type: str) -> pl.LightningDataModule:
    return DATA_MODULE[system_type]
