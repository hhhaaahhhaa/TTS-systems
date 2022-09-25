import pytorch_lightning as pl

from . import FastSpeech2System


SYSTEM = {
    "fastspeech2": FastSpeech2System.FastSpeech2System,
}

def get_system(system_type: str) -> pl.LightningModule:
    return SYSTEM[system_type]
