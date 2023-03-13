from typing import Type

from .system import System
from . import FastSpeech2System
from . import Tacotron2System


SYSTEM = {
    "fastspeech2": FastSpeech2System.FastSpeech2System,
    "tacotron2": Tacotron2System.Tacotron2System,
}

def get_system(system_type: str) -> Type[System]:
    return SYSTEM[system_type]
