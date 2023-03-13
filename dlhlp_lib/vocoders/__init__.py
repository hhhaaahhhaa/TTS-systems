from typing import Type

from .vocoders import *


VOCODERS = {
    "GriffinLim": GriffinLim,
    "MelGAN": MelGAN,
    "HifiGAN": HifiGAN, 
}

def get_vocoder(system_type: str) -> Type[BaseVocoder]:
    return VOCODERS[system_type]
