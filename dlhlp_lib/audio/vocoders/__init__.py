from .vocoders import *


VOCODERS = {
    "GriffinLim": GriffinLim,
    "MelGAN": MelGAN,
    "HifiGAN": HifiGAN, 
}

def get_vocoder(system_type: str) -> BaseVocoder:
    return VOCODERS[system_type]
