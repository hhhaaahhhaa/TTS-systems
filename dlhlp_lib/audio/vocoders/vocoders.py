import os
import numpy as np
import torch
import torch.nn as nn
import json
import math

from dlhlp_lib.Constants import MAX_WAV_VALUE

from .. import AUDIO_CONFIG
from ..stft import TacotronSTFT
from ..audio_processing import griffin_lim
from . import hifigan


class BaseVocoder(nn.Module):
    def __init__(self):
        super().__init__()

    def infer(self, mels, lengths=None, *args, **kwargs):
        pass


class GriffinLim(BaseVocoder):
    def __init__(self, stft: TacotronSTFT):
        super().__init__()
        self.stft = stft

    def infer(self, mels, lengths=None, n_iters=30, *args, **kwargs):
        # some numeric issue exists...
        self.cpu()
        wavs = []
        with torch.no_grad():
            for mel in mels:
                mel = mel.cpu()
                spectrogram = self.stft.mel2linear(mel)
                wav = griffin_lim(spectrogram.unsqueeze(0), stft_fn=self.stft.stft_fn, n_iters=n_iters)
                wav = torch.clip(wav, max=1, min=-1)
                wav = (wav.numpy() * MAX_WAV_VALUE).astype("int16")
                wavs.append(wav)
        
        for i in range(len(mels)):
            if lengths is not None:
                wavs[i] = wavs[i][: lengths[i]]
        return wavs


class MelGAN(BaseVocoder):
    def __init__(self):
        super().__init__()
        vocoder = torch.hub.load(
            "descriptinc/melgan-neurips", "load_melgan", "multi_speaker"
        )
        self.vocoder = vocoder.mel2wav
        self.vocoder.eval()

    def inverse(self, mel):
        with torch.no_grad():
            return self.vocoder(mel).squeeze(1)

    def infer(self, mels, lengths=None, *args, **kwargs):
        wavs = self.inverse(mels / math.log(10))
        wavs = torch.clip(wavs, max=1, min=-1)
        wavs = (wavs.cpu().numpy() * MAX_WAV_VALUE).astype("int16")
        wavs = [wav for wav in wavs]

        for i in range(len(mels)):
            if lengths is not None:
                wavs[i] = wavs[i][: lengths[i]]
        return wavs


class HifiGAN(BaseVocoder):
    def __init__(self):
        super().__init__()
        _current_dir = os.path.dirname(__file__)
        with open(f"{_current_dir}/hifigan/config.json", "r") as f:
            config = json.load(f)
        config = hifigan.AttrDict(config)
        vocoder = hifigan.Generator(config)
        ckpt = torch.load(f"{_current_dir}/hifigan/generator_universal.pth.tar")
        vocoder.load_state_dict(ckpt["generator"])
        vocoder.remove_weight_norm()

        self.vocoder = vocoder
        self.vocoder.eval()

    def inverse(self, mel):
        with torch.no_grad():
            return self.vocoder(mel).squeeze(1)

    def infer(self, mels, lengths=None, *args, **kwargs):
        wavs = self.inverse(mels)
        wavs = torch.clip(wavs, max=1, min=-1)
        wavs = (wavs.cpu().numpy() * MAX_WAV_VALUE).astype("int16")
        wavs = [wav for wav in wavs]

        for i in range(len(mels)):
            if lengths is not None:
                wavs[i] = wavs[i][: lengths[i]]
        return wavs
    