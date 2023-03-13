import os
import numpy as np
import torch
import torch.nn as nn
import json
import math
from torchaudio.transforms import InverseMelScale
from torchaudio.transforms import GriffinLim as TorchGriffinLim

from dlhlp_lib.Constants import MAX_WAV_VALUE

from ..audio import AUDIO_CONFIG
from . import hifigan


class BaseVocoder(nn.Module):
    def __init__(self):
        super().__init__()

    def infer(self, mels, lengths=None, *args, **kwargs):
        pass


class GriffinLim(BaseVocoder):
    def __init__(self, n_iters=30):
        super().__init__()
        self.mel2linear_spec = InverseMelScale(
            n_stft=AUDIO_CONFIG["stft"]["filter_length"] // 2 + 1,
            n_mels=AUDIO_CONFIG["mel"]["n_mel_channels"],
            sample_rate=AUDIO_CONFIG["audio"]["sampling_rate"],
            norm="slaney",
            mel_scale="slaney"
        )
        self.linear_spec2wav = TorchGriffinLim(
            n_fft=AUDIO_CONFIG["stft"]["filter_length"],
            n_iter=n_iters,
            win_length=AUDIO_CONFIG["stft"]["win_length"],
            hop_length=AUDIO_CONFIG["stft"]["hop_length"],
            power=1
        )

    def inverse(self, mels):
        with torch.no_grad():
            specs = self.mel2linear_spec(mels)
            wavs = self.linear_spec2wav(specs)
            return wavs

    def infer(self, mels, lengths=None, *args, **kwargs):
        wavs = []
        wavs = self.inverse(mels.cpu())    
        wavs = torch.clip(wavs, max=1, min=-1)
        wavs = (wavs.cpu().numpy() * MAX_WAV_VALUE).astype("int16")
        wavs = [wav for wav in wavs]

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
        self.cpu()

    def inverse(self, mels):
        with torch.no_grad():
            return self.vocoder(mels).squeeze(1)

    def infer(self, mels, lengths=None, *args, **kwargs):
        # wavs = self.inverse(mels / math.log(10))  compatible with stft framework only, deprecated
        wavs = self.inverse(mels)
        wavs = torch.clip(wavs, max=1, min=-1)
        wavs = (wavs.cpu().numpy() * MAX_WAV_VALUE).astype("int16")
        wavs = [wav for wav in wavs]

        for i in range(len(mels)):
            if lengths is not None:
                wavs[i] = wavs[i][: lengths[i]]
        return wavs


class HifiGAN(BaseVocoder):
    def __init__(self, dirpath: str=None):
        super().__init__()
        if dirpath is None:
            _current_dir = os.path.dirname(__file__)
            # with open(f"{_current_dir}/hifigan/config.json", "r") as f:
            #     config = json.load(f)
            with open(f"{_current_dir}/hifigan/Universal/config.json", "r") as f:
                config = json.load(f)
            config = hifigan.AttrDict(config)
            vocoder = hifigan.Generator(config)
            # ckpt = torch.load(f"{_current_dir}/hifigan/generator_universal.pth.tar")
            ckpt = torch.load(f"{_current_dir}/hifigan/Universal/generator.ckpt")
        else:
            with open(f"{dirpath}/config.json", "r") as f:
                config = json.load(f)
            config = hifigan.AttrDict(config)
            vocoder = hifigan.Generator(config)
            ckpt = torch.load(f"{dirpath}/generator.ckpt")
        
        vocoder.load_state_dict(ckpt["generator"])
        vocoder.remove_weight_norm()

        self.vocoder = vocoder
        self.vocoder.eval()

    def inverse(self, mels):
        with torch.no_grad():
            return self.vocoder(mels).squeeze(1)

    def infer(self, mels, lengths=None, *args, **kwargs):
        # wavs = self.inverse(mels) compatible with stft framework only, deprecated
        wavs = self.inverse(mels)
        wavs = torch.clip(wavs, max=1, min=-1)
        wavs = (wavs.cpu().numpy() * MAX_WAV_VALUE).astype("int16")
        wavs = [wav for wav in wavs]

        for i in range(len(mels)):
            if lengths is not None:
                wavs[i] = wavs[i][: lengths[i]]
        return wavs
    