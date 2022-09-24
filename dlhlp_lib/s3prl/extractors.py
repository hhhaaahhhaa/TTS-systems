import numpy as np
import torch
import librosa
import pytorch_lightning as pl
from typing import List, Union, Tuple

import s3prl.hub as hub

from dlhlp_lib import Constants


class S3PRLExtractor(pl.LightningModule):
    """
    Simplified wrapper class for original extractors in s3prl repo.
    For inference only, pretrained models are fixed.
    Currently supported names:
        hubert
        wav2vec2
        wavlm
        hubert_large_ll60k
        wav2vec2_large_ll60k
        wavlm_large_ll60k
    """
    def __init__(self, s3prl_name: str):
        super().__init__()
        self.name = s3prl_name
        self._model = getattr(hub, s3prl_name)()
        self.eval()

        self._fp = 20  # Currently all supported ssl models use 20ms window

    def extract_from_paths(self, wav_paths: Union[str, List[str]], norm=False) -> Tuple[torch.FloatTensor, List[int]]:
        """
        Easier user interface for using s3prl extractors.
        Input is viewed as single batch and then forwarded, so be cautious about GPU usage!
        Args:
            wav_paths: Single / list of audio paths (.wav) / numpy audio files (.npy).
            norm: Normalize representation or not.
        Return:
            All hidden states and n_frames (for client to remove padding). Hidden states shape are (B, L, n_layers, dim).
            If input is not a list, then returned result is also not a list.
        """
        is_str_input = False
        if isinstance(wav_paths, str):
            is_str_input = True
            wav_paths = [wav_paths]
        wavs = []
        n_frames = []
        for wav_path in wav_paths:
            if wav_path[-4:] == ".wav":  # Support .wav or .npy input format
                wav, sr = librosa.load(wav_path, sr=None)
                assert sr == Constants.S3PRL_SAMPLE_RATE, f"Sample rate need to be {Constants.S3PRL_SAMPLE_RATE} (get {sr})."
            elif wav_path[-4:] == ".npy":
                wav = np.load(wav_path)
            else:
                raise NotImplementedError

            wavs.append(torch.from_numpy(wav).float().to(self.device))
            n_frames.append(len(wav) // (Constants.S3PRL_SAMPLE_RATE // 1000 * self._fp))
        
        representation = self._extract(wavs, norm=norm)
        
        if is_str_input:
            return representation[0], n_frames[0]        
        return representation, n_frames

    def extract(self, wav_data: Union[List[torch.Tensor], List[np.ndarray]], norm=False) -> Tuple[torch.FloatTensor, List[int]]:
        """
        Easier user interface for using s3prl extractors.
        Input is viewed as single batch and then forwarded, so be cautious about GPU usage!
        Args:
            wavs: List of wavs represented as numpy arrays or torch tensors.
            norm: Normalize representation or not.
        Return:
            All hidden states and n_frames (for client to remove padding). Hidden states shape are (B, L, n_layers, dim).
        """
        wavs = []
        n_frames = []
        if isinstance(wav_data[0], np.ndarray):
            for wav in wav_data:
                wavs.append(torch.from_numpy(wav).float().to(self.device))
                n_frames.append(len(wav) // (Constants.S3PRL_SAMPLE_RATE // 1000 * self._fp))
        else:
            for wav in wav_data:
                wavs.append(wav.float().to(self.device))
                n_frames.append(len(wav) // (Constants.S3PRL_SAMPLE_RATE // 1000 * self._fp))
        
        representation = self._extract(wavs, norm=norm)
              
        return representation, n_frames

    def _extract(self, wavs, norm=False):
        if not self.training:
            with torch.no_grad():
                representation = self._model(wavs)
                representation = torch.stack(representation["hidden_states"], dim=2)  # bs, L, layer, dim
                if norm:
                    representation = torch.nn.functional.normalize(representation, dim=3)
        else:
            representation = self._model(wavs)
            representation = torch.stack(representation["hidden_states"], dim=2)  # bs, L, layer, dim
            if norm:
                representation = torch.nn.functional.normalize(representation, dim=3)
        return representation
