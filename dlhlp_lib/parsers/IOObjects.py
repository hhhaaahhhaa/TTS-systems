import os
import numpy as np
import pickle
import librosa
from scipy.io import wavfile
import tgt
import json

from .. import Constants
from .Interfaces import BaseIOObject


class NumpyIO(BaseIOObject):
    def __init__(self):
        self.extension = ".npy"
    
    def readfile(self, path):
        return np.load(path)

    def savefile(self, input, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            np.save(f, input)


class PickleIO(BaseIOObject):
    def __init__(self):
        self.extension = ".npy"
        # self.extension = ".pkl"
    
    def readfile(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data

    def savefile(self, input, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(input, f)


class JSONIO(BaseIOObject):
    def __init__(self, encoding="utf-8"):
        self.extension = ".json"
        self._encoding = encoding
    
    def readfile(self, path):
        with open(path, 'r', encoding=self._encoding) as f:
            data = json.load(f)
        return data

    def savefile(self, input, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding=self._encoding) as f:
            json.dump(input, f, indent=4)


class WavIO(BaseIOObject):
    def __init__(self, sr=16000):
        self._sr = sr
        self.extension = ".wav"
    
    def readfile(self, path) -> np.array:
        y, sr = librosa.load(path, sr=None)
        assert sr == self._sr, f"Wrong sample rate {sr}, expect {self._sr}."
        return y

    def savefile(self, input: np.array, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        wavfile.write(path, self._sr, (input * Constants.MAX_WAV_VALUE).astype(np.int16))


class TextGridIO(BaseIOObject):
    def __init__(self):
        self.extension = ".TextGrid"
    
    def readfile(self, path) -> tgt.TextGrid:
        return tgt.io.read_textgrid(path, include_empty_intervals=True)

    def savefile(self, input, path):
        # TODO: Not done yet
        os.makedirs(os.path.dirname(path), exist_ok=True)
        raise NotImplementedError


class TextIO(BaseIOObject):
    def __init__(self, encoding="utf-8"):
        self.extension = ".lab"
        self._encoding = encoding
    
    def readfile(self, path) -> str:
        with open(path, 'r', encoding=self._encoding) as f:
            data = f.read()
        return data

    def savefile(self, input: str, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding=self._encoding) as f:
            f.write(input)
