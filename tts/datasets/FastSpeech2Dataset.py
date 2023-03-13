import numpy as np
from torch.utils.data import Dataset
import json

from dlhlp_lib.utils import numpy_exist_nan
import Define
from text import text_to_sequence
from Parsers.parser import DataParser


class FastSpeech2Dataset(Dataset):
    """
    Monolingual, paired dataset for FastSpeech2.
    """
    def __init__(self, filename, data_parser: DataParser, config):
        self.data_parser = data_parser

        self.name = config["name"]
        self.lang_id = config["lang_id"]
        self.cleaners = config["text_cleaners"]
        self.config = config

        self.unit_parser = self.data_parser.units["mfa"]

        self.basename, self.speaker = self.process_meta(filename)

    def __len__(self):
        return len(self.basename)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        query = {
            "spk": speaker,
            "basename": basename,
        }

        duration = self.unit_parser.duration.read_from_query(query)
        mel = self.data_parser.mel.read_from_query(query)
        mel = np.transpose(mel[:, :sum(duration)])
        if self.config["pitch"]["feature"] == "phoneme_level":
            pitch = self.unit_parser.duration_avg_pitch.read_from_query(query)
        else:
            pitch = self.data_parser.interpolate_pitch.read_from_query(query)
            pitch = pitch[:sum(duration)]
        if self.config["energy"]["feature"] == "phoneme_level":
            energy = self.unit_parser.duration_avg_energy.read_from_query(query)
        else:
            energy = self.data_parser.energy.read_from_query(query)
            energy = energy[:sum(duration)]
        phonemes = self.unit_parser.phoneme.read_from_query(query)
        phonemes = f"{{{phonemes}}}"
        raw_text = self.data_parser.text.read_from_query(query)

        _, _, global_pitch_mu, global_pitch_std, _, _, global_energy_mu, global_energy_std = Define.ALLSTATS["global"]
        if self.config["pitch"]["normalization"]:
            pitch = (pitch - global_pitch_mu) / global_pitch_std
        if self.config["energy"]["normalization"]:
            energy = (energy - global_energy_mu) / global_energy_std
        text = np.array(text_to_sequence(phonemes, self.cleaners, self.lang_id))
        
        assert not numpy_exist_nan(mel)
        assert not numpy_exist_nan(pitch)
        assert not numpy_exist_nan(energy)
        assert not numpy_exist_nan(duration)
        try:
            assert len(text) == len(duration) == len(pitch) == len(energy)
        except:
            print(query)
            print(text)
            print(len(text), len(phonemes), len(duration), len(pitch), len(energy))
            raise

        sample = {
            "id": basename,
            "speaker": speaker,
            "text": text,
            "raw_text": raw_text,
            "mel": mel,
            "pitch": pitch,
            "energy": energy,
            "duration": duration,
            "lang_id": self.lang_id,
        }

        return sample

    def process_meta(self, filename):
        with open(filename, "r", encoding="utf-8") as f:
            name = []
            speaker = []
            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
            return name, speaker
