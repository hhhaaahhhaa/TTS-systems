import numpy as np
from torch.utils.data import Dataset
import json

from dlhlp_lib.utils import numpy_exist_nan
from tts.build import build_id2symbols
import Define
from text import text_to_sequence
from Parsers.parser import DataParser


class FastSpeech2UnitDataset(Dataset):
    def __init__(self, filename, data_parser: DataParser, config):
        self.data_parser = data_parser

        self.name = config["name"]
        self.lang_id = config["lang_id"]
        self.symbol_id = config["symbol_id"]
        # self.cleaners = config["text_cleaners"]
        self.config = config

        self.unit_name = config["unit_name"]

        self.unit_parser = self.data_parser.units[self.unit_name]
        self.id2symbols = build_id2symbols([config])
        self.unit2id = {p: i for i, p in enumerate(self.id2symbols[self.symbol_id])}

        with open(filename, "r", encoding="utf-8") as f:  # Unify IO interface
            self.data_infos = json.load(f)

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        query = self.data_infos[idx]

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
        # phonemes = f"{{{phonemes}}}"
        # raw_text = self.data_parser.text.read_from_query(query)

        _, _, global_pitch_mu, global_pitch_std, _, _, global_energy_mu, global_energy_std = Define.ALLSTATS["global"]
        if self.config["pitch"]["normalization"]:
            pitch = (pitch - global_pitch_mu) / global_pitch_std
        if self.config["energy"]["normalization"]:
            energy = (energy - global_energy_mu) / global_energy_std
        # text = np.array(text_to_sequence(phonemes, self.cleaners, self.lang_id))
        text = np.array([self.unit2id[phn] for phn in phonemes.split(" ")])
        
        # Sanity check
        assert not numpy_exist_nan(mel)
        assert not numpy_exist_nan(pitch)
        assert not numpy_exist_nan(energy)
        assert not numpy_exist_nan(duration)
        try:
            assert len(text) == len(duration)
            if self.config["pitch"]["feature"] == "phoneme_level":
                assert len(duration) == len(pitch)
            else:
                assert sum(duration) == len(pitch)
            if self.config["energy"]["feature"] == "phoneme_level":
                assert len(duration) == len(energy)
            else:
                assert sum(duration) == len(pitch)
        except:
            print("Length mismatch: ", query)
            pp = self.unit_parser.phoneme.read_from_query(query)
            print(pp)
            print(len(pp.strip().split(" ")))
            print(len(text), len(phonemes), len(duration), len(pitch), len(energy))
            raise

        sample = {
            "id": query["basename"],
            "speaker": query["spk"],
            "text": text,
            "raw_text": "",
            "mel": mel,
            "pitch": pitch,
            "energy": energy,
            "duration": duration,
            "lang_id": self.lang_id,
            "symbol_id": self.symbol_id,
        }

        return sample
