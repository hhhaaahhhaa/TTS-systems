import numpy as np
from torch.utils.data import Dataset
import json

from dlhlp_lib.utils import numpy_exist_nan
from text import text_to_sequence
from Parsers.parser import DataParser


class Tacotron2Dataset(Dataset):
    """
    Monolingual, paired dataset for Tacotron2.
    """
    def __init__(self, filename, data_parser: DataParser, config):
        self.data_parser = data_parser

        self.name = config["name"]
        self.lang_id = config["lang_id"]
        self.cleaners = config["text_cleaners"]

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

        mel = self.data_parser.mel.read_from_query(query)
        raw_text = self.data_parser.text.read_from_query(query)
        text = np.array(text_to_sequence(raw_text, self.cleaners, self.lang_id))

        # In case you want to use phoneme to train tacotron
        # phonemes = self.unit_parser.phoneme.read_from_query(query)
        # phonemes = f"{{{phonemes}}}"
        # text = np.array(text_to_sequence(phonemes, self.cleaners, self.lang_id))
        
        assert not numpy_exist_nan(mel)

        sample = {
            "id": basename,
            "speaker": speaker,
            "text": text,
            "raw_text": raw_text,
            "mel": mel,
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
