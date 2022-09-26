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

        self.basename, self.speaker = self.process_meta(filename)
        with open(self.data_parser.speakers_path, 'r', encoding='utf-8') as f:
            self.speakers = json.load(f)
            self.speaker_map = {spk: i for i, spk in enumerate(self.speakers)}

    def __len__(self):
        return len(self.basename)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        query = {
            "spk": speaker,
            "basename": basename,
        }

        mel = self.data_parser.mel.read_from_query(query)
        mel = np.transpose(mel)
        raw_text = self.data_parser.text.read_from_query(query)
        text = np.array(text_to_sequence(raw_text, self.cleaners, self.lang_id))
        
        assert not numpy_exist_nan(mel)

        sample = {
            "id": basename,
            "speaker": speaker_id,
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
