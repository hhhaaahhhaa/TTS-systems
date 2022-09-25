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
        pitch = self.data_parser.mfa_duration_avg_pitch.read_from_query(query)
        energy = self.data_parser.mfa_duration_avg_energy.read_from_query(query)
        duration = self.data_parser.mfa_duration.read_from_query(query)
        phonemes = self.data_parser.phoneme.read_from_query(query)
        raw_text = self.data_parser.text.read_from_query(query)
        mel = np.transpose(mel[:, :sum(duration)])
        phonemes = f"{{{phonemes}}}"

        _, _, global_pitch_mu, global_pitch_std, _, _, global_energy_mu, global_energy_std = Define.ALLSTATS["global"]
        pitch = (pitch - global_pitch_mu) / global_pitch_std  # normalize
        energy = (energy - global_energy_mu) / global_energy_std  # normalize
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
            "speaker": speaker_id,
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
