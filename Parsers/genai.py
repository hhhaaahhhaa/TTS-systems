import os
from tqdm import tqdm
import json
from pathlib import Path
from multiprocessing import Pool
import librosa
import yaml

from dlhlp_lib.audio.tools import wav_normalization
from dlhlp_lib.tts_preprocess.utils import ImapWrapper
from dlhlp_lib.tts_preprocess.basic import *

import Define
from Parsers.interface import BaseRawParser, BasePreprocessor
from .parser import DataParser
from . import template


UNIT_NAME = "50Hz"  # Hack


class GenAIRawParser(BaseRawParser):
    def __init__(self, root: Path, preprocessed_root: Path):
        super().__init__(root)
        self.data_parser = DataParser(str(preprocessed_root))
        self.unit_name = UNIT_NAME

    def prepare_initial_features(self, query, data):
        wav_22050, _ = librosa.load(data["wav_path"], sr=22050)
        wav_22050 = wav_normalization(wav_22050)
        self.data_parser.wav_22050.save(wav_22050, query)

        # Hardcode
        self.data_parser.units[self.unit_name].phoneme.save(data["text"], query)
        n = len(data["text"].strip().split(" "))
        segment = [[i * 0.02, (i+1) * 0.02] for i in range(n)]
        self.data_parser.units[self.unit_name].segment.save(segment, query)

    def parse(self, n_workers=4):
        self.data_parser.create_unit_feature(self.unit_name)
        res = {"data": [], "data_info": [], "all_speakers": ["GenAI"]}

        # Hardcode
        wav_dir = f"{self.root}/wavs"
        unit_dir = f"{self.root}/units/{self.unit_name}"

        speaker = "GenAI"
        for filename in os.listdir(wav_dir):
            wav_path = f"{wav_dir}/{filename}"
            basename = filename[:-4]
            with open(f"{unit_dir}/{basename}.txt", encoding="utf-8") as f:
                text = f.read().strip()
            data = {
                "wav_path": wav_path,
                "text": text,
            }
            data_info = {
                "spk": speaker,
                "basename": basename,
            }
            res["data"].append(data)
            res["data_info"].append(data_info)
        
        with open(self.data_parser.metadata_path, "w", encoding="utf-8") as f:
            json.dump(res["data_info"], f, indent=4)
        with open(self.data_parser.speakers_path, "w", encoding="utf-8") as f:
            json.dump(res["all_speakers"], f, indent=4)

        n = len(res["data_info"])
        tasks = list(zip(res["data_info"], res["data"], [False] * n))
        
        with Pool(processes=n_workers) as pool:
            for res in tqdm(pool.imap(ImapWrapper(self.prepare_initial_features), tasks, chunksize=64), total=n):
                pass
        self.data_parser.text.build_cache()


class GenAIPreprocessor(BasePreprocessor):
    def __init__(self, preprocessed_root: Path):
        super().__init__(preprocessed_root)
        self.data_parser = DataParser(str(preprocessed_root))
        self.unit_name = UNIT_NAME

    def prepare_mfa(self, mfa_data_dir: Path):
        pass
    
    def mfa(self, mfa_data_dir: Path):
        pass
    
    def denoise(self):
        pass

    def preprocess(self):
        queries = self.data_parser.get_all_queries()
        if Define.DEBUG:
            queries = queries[:128]

        ignore_errors = True
        if Define.DEBUG:
            ignore_errors = False
        wav_to_mel_energy_pitch_mp(
            self.data_parser, queries,
            wav_featname="wav_22050",
            mel_featname="mel",
            energy_featname="energy",
            pitch_featname="pitch",
            interp_pitch_featname="interpolate_pitch",
            ignore_errors=ignore_errors,
            n_workers=1
        )
        segment2duration_mp(
            self.data_parser, queries, inv_frame_period=template.INV_FRAME_PERIOD,
            segment_featname=f"units/{self.unit_name}/segment",
            duration_featname=f"units/{self.unit_name}/duration",
            refresh=True,
            ignore_errors=ignore_errors
        )
        duration_avg_pitch_and_energy_mp(
            self.data_parser, queries,
            duration_featname=f"units/{self.unit_name}/duration",
            pitch_featname="interpolate_pitch",
            energy_featname="energy",
            refresh=True,
            ignore_errors=ignore_errors
        )
        stats = get_stats(
            self.data_parser,
            pitch_featname="interpolate_pitch",
            energy_featname="energy",
            refresh=True
        )
        with open(self.data_parser.stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f)
        
        # Generate cache
        self.data_parser.units[self.unit_name].phoneme.build_cache()
        self.data_parser.units[self.unit_name].segment.build_cache()
        self.data_parser.units[self.unit_name].duration.build_cache()
        self.data_parser.units[self.unit_name].duration_avg_energy.build_cache()
        self.data_parser.units[self.unit_name].duration_avg_pitch.build_cache()

    def split_dataset(self, cleaned_data_info_path: str):
        output_dir = os.path.dirname(cleaned_data_info_path)
        with open(cleaned_data_info_path, 'r', encoding='utf-8') as f:
            queries = json.load(f)

        val_size = 1000
        train_set = queries[:-val_size]
        val_set = queries[-val_size:]
        test_set = val_set
        with open(f"{output_dir}/train.json", 'w', encoding='utf-8') as f:
            json.dump(train_set, f, indent=4)
        with open(f"{output_dir}/val.json", 'w', encoding='utf-8') as f:
            json.dump(val_set, f, indent=4)
        with open(f"{output_dir}/test.json", 'w', encoding='utf-8') as f:
            json.dump(test_set, f, indent=4)

        # Generate config.yaml
        with open("data_config/GenAI/config.yaml", 'w') as yamlfile:
            config = {
                "name": "GenAI",
                "data_dir": self.data_parser.root,
                "subsets": {
                    "train": "train.json",
                    "val": "test.json",
                    "test": "test.json"
                }
            }
            yaml.dump(config, yamlfile)
