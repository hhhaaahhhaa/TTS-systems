import os
from tqdm import tqdm
import json
from pathlib import Path
from multiprocessing import Pool
import random

from dlhlp_lib.tts_preprocess.utils import ImapWrapper
from dlhlp_lib.tts_preprocess.basic import *

import Define
from Parsers.interface import BaseRawParser, BasePreprocessor
from .parser import DataParser
from . import template


"""
de_DE
en_UK
en_US
es_ES
fr_FR
it_IT
pl_PL
ru_RU
uk_UK
"""


class MAILABSRawParser(BaseRawParser):
    def __init__(self, root: Path, preprocessed_root: Path):
        random.seed(0)
        super().__init__(root)
        self.lang = str(self.root).split('/')[-1]
        self.data_parser = DataParser(str(preprocessed_root))

    def prepare_initial_features(self, query, data):
        template.prepare_initial_features(self.data_parser, query, data)

    def parse(self, n_workers=4):
        res = {"data": [], "data_info": [], "all_speakers": []}
        if self.lang == "fr_FR":  # french has wrong format, might be fixed in future by official
            paths = [self.root / "male", self.root / "female"]
        else:
            paths = [self.root / "by_book/male", self.root / "by_book/female"]
        for path in paths:
            if not os.path.isdir(path):
                continue
            speakers = os.listdir(path)
            res["all_speakers"].extend(speakers)
            for speaker in tqdm(speakers):
                books = os.listdir(path / speaker)
                for book in books:
                    if not os.path.isdir(path / speaker / book):
                        continue
                    dirpath = path / speaker / book
                    with open(dirpath / "metadata.csv", 'r', encoding='utf-8') as f:
                        for line in f:
                            if line == "\n":
                                continue
                            wav_name, origin_text, text = line.strip().split("|")
                            wav_path = dirpath / "wavs" / f"{wav_name}.wav"
                            if os.path.isfile(wav_path):
                                basename = wav_name
                                data = {
                                    "wav_path": wav_path,
                                    "text": text,
                                }
                                data_info = {
                                    "spk": speaker,
                                    "basename": basename,
                                    "book": book,
                                    "origin_text": origin_text,
                                }
                                res["data"].append(data)
                                res["data_info"].append(data_info)
                            else:
                                print("metadata.csv should not contain non-exist wav files, data might be corrupted.")
                                print(f"Can not find {wav_path}.")

        # Random choice up to 20000 samples
        if len(res["data_info"]) > 20000:
            res["data_info"], res["data"] = zip(*random.sample(list(zip(res["data_info"], res["data"])), 20000))
        with open(self.data_parser.metadata_path, "w", encoding="utf-8") as f:
            json.dump(res["data_info"], f, indent=4)
        with open(self.data_parser.speakers_path, "w", encoding="utf-8") as f:
            json.dump(res["all_speakers"], f, indent=4)

        n = len(res["data_info"])
        tasks = list(zip(res["data_info"], res["data"], [False] * n))
        
        with Pool(processes=n_workers) as pool:
            for res in tqdm(pool.imap(ImapWrapper(self.prepare_initial_features), tasks, chunksize=64), total=n):
                pass
        self.data_parser.text.read_all(refresh=True)


class MAILABSPreprocessor(BasePreprocessor):
    def __init__(self, preprocessed_root: Path):
        super().__init__(preprocessed_root)
        self.lang = str(self.root).split('/')[-1]
        self.data_parser = DataParser(str(preprocessed_root))

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
        template.preprocess(self.data_parser, queries)

    def split_dataset(self, cleaned_data_info_path: str):
        output_dir = os.path.dirname(cleaned_data_info_path)
        with open(cleaned_data_info_path, 'r', encoding='utf-8') as f:
            queries = json.load(f)
        template.split_monospeaker_dataset(self.data_parser, queries, output_dir, val_size=1000)
