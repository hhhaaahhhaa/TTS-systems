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

import xmltodict


"""
Living Audio Dataset (https://github.com/Idlak/Living-Audio-Dataset)

en/ga/nl/ru
"""


class LADRawParser(BaseRawParser):
    def __init__(self, root: Path, preprocessed_root: Path):
        super().__init__(root)
        self.lang = str(self.root).split('/')[-1]
        if self.lang == "en":
            self.identifier = "rp"
        elif self.lang == "ga":
            self.identifier = "ie"
        elif self.lang == "nl":
            self.identifier = "nl"
        elif self.lang == "ru":
            self.identifier = "ru"
        self.data_parser = DataParser(str(preprocessed_root))

    def prepare_initial_features(self, query, data):
        template.prepare_initial_features(self.data_parser, query, data)

    def parse(self, n_workers=4):
        res = {"data": [], "data_info": [], "all_speakers": []}
        wav_dir = self.root / "48000_orig"
        spk = os.listdir(wav_dir)[0].split("_")[0]
        res["all_speakers"].append(spk)

        # Open the file and read the contents
        with open(self.root/ self.identifier / spk / "text.xml", 'r', encoding='utf-8') as file:
            my_xml = file.read()
        my_dict = xmltodict.parse(my_xml)
        
        for sample in tqdm(my_dict["recording_script"]["fileid"]):
            basename = sample["@id"]
            text = sample["#text"]
            if self.lang == "en":
                basename = spk + "_" + basename
            wav_path = wav_dir / f"{basename}.wav"
            if os.path.isfile(wav_path):
                data = {
                    "wav_path": wav_path,
                    "text": text,
                }
                data_info = {
                    "spk": spk,
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
        self.data_parser.text.read_all(refresh=True)


class LADPreprocessor(BasePreprocessor):
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
