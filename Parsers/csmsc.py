import os
from tqdm import tqdm
import json
from pathlib import Path
from multiprocessing import Pool

from dlhlp_lib.tts_preprocess.utils import ImapWrapper
from dlhlp_lib.tts_preprocess.basic import *

import Define
from Parsers.interface import BaseRawParser, BasePreprocessor
from .parser import DataParser
from . import template


class CSMSCRawParser(BaseRawParser):
    def __init__(self, root: Path, preprocessed_root: Path):
        super().__init__(root)
        self.data_parser = DataParser(str(preprocessed_root))

    def prepare_initial_features(self, query, data):
        template.prepare_initial_features(self.data_parser, query, data)

    def parse(self, n_workers=4):
        res = {"data": [], "data_info": [], "all_speakers": ["csmsc"]}
        path = f"{self.root}/ProsodyLabeling/000001-010000.txt"
        speaker = "csmsc"
        with open(path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                if line == "\n" or line[0] == "\t":
                    continue
                wav_name, text = line.strip().split("\t")
                parsed_text = ""
                st = 0
                while st < len(text):
                    if text[st] == "#":
                        st += 2
                    else:
                        parsed_text += text[st]
                        st += 1
                
                wav_path = f"{self.root}/Wave/{wav_name}.wav"
                if os.path.isfile(wav_path):
                    basename = f"{speaker}-{wav_name}"
                    data = {
                        "wav_path": wav_path,
                        "text": parsed_text,
                    }
                    data_info = {
                        "spk": speaker,
                        "basename": basename,
                    }
                    res["data"].append(data)
                    res["data_info"].append(data_info)
                else:
                    print("metadata.csv should not contain non-exist wav files, data might be corrupted.")
                    print(f"Can not find {wav_path}.")
        
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


class CSMSCPreprocessor(BasePreprocessor):
    def __init__(self, preprocessed_root: Path):
        super().__init__(preprocessed_root)
        self.data_parser = DataParser(str(preprocessed_root))

    # Use prepared textgrids from ming024's repo
    def prepare_mfa(self, mfa_data_dir: Path):
        pass
    
    # Use prepared textgrids from ming024's repo
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
