import os
from tqdm import tqdm
import json
from pathlib import Path
from multiprocessing import Pool

from dlhlp_lib.tts_preprocess.utils import ImapWrapper
from dlhlp_lib.tts_preprocess.basic import *
from dlhlp_lib.text.utils import lowercase

import Define
from Parsers.interface import BaseRawParser, BasePreprocessor
from .parser import DataParser
from . import template


class ALFFARawParser(BaseRawParser):
    def __init__(self, root: Path, preprocessed_root: Path, lang: str):
        super().__init__(root)
        self.lang = lang
        self.data_parser = DataParser(str(preprocessed_root))

    def prepare_initial_features(self, query, data):
        template.prepare_initial_features(self.data_parser, query, data)

    def parse_data_info_sw(self):
        res = {"data": [], "data_info": [], "all_speakers": []}
        # train
        path = self.root / "data_broadcastnews_sw/data/train"
        with open(path / "text", 'r', encoding='utf-8') as f:
            for line in f:
                if line == "\n":
                    continue
                try:
                    basename, text = line.strip().split('\t')
                except:
                    continue

                speaker = basename[:15]
                wav_path = path / "wav" / speaker / f"{basename}.wav"
                if speaker not in res["all_speakers"]:
                    res["all_speakers"].append(speaker)

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
        # test
        utt2spk = {}
        path = self.root / "data_broadcastnews_sw/data/test"
        with open(path / "utt2spk", 'r', encoding='utf-8') as f:
            for line in f:
                if line == "\n":
                    continue
                k, v = line.strip().split()
                utt2spk[k] = v
        with open(path / "text", 'r', encoding='utf-8') as f:
            for line in f:
                if line == "\n":
                    continue
                try:
                    x = line.strip().split()
                    basename = x[0]
                    text = ' '.join(x[1:])
                except:
                    continue
                speaker = utt2spk[basename]
                wav_path = path / "wav5" / speaker / f"{basename}.wav"
                if speaker not in res["all_speakers"]:
                    res["all_speakers"].append(speaker)

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

        return res

    def parse_data_info_am(self):
        res = {"data": [], "data_info": [], "all_speakers": []}
        for path in [self.root / "data_readspeech_am/data/train", self.root / "data_readspeech_am/data/test"]:
            utt2spk = {}
            with open(path / "utt2spk", 'r', encoding='utf-8') as f:
                for line in f:
                    if line == "\n":
                        continue
                    k, v = line.strip().split()
                    utt2spk[k] = v
            with open(path / "text", 'r', encoding='utf-8') as f:
                for line in f:
                    if line == "\n":
                        continue
                    try:
                        x = line.strip().split()
                        basename = x[0]
                        text = ' '.join(x[1:])
                    except:
                        continue
                    speaker = utt2spk[basename]
                    wav_path = path / "wav" / f"{basename}.wav"
                    if speaker not in res["all_speakers"]:
                        res["all_speakers"].append(speaker)

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

        return res

    def parse_data_info_wo(self):
        res = {"data": [], "data_info": [], "all_speakers": []}
        for path in [self.root / "data_readspeech_wo/data/train", self.root / "data_readspeech_wo/data/test", self.root / "data_readspeech_wo/data/dev"]:
            with open(path / "text", 'r', encoding='utf-8') as f:
                for line in f:
                    if line == "\n":
                        continue
                    try:
                        x = line.strip().split()
                        basename = x[0]
                        text = ' '.join(x[1:])
                    except:
                        continue
                    speaker = basename[4:6]
                    wav_path = path / speaker / f"{basename}.wav"
                    if speaker not in res["all_speakers"]:
                        res["all_speakers"].append(speaker)

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

        return res

    def parse(self, n_workers=4):
        if self.lang == "sw":
            res = self.parse_data_info_sw()
        elif self.lang == "am":
            res = self.parse_data_info_am()
        elif self.lang == "wo":
            res = self.parse_data_info_wo()
        else:
            raise NotImplementedError

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


class ALFFAPreprocessor(BasePreprocessor):
    def __init__(self, preprocessed_root: Path, lang: str):
        super().__init__(preprocessed_root)
        self.lang = lang
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


class ALFFASWRawParser(ALFFARawParser):
    def __init__(self, root: Path, preprocessed_root: Path):
        super().__init__(root, preprocessed_root, lang="sw")


class ALFFAAMRawParser(ALFFARawParser):
    def __init__(self, root: Path, preprocessed_root: Path):
        super().__init__(root, preprocessed_root, lang="am")


class ALFFAWORawParser(ALFFARawParser):
    def __init__(self, root: Path, preprocessed_root: Path):
        super().__init__(root, preprocessed_root, lang="wo")


class ALFFASWPreprocessor(ALFFAPreprocessor):
    def __init__(self, preprocessed_root: Path):
        super().__init__(preprocessed_root, lang="sw")


class ALFFAAMPreprocessor(ALFFAPreprocessor):
    def __init__(self, preprocessed_root: Path):
        super().__init__(preprocessed_root, lang="am")


class ALFFAWOPreprocessor(ALFFAPreprocessor):
    def __init__(self, preprocessed_root: Path):
        super().__init__(preprocessed_root, lang="wo")
