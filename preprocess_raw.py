import os
import numpy as np
from multiprocessing import Pool, set_start_method
import librosa
from tqdm import tqdm
import json

from Parsers.parser import DataParser
from Parsers.aishell3 import AISHELL3RawParser
from Parsers.css10 import CSS10RawParser
from Parsers.jsut import JSUTRawParser
from Parsers.kss import KSSRawParser
from Parsers.libritts import LibriTTSRawParser
from Parsers.globalphone import GlobalPhoneRawParser
from Parsers.TAT_TTS import TATTTSRawParser


def wav_normalization(wav: np.array) -> np.array:
    return wav / max(abs(wav))


def preprocess_func(data_parser: DataParser, data_info, data):
    wav_16000, _ = librosa.load(data["wav_path"], sr=16000)
    wav_22050, _ = librosa.load(data["wav_path"], sr=22050)
    wav_16000 = wav_normalization(wav_16000)
    wav_22050 = wav_normalization(wav_22050)
    query = {
        "spk": data_info["spk"],
        "basename": data_info["basename"],
    }
    data_parser.wav_16000.save(wav_16000, query)
    data_parser.wav_22050.save(wav_22050, query)
    data_parser.text.save(data["text"], query)


def imap_preprocess_func(task):
    preprocess_func(*task)


def preprocess_raw(parser_name, raw_root, preprocessed_root, n_workers=4):
    os.makedirs(preprocessed_root, exist_ok=True)
    print(f"Parsing raw data from {raw_root}...")
    if parser_name == "AISHELL-3":
        raw_parser = AISHELL3RawParser(raw_root)
    elif parser_name == "CSS10":
        raw_parser = CSS10RawParser(raw_root)
    elif parser_name == "JSUT":
        raw_parser = JSUTRawParser(raw_root)
    elif parser_name == "KSS":
        raw_parser = KSSRawParser(raw_root)
    elif parser_name == "LibriTTS":
        raw_parser = LibriTTSRawParser(raw_root)
    elif parser_name == "GlobalPhone":
        raw_parser = GlobalPhoneRawParser(raw_root)
    elif parser_name == "TATTTS":
        raw_paerser = TATTTSRawParser(raw_root)
    else:
        raise NotImplementedError

    raw_parser.parse()
    data_infos = raw_parser.data["data_info"]
    datas = raw_parser.data["data"]

    with open(f"{preprocessed_root}/data_info.json", "w", encoding="utf-8") as f:
        json.dump(raw_parser.data["data_info"], f, indent=4)

    with open(f"{preprocessed_root}/speakers.json", "w", encoding="utf-8") as f:
        json.dump(raw_parser.data["all_speakers"], f, indent=4)

    data_parser = DataParser(preprocessed_root)
    n = len(data_infos)
    tasks = list(zip([data_parser] * n, data_infos, datas))
    
    with Pool(processes=n_workers) as pool:
        for i in tqdm(pool.imap(imap_preprocess_func, tasks, chunksize=64), total=n):
            pass


if __name__ == "__main__":
    from sys import platform
    if platform == "linux" or platform == "linux2":
        set_start_method("spawn", force=True)
    # preprocess_raw("AISHELL-3", "/work/Data/AISHELL-3", "./preprocessed_data/AISHELL-3")
    # preprocess_raw("CSS10", "/work/Data/CSS10/german", "./preprocessed_data/CSS10/german")
    # preprocess_raw("JSUT", "/work/Data/jsut_ver1.1", "./preprocessed_data/JSUT")
    # preprocess_raw("KSS", "/work/Data/kss", "./preprocessed_data/kss")
    # preprocess_raw("LibriTTS", "/work/Data/LibriTTS", "./preprocessed_data/LibriTTS")
    # preprocess_raw("GlobalPhone", "/work/Data/GlobalPhone/French", "./preprocessed_data/GlobalPhone/french")
    preprocess_raw("TATTTS", "/mnt/d/Data/TAT-TTS", "./preprocessed_data/TATTTS")