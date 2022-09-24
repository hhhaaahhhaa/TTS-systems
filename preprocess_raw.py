import os
import numpy as np
from multiprocessing import Pool, set_start_method
import librosa
from tqdm import tqdm
import json

from Parsers.parser import DataParser
from Parsers.aishell3 import AISHELL3RawParser
from Parsers.ljspeech import LJSpeechRawParser
from Parsers.libritts import LibriTTSRawParser


def wav_normalization(wav: np.array) -> np.array:
    return wav / max(abs(wav))


def preprocess_func(data_parser: DataParser, data_info, data):
    wav_22050, _ = librosa.load(data["wav_path"], sr=22050)
    wav_22050 = wav_normalization(wav_22050)
    query = {
        "spk": data_info["spk"],
        "basename": data_info["basename"],
    }
    data_parser.wav_22050.save(wav_22050, query)
    data_parser.text.save(data["text"], query)


def imap_preprocess_func(task):
    preprocess_func(*task)


def preprocess_raw(parser_name, raw_root, preprocessed_root, n_workers=4):
    os.makedirs(preprocessed_root, exist_ok=True)
    print(f"Parsing raw data from {raw_root}...")
    if parser_name == "AISHELL-3":
        raw_parser = AISHELL3RawParser(raw_root)
    elif parser_name == "LibriTTS":
        raw_parser = LibriTTSRawParser(raw_root)
    elif parser_name == "LJSpeech":
        raw_parser = LJSpeechRawParser(raw_root)
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
    # 如果同學有比較好的 multiprocessing 寫法請私訊助教，助教不太會用 python 的 multiprocessing，QQ。
    if n_workers > 1:  
        tasks = list(zip([data_parser] * n, data_infos, datas))
        
        with Pool(processes=n_workers) as pool:
            for i in tqdm(pool.imap(imap_preprocess_func, tasks, chunksize=64), total=n):
                pass
    else:
        for i in range(n):
            preprocess_func(data_parser, data_infos[i], datas[i])


if __name__ == "__main__":
    # 如果同學有比較好的 multiprocessing 寫法請私訊助教，助教不太會用 python 的 multiprocessing，QQ。
    from sys import platform
    if platform == "linux" or platform == "linux2":
        set_start_method("spawn", force=True)
    
    preprocess_raw("LJSpeech", "/mnt/d/Data/LJSpeech-1.1", "./preprocessed_data/LJSpeech-1.1", n_workers=4)
    # preprocess_raw("LibriTTS", "/mnt/d/Data/AISHELL-3", "./preprocessed_data/LibriTTS", n_workers=4)
    # preprocess_raw("AISHELL-3", "/mnt/d/Data/AISHELL-3", "./preprocessed_data/AISHELL-3", n_workers=4)
    