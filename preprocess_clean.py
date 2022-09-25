import torch
from tqdm import tqdm

from dlhlp_lib.utils import numpy_exist_nan

from Parsers.parser import DataParser


def check_duration(queries, data_parser: DataParser):
    res = []
    for query in tqdm(queries):
        try:
            wav = data_parser.wav_trim_22050.read_from_query(query)
            assert len(wav) <= 22050 * 15
            res.append(query)
        except:
            print("Audio too long, skipped:")
            print(query)
    return res


def check_existence_and_nan(queries, data_parser: DataParser):
    res = []
    for query in tqdm(queries):
        try:
            assert not numpy_exist_nan(data_parser.mfa_duration.read_from_query(query))
            assert not numpy_exist_nan(data_parser.mel.read_from_query(query))
            assert not numpy_exist_nan(data_parser.interpolate_pitch.read_from_query(query))
            assert not numpy_exist_nan(data_parser.energy.read_from_query(query))
            res.append(query)
        except:
            print("NaN in feature or feature does not exist, skipped:")
            print(query)
    return res


def preprocess_ljspeech(data_parser: DataParser, output_dir: str):
    print("Clean LJSpeech...")
    queries = data_parser.get_all_queries()
    queries = check_existence_and_nan(queries, data_parser)
    queries = check_duration(queries, data_parser)

    print("Split LJSpeech...")
    split = {
        "train": slice(0, -1500),
        "val": slice(-1500, -500),
        "test": slice(-500, None)
    }
    # train/val/test split
    phoneme_feat = data_parser.phoneme
    text_feat = data_parser.text
    for k, s in split.items():
        with open(f"{output_dir}/{k}.txt", 'w', encoding='utf-8') as f:
            for q in tqdm(queries[s], desc=k):
                phoneme = phoneme_feat.read_from_query(q)
                text = text_feat.read_from_query(q)
                line = f"{q['basename']}|{q['spk']}|{{{phoneme}}}|{text}"
                f.write(line + '\n')


def preprocess_libritts(data_parser: DataParser, output_dir: str):
    print("Clean LibriTTS...")
    queries = data_parser.get_all_queries()
    queries = check_existence_and_nan(queries, data_parser)
    queries = check_duration(queries, data_parser)

    print("Split LibriTTS...")
    split = {
        "train-clean-100": [],
        "dev-clean": [],
        "test-clean": []
    }
    # train/val/test split
    phoneme_feat = data_parser.phoneme
    text_feat = data_parser.text
    for q in tqdm(queries[s], desc=k):
        phoneme = phoneme_feat.read_from_query(q)
        text = text_feat.read_from_query(q)
        line = f"{q['basename']}|{q['spk']}|{{{phoneme}}}|{text}"
        split[q["dset"]].append(line)
    
    for k, s in split.items():
        with open(f"{output_dir}/{k}.txt", 'w', encoding='utf-8') as f:
            for line in s:
                f.write(line + '\n')


if __name__ == "__main__":
    parser = DataParser("./preprocessed_data/LJSpeech-1.1")
    preprocess_ljspeech(parser, "data_config/LJSpeech-1.1")

    # parser = DataParser("./preprocessed_data/LibriTTS")
    # preprocess_ljspeech(parser, "data_config/LibriTTS")

    # parser = DataParser("./preprocessed_data/AISHELL-3")
    # preprocess_ljspeech(parser, "data_config/AISHELL-3")
    