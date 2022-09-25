import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
from tqdm import tqdm

from dlhlp_lib.parsers.preprocess import *
from dlhlp_lib.audio import AUDIO_CONFIG
from Parsers.parser import DataParser


INV_FRAME_PERIOD = AUDIO_CONFIG["audio"]["sampling_rate"] / AUDIO_CONFIG["stft"]["hop_length"]


def textgrid2segment_and_phoneme_wrapped(data_parser: DataParser, queries):
    print("textgrid2segment_and_phoneme:")
    for query in tqdm(queries):
        textgrid2segment_and_phoneme(data_parser, query, "textgrid", "mfa_segment", "phoneme")


def trim_wav_by_segment_wrapped(data_parser: DataParser, queries, sr: int):
    print("trim_wav_by_segment:")
    for query in tqdm(queries):
        trim_wav_by_segment(data_parser, query, sr, f"wav_{sr}", "mfa_segment", f"wav_trim_{sr}")


def wav_to_mel_energy_pitch_wrapped(data_parser: DataParser, queries):
    print("wav_to_mel_energy_pitch:")
    for query in tqdm(queries):
        wav_to_mel_energy_pitch(data_parser, query, "wav_trim_22050", "mel", "energy", "pitch", "interpolate_pitch")


def segment2duration_wrapped(data_parser: DataParser, queries, inv_frame_period: float, segment_featname: str):
    print("segment2duration:")
    duration_featname = segment_featname.replace("segment", "duration")
    for query in tqdm(queries):
        segment2duration(data_parser, query, inv_frame_period, segment_featname, duration_featname)


def duration_avg_pitch_and_energy_wrapped(data_parser: DataParser, queries, duration_featname: str):
    print("duration_avg_pitch_and_energy:")
    for query in tqdm(queries):
        duration_avg_pitch_and_energy(data_parser, query, duration_featname, "interpolate_pitch", "energy")


def preprocess(root):
    print(f"Preprocess data from {root}...")

    data_parser = DataParser(root)
    queries = data_parser.get_all_queries()
    
    textgrid2segment_and_phoneme_wrapped(data_parser, queries)
    trim_wav_by_segment_wrapped(data_parser, queries, sr=22050)
    wav_to_mel_energy_pitch_wrapped(data_parser, queries)
    segment2duration_wrapped(data_parser, queries, INV_FRAME_PERIOD, "mfa_segment")
    duration_avg_pitch_and_energy_wrapped(data_parser, queries, "mfa_duration")
    
    get_stats(data_parser, "interpolate_pitch", "energy", "stats_path", refresh=True)


if __name__ == "__main__":
    preprocess("./preprocessed_data/LJSpeech-1.1")
    # preprocess("./preprocessed_data/LibriTTS")
    # preprocess("./preprocessed_data/AISHELL-3")
