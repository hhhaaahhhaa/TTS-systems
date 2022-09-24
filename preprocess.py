import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
from multiprocessing import set_start_method

from dlhlp_lib.parsers.preprocess import *
from dlhlp_lib.audio import AUDIO_CONFIG
from Parsers.parser import DataParser


INV_FRAME_PERIOD = AUDIO_CONFIG["audio"]["sampling_rate"] / AUDIO_CONFIG["stft"]["hop_length"]


def preprocess(root):
    print(f"Preprocess data from {root}...")

    data_parser = DataParser(root)
    queries = data_parser.get_all_queries()
    
    textgrid2segment_and_phoneme_mp(data_parser, queries, n_workers=os.cpu_count() // 2)
    trim_wav_by_mfa_segment_mp(data_parser, queries, sr=22050, n_workers=2, refresh=True)
    trim_wav_by_mfa_segment_mp(data_parser, queries, sr=16000, n_workers=2, refresh=False)
    wav_trim_22050_to_mel_energy_pitch_mp(data_parser, queries, n_workers=4)
    extract_spk_ref_mel_slices_from_wav_mp(data_parser, queries, sr=16000, n_workers=4)
    segment2duration_mp(data_parser, queries, "mfa_segment", "mfa_duration", INV_FRAME_PERIOD, n_workers=os.cpu_count() // 2, refresh=True)
    duration_avg_pitch_and_energy_mp(data_parser, queries, "mfa_duration", n_workers=os.cpu_count() // 2, refresh=True)
    
    get_stats(data_parser, refresh=True)


if __name__ == "__main__":
    from sys import platform
    if platform == "linux" or platform == "linux2":
        set_start_method("spawn", force=True)
    # preprocess("./preprocessed_data/AISHELL-3")
    # preprocess("./preprocessed_data/CSS10/german")
    # preprocess("./preprocessed_data/JSUT")
    preprocess("./preprocessed_data/kss")
    preprocess("./preprocessed_data/LibriTTS")
    # preprocess("./preprocessed_data/GlobalPhone/french")
