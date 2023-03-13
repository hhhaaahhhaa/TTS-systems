import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import librosa
import json
import random

from dlhlp_lib.audio.tools import wav_normalization
from dlhlp_lib.audio import AUDIO_CONFIG
from dlhlp_lib.tts_preprocess.basic import *

import Define
from .parser import DataParser
from .utils import write_queries_to_txt


INV_FRAME_PERIOD = AUDIO_CONFIG["audio"]["sampling_rate"] / AUDIO_CONFIG["stft"]["hop_length"]
random.seed(0)


def prepare_initial_features(data_parser: DataParser, query, data):
    wav_22050, _ = librosa.load(data["wav_path"], sr=22050)
    wav_22050 = wav_normalization(wav_22050)
    data_parser.wav_22050.save(wav_22050, query)
    data_parser.text.save(data["text"], query)


def preprocess(data_parser: DataParser, queries):
    data_parser.create_unit_feature("mfa")
    ignore_errors = True
    if Define.DEBUG:
        ignore_errors = False
    textgrid2segment_and_phoneme_mp(
        data_parser, queries, 
        textgrid_featname="textgrid",
        segment_featname="units/mfa/segment",
        phoneme_featname="units/mfa/phoneme",
        ignore_errors=ignore_errors
    )
    trim_wav_by_segment_mp(
        data_parser, queries, sr=22050,
        wav_featname="wav_22050",
        segment_featname="units/mfa/segment",
        wav_trim_featname="wav_trim_22050",
        refresh=True,
        ignore_errors=ignore_errors
    )
    wav_to_mel_energy_pitch_mp(
        data_parser, queries,
        wav_featname="wav_trim_22050",
        mel_featname="mel",
        energy_featname="energy",
        pitch_featname="pitch",
        interp_pitch_featname="interpolate_pitch",
        ignore_errors=ignore_errors,
        n_workers=1
    )
    segment2duration_mp(
        data_parser, queries, inv_frame_period=INV_FRAME_PERIOD,
        segment_featname="units/mfa/segment",
        duration_featname="units/mfa/duration",
        refresh=True,
        ignore_errors=ignore_errors
    )
    duration_avg_pitch_and_energy_mp(
        data_parser, queries,
        duration_featname="units/mfa/duration",
        pitch_featname="interpolate_pitch",
        energy_featname="energy",
        refresh=True,
        ignore_errors=ignore_errors
    )
    stats = get_stats(
        data_parser,
        pitch_featname="interpolate_pitch",
        energy_featname="energy",
        refresh=True
    )
    with open(data_parser.stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f)
    
    # Generate cache
    data_parser.units["mfa"].phoneme.build_cache()
    data_parser.units["mfa"].segment.build_cache()
    data_parser.units["mfa"].duration.build_cache()
    data_parser.units["mfa"].duration_avg_energy.build_cache()
    data_parser.units["mfa"].duration_avg_pitch.build_cache()


def split_monospeaker_dataset(data_parser: DataParser, queries, output_dir, val_size=1000):
    train_set = queries[:-val_size]
    val_set = queries[-val_size:]
    test_set = random.sample(val_set, k=200)
    write_queries_to_txt(data_parser, train_set, f"{output_dir}/train.txt")
    write_queries_to_txt(data_parser, val_set, f"{output_dir}/val.txt")
    write_queries_to_txt(data_parser, test_set, f"{output_dir}/test.txt")


def split_multispeaker_dataset(data_parser: DataParser, queries, output_dir, val_spk_size=40):
    spks = data_parser.get_all_speakers()
    assert len(spks) > val_spk_size
    train_spk, val_spk = spks[:-val_spk_size], spks[-val_spk_size:]

    train_set, val_set = [], []
    for q in queries:
        if q["spk"] in train_spk:
            train_set.append(q)
        elif q["spk"] in val_spk:
            val_set.append(q)
        else:
            raise ValueError("Unknown speaker detected, some error exists when preprocessing data.")
    test_set = random.sample(val_set, k=200)
    
    write_queries_to_txt(data_parser, train_set, f"{output_dir}/train.txt")
    write_queries_to_txt(data_parser, val_set, f"{output_dir}/val.txt")
    write_queries_to_txt(data_parser, test_set, f"{output_dir}/test.txt")
