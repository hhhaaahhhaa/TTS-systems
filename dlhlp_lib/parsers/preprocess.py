from typing import List
import os
import numpy as np
import pyworld as pw
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
import json

from dlhlp_lib import audio
from dlhlp_lib.parsers.Interfaces import BaseDataParser 


SILENCE = ["sil", "sp", "spn"]


def textgrid2segment_and_phoneme(
    dataset: BaseDataParser, query,
    textgrid_featname: str,
    segment_featname: str,
    phoneme_featname: str
):
    try:
        textgrid_feat = dataset.get_feature(textgrid_featname)

        segment_feat = dataset.get_feature(segment_featname)
        phoneme_feat = dataset.get_feature(phoneme_featname)

        tier = textgrid_feat.read_from_query(query).get_tier_by_name("phones")
            
        phones = []
        durations = []
        end_idx = 0
        for t in tier._objects:
            s, e, p = t.start_time, t.end_time, t.text

            # Trim leading silences
            if phones == []:
                if p in SILENCE:
                    continue
                else:
                    pass
            phones.append(p)
            durations.append((s, e))
            if p not in SILENCE:
                end_idx = len(phones)

        durations = durations[:end_idx]
        phones = phones[:end_idx]

        segment_feat.save(durations, query)
        phoneme_feat.save(" ".join(phones), query)
    except:
        print("Skipped: ", query)


def trim_wav_by_segment(
    dataset: BaseDataParser, query, sr: int,
    wav_featname: str,
    segment_featname: str,
    wav_trim_featname: str
):
    try:
        wav_feat = dataset.get_feature(wav_featname)
        segment_feat = dataset.get_feature(segment_featname)

        wav_trim_feat = dataset.get_feature(wav_trim_featname)

        wav = wav_feat.read_from_query(query)
        segment = segment_feat.read_from_query(query)

        wav_trim_feat.save(wav[int(sr * segment[0][0]) : int(sr * segment[-1][1])], query)
    except:
        print("Skipped: ", query)


def wav_to_mel_energy_pitch(
    dataset: BaseDataParser, query,
    wav_featname: str,
    mel_featname: str,
    energy_featname: str,
    pitch_featname: str,
    interp_pitch_featname: str,
):
    try:
        wav_feat = dataset.get_feature(wav_featname)

        mel_feat = dataset.get_feature(mel_featname)
        energy_feat = dataset.get_feature(energy_featname)
        pitch_feat = dataset.get_feature(pitch_featname)
        interp_pitch_feat = dataset.get_feature(interp_pitch_featname)

        sr = audio.AUDIO_CONFIG["audio"]["sampling_rate"]
        hop_length = audio.AUDIO_CONFIG["stft"]["hop_length"]
        assert sr == 22050

        wav = wav_feat.read_from_query(query)

        # Compute fundamental frequency
        pitch, t = pw.dio(
            wav.astype(np.float64),
            22050,
            frame_period=hop_length / sr * 1000,
        )
        pitch = pw.stonemask(wav.astype(np.float64), pitch, t, sr)

        nonzero_ids = np.where(pitch != 0)[0]
        interp_fn = interp1d(
            nonzero_ids,
            pitch[nonzero_ids],
            fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
            bounds_error=False,
        )
        interp_pitch = interp_fn(np.arange(0, len(pitch)))

        if np.sum(pitch != 0) <= 1:
            raise ValueError("Zero pitch detected")

        # Compute mel-scale spectrogram and energy
        mel_spectrogram, energy = audio.get_mel_from_wav(wav, audio.STFT)

        mel_feat.save(mel_spectrogram, query)
        energy_feat.save(energy, query)
        pitch_feat.save(pitch, query)
        interp_pitch_feat.save(interp_pitch, query)
    except:
        print("Skipped: ", query)


def segment2duration(
    dataset: BaseDataParser, query, inv_frame_period: float,
    segment_featname: str, 
    duration_featname: str
):
    try:
        segment_feat = dataset.get_feature(segment_featname)

        duration_feat = dataset.get_feature(duration_featname)

        segment = segment_feat.read_from_query(query)
        durations = []
        for (s, e) in segment:
            durations.append(int(np.round(e * inv_frame_period) - np.round(s * inv_frame_period)))
        
        duration_feat.save(durations, query)
    except:
        print("Skipped: ", query)


def duration_avg_pitch_and_energy(
    dataset: BaseDataParser, query,
    duration_featname: str,
    pitch_featname: str,
    energy_featname: str,
):
    try:
        duration_feat = dataset.get_feature(duration_featname)
        pitch_feat = dataset.get_feature(pitch_featname)
        energy_feat = dataset.get_feature(energy_featname)

        avg_pitch_feat = dataset.get_feature(f"{duration_featname}_avg_pitch")
        avg_energy_feat = dataset.get_feature(f"{duration_featname}_avg_energy")

        durations = duration_feat.read_from_query(query)
        pitch = pitch_feat.read_from_query(query)
        energy = energy_feat.read_from_query(query)

        avg_pitch, avg_energy = pitch[:], energy[:]
        avg_pitch = representation_average(avg_pitch, durations)
        avg_energy = representation_average(avg_energy, durations)

        avg_pitch_feat.save(avg_pitch, query)
        avg_energy_feat.save(avg_energy, query)
    except:
        print("Skipped: ", query)


def get_stats(
    dataset: BaseDataParser,
    pitch_featname: str, 
    energy_featname: str,
    stat_featname: str,
    refresh=False
):
    interp_pitch_feat = dataset.get_feature(pitch_featname)
    energy_feat = dataset.get_feature(energy_featname)

    stats_path = dataset.get_feature(stat_featname)

    interp_pitch_feat.read_all(refresh=refresh)
    energy_feat.read_all(refresh=refresh)
    all_pitches, all_energies = [], []
    for k, v in interp_pitch_feat._data.items():
        for x in v:
            all_pitches.append(x)
    for k, v in energy_feat._data.items():
        for x in v:
            all_energies.append(x)

    pitch_min, pitch_max = min(all_pitches), max(all_pitches)
    energy_min, energy_max = min(all_energies), max(all_energies)

    pitch_scaler = StandardScaler()
    energy_scaler = StandardScaler()
    pitch_scaler.partial_fit(remove_outlier(all_pitches).reshape((-1, 1)))
    energy_scaler.partial_fit(remove_outlier(all_energies).reshape((-1, 1)))
    pitch_mean = pitch_scaler.mean_[0]
    pitch_std = pitch_scaler.scale_[0]
    energy_mean = energy_scaler.mean_[0]
    energy_std = energy_scaler.scale_[0]

    stats = {
        "pitch": [
            float(pitch_min),
            float(pitch_max),
            float(pitch_mean),
            float(pitch_std),
        ],
        "energy": [
            float(energy_min),
            float(energy_max),
            float(energy_mean),
            float(energy_std),
        ],
    }

    with open(stats_path, 'w', encoding="utf-8") as f:
        json.dump(stats, f)


"""
Helper functions.
"""
def remove_outlier(values: List):
    values = np.array(values)
    p25 = np.percentile(values, 25)
    p75 = np.percentile(values, 75)
    lower = p25 - 1.5 * (p75 - p25)
    upper = p75 + 1.5 * (p75 - p25)
    normal_indices = np.logical_and(values > lower, values < upper)

    return values[normal_indices]


def representation_average(representation, durations, pad=0):
    pos = 0
    for i, d in enumerate(durations):
        if d > 0:
            representation[i] = np.mean(
                representation[pos: pos + d], axis=0)
        else:
            representation[i] = pad
        pos += d
    return representation[: len(durations)]
