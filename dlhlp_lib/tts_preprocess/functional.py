"""
Functional version for tts_preprocess
"""
import os
import numpy as np
from scipy.interpolate import interp1d
import resemblyzer
import tgt
from typing import Tuple, List

from dlhlp_lib.audio import AUDIO_CONFIG
from dlhlp_lib.audio.features import Energy, LogMelSpectrogram, get_feature_from_wav, get_f0_from_wav
from dlhlp_lib.tts_preprocess.utils import *


ENERGY_NN = Energy(
    n_fft=AUDIO_CONFIG["stft"]["filter_length"],
    win_length=AUDIO_CONFIG["stft"]["win_length"],
    hop_length=AUDIO_CONFIG["stft"]["hop_length"]
)
MEL_NN = LogMelSpectrogram(
    sample_rate=AUDIO_CONFIG["audio"]["sampling_rate"],
    n_fft=AUDIO_CONFIG["stft"]["filter_length"],
    win_length=AUDIO_CONFIG["stft"]["win_length"],
    hop_length=AUDIO_CONFIG["stft"]["hop_length"],
    n_mels=AUDIO_CONFIG["mel"]["n_mel_channels"],
    pad=(AUDIO_CONFIG["stft"]["filter_length"] - AUDIO_CONFIG["stft"]["hop_length"]) // 2,
    power=1,
    norm="slaney",
    mel_scale="slaney"
)
SILENCE = ["sil", "sp", "spn"]


def textgrid2segment_and_phoneme(
    tier_obj: tgt.TextGrid,
) -> Tuple[List[str], List[Tuple[float, float]]]:
    if tier_obj.has_tier("phones"):
        tier = tier_obj.get_tier_by_name("phones")
    elif tier_obj.has_tier("phoneme"): 
        tier = tier_obj.get_tier_by_name("phoneme")
    else:
        raise ValueError("Can not find phoneme tiers...")
    phones, segments = get_alignment(tier)

    return phones, segments


def trim_wav_by_segment(
    wav: np.ndarray,
    segment: List[Tuple[float, float]],
    sr: int
) -> np.ndarray:
    return wav[int(sr * segment[0][0]) : int(sr * segment[-1][1])]


def wav_to_mel_energy_pitch(
    wav: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pitch = get_f0_from_wav(
        wav,
        sample_rate=AUDIO_CONFIG["audio"]["sampling_rate"],
        hop_length=AUDIO_CONFIG["stft"]["hop_length"]
    )  # (time, )
    mel = get_feature_from_wav(
        wav, feature_nn=MEL_NN
    )  # (n_mels, time)
    energy = get_feature_from_wav(
        wav, feature_nn=ENERGY_NN
    )  # (time, )

    # interpolate
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

    return mel, energy, pitch, interp_pitch


def segment2duration(
    segment: List[Tuple[float, float]],
    inv_frame_period: float,
) -> List[int]:
    durations = []
    for (s, e) in segment:
        durations.append(int(np.round(e * inv_frame_period) - np.round(s * inv_frame_period)))
    return durations


def extract_spk_ref_mel_slices_from_wav(
    wav: np.ndarray,
    sr: int
) -> List[np.ndarray]:
    wav = resemblyzer.preprocess_wav(wav, source_sr=sr)

    # Compute where to split the utterance into partials and pad the waveform
    # with zeros if the partial utterances cover a larger range.
    wav_slices, mel_slices = resemblyzer.VoiceEncoder.compute_partial_slices(
        len(wav), rate=1.3, min_coverage=0.75
    )
    max_wave_length = wav_slices[-1].stop
    if max_wave_length >= len(wav):
        wav = np.pad(wav, (0, max_wave_length - len(wav)), "constant")
    # Split the utterance into partials and forward them through the model
    spk_ref_mel = resemblyzer.wav_to_mel_spectrogram(wav)
    spk_ref_mel_slices = [spk_ref_mel[s] for s in mel_slices]
    
    return spk_ref_mel_slices
