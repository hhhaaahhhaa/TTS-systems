"""
Deprecated file, will be replaced by tts_preprocess/ soon!
"""
from typing import List
import os
import numpy as np
import pyworld as pw
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
import resemblyzer
from resemblyzer import preprocess_wav, wav_to_mel_spectrogram
import json
from multiprocessing import Pool
from tqdm import tqdm
import time

# from UnsupSeg import ModelTag, load_model_from_tag

from dlhlp_lib import audio


SILENCE = ["sil", "sp", "spn"]


def textgrid2segment_and_phoneme(dataset, query):
    try:
        textgrid_feat = getattr(dataset, "textgrid")

        segment_feat = getattr(dataset, "mfa_segment")
        phoneme_feat = getattr(dataset, "phoneme")

        tier = textgrid_feat.read_from_query(query).get_tier_by_name("phones")
            
        phones = []
        durations = []
        # start_time, end_time = 0, 0
        end_idx = 0
        for t in tier._objects:
            s, e, p = t.start_time, t.end_time, t.text

            # Trim leading silences
            if phones == []:
                if p in SILENCE:
                    continue
                else:
                    pass
                    # start_time = s
            phones.append(p)
            durations.append((s, e))
            if p not in SILENCE:
                # end_time = e
                end_idx = len(phones)

        durations = durations[:end_idx]
        phones = phones[:end_idx]

        segment_feat.save(durations, query)
        phoneme_feat.save(" ".join(phones), query)
    except:
        print(query)


def imap_textgrid2segment_and_phoneme(task):
    textgrid2segment_and_phoneme(*task)


def textgrid2segment_and_phoneme_mp(dataset, queries, n_workers=os.cpu_count()-2, chunksize=64):
    print("textgrid2segment_and_phoneme_mp")
    n = len(queries)
    tasks = list(zip([dataset] * n, queries))
    
    with Pool(processes=n_workers) as pool:
        for i in tqdm(pool.imap(imap_textgrid2segment_and_phoneme, tasks, chunksize=chunksize), total=n):
            pass


def trim_wav_by_mfa_segment(dataset, query, sr):
    try:
        wav_feat = getattr(dataset, f"wav_{sr}")
        segment_feat = getattr(dataset, "mfa_segment")

        wav_trim_feat = getattr(dataset, f"wav_trim_{sr}")

        wav = wav_feat.read_from_query(query)
        segment = segment_feat.read_from_query(query)

        wav_trim_feat.save(wav[int(sr * segment[0][0]) : int(sr * segment[-1][1])], query)
    except:
        print(query)


def imap_trim_wav_by_mfa_segment(task):
    trim_wav_by_mfa_segment(*task)


def trim_wav_by_mfa_segment_mp(dataset, queries, sr, n_workers=2, chunksize=256, refresh=False):
    """
    Multiprocessing does not help too much.
    """
    print("trim_wav_by_mfa_segment_mp")
    n = len(queries)
    tasks = list(zip([dataset] * n, queries, [sr] * n))

    if n_workers == 1:
        segment_feat = getattr(dataset, "mfa_segment")
        segment_feat.read_all(refresh=refresh)
        for i in tqdm(range(n)):
            trim_wav_by_mfa_segment(dataset, queries[i], sr)
        return
    
    with Pool(processes=n_workers) as pool:
        for i in tqdm(pool.imap(imap_trim_wav_by_mfa_segment, tasks, chunksize=chunksize), total=n):
            pass


def wav_trim_22050_to_mel_energy_pitch(dataset, query):
    try:
        wav_feat = getattr(dataset, "wav_trim_22050")

        mel_feat = getattr(dataset, "mel")
        energy_feat = getattr(dataset, "energy")
        pitch_feat = getattr(dataset, "pitch")
        interp_pitch_feat = getattr(dataset, "interpolate_pitch")

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
        print(query)


def imap_wav_trim_22050_to_mel_energy_pitch(task):
    wav_trim_22050_to_mel_energy_pitch(*task)


def wav_trim_22050_to_mel_energy_pitch_mp(dataset, queries, n_workers=4, chunksize=32):
    print("wav_trim_22050_to_mel_energy_pitch_mp")
    n = len(queries)
    tasks = list(zip([dataset] * n, queries))

    if n_workers == 1:
        for i in tqdm(range(n)):
            wav_trim_22050_to_mel_energy_pitch(dataset, queries[i])
        return
    
    with Pool(processes=n_workers) as pool:
        for i in tqdm(pool.imap(imap_wav_trim_22050_to_mel_energy_pitch, tasks, chunksize=chunksize), total=n):
            pass


def extract_spk_ref_mel_slices_from_wav(dataset, query, sr=16000):
    try:
        wav_feat = getattr(dataset, f"wav_trim_{sr}")

        ref_feat = getattr(dataset, "spk_ref_mel_slices")

        wav = wav_feat.read_from_query(query)

        wav = preprocess_wav(wav, source_sr=sr)

        # Compute where to split the utterance into partials and pad the waveform
        # with zeros if the partial utterances cover a larger range.
        wav_slices, mel_slices = resemblyzer.VoiceEncoder.compute_partial_slices(
            len(wav), rate=1.3, min_coverage=0.75
        )
        max_wave_length = wav_slices[-1].stop
        if max_wave_length >= len(wav):
            wav = np.pad(wav, (0, max_wave_length - len(wav)), "constant")
        # Split the utterance into partials and forward them through the model
        spk_ref_mel = wav_to_mel_spectrogram(wav)
        spk_ref_mel_slices = [spk_ref_mel[s] for s in mel_slices]
        
        ref_feat.save(spk_ref_mel_slices, query)
    except:
        print(query)


def imap_extract_spk_ref_mel_slices_from_wav(task):
    extract_spk_ref_mel_slices_from_wav(*task)


def extract_spk_ref_mel_slices_from_wav_mp(dataset, queries, sr=16000, n_workers=1, chunksize=64):
    print("extract_spk_ref_mel_slices_from_wav_mp")
    n = len(queries)
    tasks = list(zip([dataset] * n, queries, [sr] * n))

    if n_workers == 1:
        for i in tqdm(range(n)):
            extract_spk_ref_mel_slices_from_wav(dataset, queries[i], sr)
        return
    
    with Pool(processes=n_workers) as pool:
        for i in tqdm(pool.imap(imap_extract_spk_ref_mel_slices_from_wav, tasks, chunksize=chunksize), total=n):
            pass


def segment2duration(dataset, query, featname, target_featname, inv_frame_period):
    try:
        segment_feat = getattr(dataset, featname)

        duration_feat = getattr(dataset, target_featname)

        segment = segment_feat.read_from_query(query)
        durations = []
        for (s, e) in segment:
            durations.append(int(np.round(e * inv_frame_period) - np.round(s * inv_frame_period)))
        
        duration_feat.save(durations, query)
    except:
        print(query)


def imap_segment2duration(task):
    segment2duration(*task)


def segment2duration_mp(dataset, queries, featname, target_featname, inv_frame_period, 
                            n_workers=1, chunksize=256, refresh=False):
    print("segment2duration_mp")
    n = len(queries)
    tasks = list(zip([dataset] * n, queries, [featname] * n, [target_featname] * n, [inv_frame_period] * n))

    if n_workers == 1:
        segment_feat = getattr(dataset, featname)
        segment_feat.read_all(refresh=refresh)
        for i in tqdm(range(n)):
            segment2duration(dataset, queries[i], featname, target_featname, inv_frame_period)
        return
    
    with Pool(processes=n_workers) as pool:
        for i in tqdm(pool.imap(imap_segment2duration, tasks, chunksize=chunksize), total=n):
            pass


def duration_avg_pitch_and_energy(dataset, query, featname):
    try:
        duration_feat = dataset.get_feature(featname)
        pitch_feat = dataset.get_feature("interpolate_pitch")
        energy_feat = dataset.get_feature("energy")

        avg_pitch_feat = dataset.get_feature(f"{featname}_avg_pitch")
        avg_energy_feat = dataset.get_feature(f"{featname}_avg_energy")

        durations = duration_feat.read_from_query(query)
        pitch = pitch_feat.read_from_query(query)
        energy = energy_feat.read_from_query(query)

        avg_pitch, avg_energy = pitch[:], energy[:]
        avg_pitch = representation_average(avg_pitch, durations)
        avg_energy = representation_average(avg_energy, durations)

        avg_pitch_feat.save(avg_pitch, query)
        avg_energy_feat.save(avg_energy, query)
    except:
        print(query)


def imap_duration_avg_pitch_and_energy(task):
    duration_avg_pitch_and_energy(*task)


def duration_avg_pitch_and_energy_mp(dataset, queries, featname, 
                                        n_workers=1, chunksize=256, refresh=False):
    print("duration_avg_pitch_and_energy_mp")
    n = len(queries)
    tasks = list(zip([dataset] * n, queries, [featname] * n))

    if n_workers == 1:
        duration_feat = dataset.get_feature(featname)
        pitch_feat = dataset.get_feature("interpolate_pitch")
        energy_feat = dataset.get_feature("energy")
        duration_feat.read_all(refresh=refresh)
        pitch_feat.read_all(refresh=refresh)
        energy_feat.read_all(refresh=refresh)
        for i in tqdm(range(n)):
            duration_avg_pitch_and_energy(dataset, queries[i], featname)
        return
    
    with Pool(processes=n_workers) as pool:
        for i in tqdm(pool.imap(imap_duration_avg_pitch_and_energy, tasks, chunksize=chunksize), total=n):
            pass


"""
No multiprocess version.
"""
# def wav_trim_16000_to_unsup_seg(dataset, queries):
#     print("wav_trim_16000_to_unsup_seg")
#     wav_feat = getattr(dataset, "wav_trim_16000")

#     segment_feat = getattr(dataset, "unsup_segment")

#     model = load_model_from_tag(ModelTag.BUCKEYE)

#     for query in tqdm(queries):
#         try:
#             wav = wav_feat.read_from_query(query)
#             boundaries = model.predict(wav)

#             segment = []
#             if boundaries[0] != 0:
#                 segment.append((0, boundaries[0]))
#             for i in range(len(boundaries) - 1):
#                 segment.append((boundaries[i], boundaries[i + 1]))
#             if boundaries[-1] != len(wav) / 16000:
#                 segment.append((boundaries[-1], len(wav) / 16000))
#             segment_feat.save(segment, query)
#         except:
#             print(query)


def get_stats(dataset, refresh=False):
    interp_pitch_feat = getattr(dataset, "interpolate_pitch")
    energy_feat = getattr(dataset, "energy")

    stats_path = getattr(dataset, "stats_path")

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
