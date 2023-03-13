
import torch
import numpy as np
from functools import partial

from text.define import LANG_NAME2ID
from tts.build import build_id2symbols, build_all_speakers
from tts.utils.tool import pad_1D, pad_2D


class FastSpeech2Collate(object):
    def __init__(self, data_configs):
        # calculate re-id increment
        id2symbols = build_id2symbols(data_configs)
        increment = 0
        self.re_id_increment = {}
        for k, v in id2symbols.items():
            self.re_id_increment[k] = increment
            increment += len(v)
        self.n_symbols = increment

        # calculate speaker map
        speakers = build_all_speakers(data_configs)
        self.speaker_map = {spk: i for i, spk in enumerate(speakers)}

    def collate_fn(self, sort=False, re_id=False, mode="train"):
        return partial(self._collate_fn, sort=sort, re_id=re_id, mode=mode)

    def _collate_fn(self, data, sort=False, re_id=False, mode="train"):
        data_size = len(data)

        if sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        # if multiple languages are used, concat embedding layers and re-id each phoneme
        if re_id:
            for idx in idx_arr:
                data[idx]["text"] += self.re_id_increment[data[idx]["lang_id"]]
        
        # remap speakers and language
        for idx in idx_arr:
            data[idx]["speaker"] = self.speaker_map[data[idx]["speaker"]]
            data[idx]["lang_id"] = LANG_NAME2ID[data[idx]["lang_id"]]
        
        output = reprocess(data, idx_arr, mode=mode)

        return output


def reprocess(data, idxs, mode="train"):
    """
    Pad data.
    Test version has only text and speaker data.
    Args:
        data: data batch returned from dataset:
            "id": basename,
            "speaker": speaker_id,
            "text": text,
            "raw_text": raw_text,
            "mel": mel,
            "pitch": pitch,
            "energy": energy,
            "duration": duration,
            "lang_id": self.lang_id,

        mode: "train" or "test"
    """
    ids = [data[idx]["id"] for idx in idxs]
    lang_ids = [data[idx]["lang_id"] for idx in idxs]
    speakers = [data[idx]["speaker"] for idx in idxs]
    speakers = np.array(speakers)

    if mode in ["train", "test"]:
        texts = [data[idx]["text"] for idx in idxs]
        raw_texts = [data[idx]["raw_text"] for idx in idxs]
        text_lens = np.array([text.shape[0] for text in texts])
        texts = pad_1D(texts)

    if mode in ["train"]:
        mels = [data[idx]["mel"] for idx in idxs]
        pitches = [data[idx]["pitch"] for idx in idxs]
        energies = [data[idx]["energy"] for idx in idxs]
        durations = [data[idx]["duration"] for idx in idxs]
        mel_lens = np.array([mel.shape[0] for mel in mels])

    if mode in ["train"]:
        mels = pad_2D(mels)
        pitches = pad_1D(pitches)
        energies = pad_1D(energies)
        durations = pad_1D(durations)

    speaker_args = torch.from_numpy(speakers).long()

    if mode == "train":
        return (
            ids,
            raw_texts,
            speaker_args,
            torch.from_numpy(texts).long(),
            torch.from_numpy(text_lens),
            max(text_lens),
            torch.from_numpy(mels).float(),
            torch.from_numpy(mel_lens),
            max(mel_lens),
            torch.from_numpy(pitches).float(),
            torch.from_numpy(energies),
            torch.from_numpy(durations).long(),
            lang_ids
        )
    elif mode == "test":
        return (
            ids,
            raw_texts,
            speaker_args,
            torch.from_numpy(texts).long(),
            torch.from_numpy(text_lens),
            max(text_lens),
            None, 
            None, 
            None,
            None,
            None,
            None,
            lang_ids,
        )
    else:
        raise NotImplementedError
