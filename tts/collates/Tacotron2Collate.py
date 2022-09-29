import torch
import numpy as np
from functools import partial

from text.define import LANG_ID2SYMBOLS


class Tacotron2Collate(object):
    def __init__(self, n_frames_per_step=2):
        # calculate re-id increment
        increment = 0
        self.re_id_increment = {}
        for k, v in LANG_ID2SYMBOLS.items():
            self.re_id_increment[k] = increment
            increment += len(v)
        self.n_symbols = increment

        self.n_frames_per_step = n_frames_per_step

    def collate_fn(self, re_id=False):
        return partial(self._collate_fn, re_id=re_id)

    def _collate_fn(self, data, re_id=False):
        data_size = len(data)
        idx_arr = np.arange(data_size)

        # if multiple languages are used, concat embedding layers and re-id each phoneme
        if re_id:
            for idx in idx_arr:
                data[idx]["text"] += self.re_id_increment[data[idx]["lang_id"]]
        output = reprocess(data, n_frames_per_step=self.n_frames_per_step)

        return output
    

def reprocess(batch, n_frames_per_step):
    """
    Pad data.
    Test version has only text and speaker data.
    Args:
        batch: data batch returned from dataset:
            "id": basename,
            "speaker": speaker_id,
            "text": text,
            "raw_text": raw_text,
            "mel": mel,
            "lang_id": self.lang_id,

        n_frames_per_step: generated frames at every time step, 1 is very hard to train, better set to > 1
        mode: "train" or "test"
    """
    # Right zero-pad all one-hot text sequences to max input length
    input_lengths, ids_sorted_decreasing = torch.sort(
        torch.LongTensor([len(x["text"]) for x in batch]),
        dim=0, descending=True)
    max_input_len = input_lengths[0]

    text_padded = torch.LongTensor(len(batch), max_input_len)
    text_padded.zero_()
    for i in range(len(ids_sorted_decreasing)):
        text = batch[ids_sorted_decreasing[i]]["text"]
        text_padded[i, :len(text)] = torch.from_numpy(text).long()

    # Right zero-pad mel-spec
    num_mels = batch[0]["mel"].shape[0]
    max_target_len = max([x["mel"].shape[1] for x in batch])
    if max_target_len % n_frames_per_step != 0:
        max_target_len += n_frames_per_step - max_target_len % n_frames_per_step
        assert max_target_len % n_frames_per_step == 0

    # include mel padded and gate padded
    mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
    mel_padded.zero_()
    gate_padded = torch.FloatTensor(len(batch), max_target_len)
    gate_padded.zero_()
    output_lengths = torch.LongTensor(len(batch))
    spks = []
    lang_ids = []
    for i in range(len(ids_sorted_decreasing)):
        mel = batch[ids_sorted_decreasing[i]]["mel"]
        mel_padded[i, :, :mel.shape[1]] = torch.from_numpy(mel)
        gate_padded[i, mel.shape[1]-1:] = 1
        output_lengths[i] = mel.shape[1]
        spks.append(batch[ids_sorted_decreasing[i]]["speaker"])
        lang_ids.append(batch[ids_sorted_decreasing[i]]["lang_id"])
    spks = torch.LongTensor(spks)

    return text_padded, input_lengths, mel_padded, gate_padded, output_lengths, spks, lang_ids
