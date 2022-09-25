import numpy as np
from functools import partial

from text.define import LANG_ID2SYMBOLS
from .utils import reprocess


class LanguageCollate(object):
    """
    For baseline multilingual FastSpeech2 training.
    """
    def __init__(self):
        # calculate re-id increment
        increment = 0
        self.re_id_increment = {}
        for k, v in LANG_ID2SYMBOLS.items():
            self.re_id_increment[k] = increment
            increment += len(v)
        self.n_symbols = increment

    def collate_fn(self, sort=False, re_id=False):
        return partial(self._collate_fn, sort=sort, re_id=re_id)

    def _collate_fn(self, data, sort=False, re_id=False):
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
        output = reprocess(data, idx_arr)

        return output
