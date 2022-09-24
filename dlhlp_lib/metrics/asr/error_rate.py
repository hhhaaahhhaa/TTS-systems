import numpy as np
from tqdm import tqdm
from typing import Dict, Union
import jiwer
from jiwer.transformations import wer_default

from dlhlp_lib.parsers.Interfaces import BaseDataParser


def segment2duration(segment, fp):
    res = []
    for (s, e) in segment:
        res.append(
            int(
                round(round(e * 1 / fp, 4))
                - round(round(s * 1 / fp, 4))
            )
        )
    return res


def expand(seq, dur):
    assert len(seq) == len(dur)
    res = []
    for (x, d) in zip(seq, dur):
        if d > 0:
            res.extend([x] * d)
    return res


class FERCalculator(object):
    """
    FER calculation, allow phonemes from different sources. Need to pass unify mappings.
    """
    def __init__(self):
        pass

    def exec(self,
            data_parser: BaseDataParser, 
            queries,
            ref_phoneme_featname: str, ref_segment_featname: str,
            pred_phoneme_featname: str, pred_segment_featname: str,
            symbol_ref2unify: Dict, symbol_pred2unify: Dict,
            fp: float
        ) -> float:
        ref_phn_feat = data_parser.get_feature(ref_phoneme_featname)
        ref_seg_feat = data_parser.get_feature(ref_segment_featname)
        pred_phn_feat = data_parser.get_feature(pred_phoneme_featname)
        pred_seg_feat = data_parser.get_feature(pred_segment_featname)

        n_frames, correct = 0, 0
        ref_n_seg, pred_n_seg = 0, 0
        for query in tqdm(queries):
            try:
                ref_phoneme = ref_phn_feat.read_from_query(query).strip().split(" ")
                ref_segment = ref_seg_feat.read_from_query(query)
                pred_phoneme = pred_phn_feat.read_from_query(query).strip().split(" ")
                pred_segment = pred_seg_feat.read_from_query(query)
            except:
                continue

            ref_n_seg += len(ref_phoneme)
            pred_n_seg += len(pred_phoneme)

            ref_duration, pred_duration = segment2duration(ref_segment, fp), segment2duration(pred_segment, fp)
            ref_seq, pred_seq = expand(ref_phoneme, ref_duration), expand(pred_phoneme, pred_duration)
            if len(pred_seq) >= len(ref_seq):
                pred_seq = pred_seq[:len(ref_seq)]
            else:
                padding = [pred_seq[-1]] * (len(ref_seq) - len(pred_seq))
                pred_seq.extend(padding)
            assert len(pred_seq) == len(ref_seq)

            for (x1, x2) in zip(ref_seq, pred_seq):
                if symbol_ref2unify[x1] == symbol_pred2unify[x2]:
                    correct += 1
            n_frames += len(ref_seq)
        facc = correct / n_frames
        fer = 1 - facc

        print(f"Segments: {ref_n_seg}, {pred_n_seg}.")
        print(f"Frame error rate: 1 - {correct}/{n_frames} = {fer * 100:.2f}%")
        return fer


class PERCalculator(object):
    """
    PER calculation, allow phonemes from different sources. Need to pass unify mappings.
    Note that PER is not symmetric.
    """
    def __init__(self):
        pass

    def exec(self,
            data_parser: BaseDataParser, 
            queries,
            ref_phoneme_featname: str, pred_phoneme_featname: str,
            symbol_ref2unify: Dict, symbol_pred2unify: Dict,
            return_dict: bool=False
        ) -> Union[float, Dict]:
        ref_phn_feat = data_parser.get_feature(ref_phoneme_featname)
        pred_phn_feat = data_parser.get_feature(pred_phoneme_featname)

        wer_list = []
        substitutions, insertions, deletions = 0, 0, 0
        for query in tqdm(queries):
            try:
                ref_phoneme = ref_phn_feat.read_from_query(query).strip().split(" ")
                pred_phoneme = pred_phn_feat.read_from_query(query).strip().split(" ")
            except:
                continue

            ref_sentence = " ".join([symbol_ref2unify[p] for p in ref_phoneme])
            pred_sentence = " ".join([symbol_pred2unify[p] for p in pred_phoneme])

            if return_dict:
                measures = jiwer.compute_measures(ref_sentence, pred_sentence, wer_default, wer_default, return_dict=return_dict)
                wer_list.append(measures['wer'])
                substitutions += measures['substitutions']
                insertions += measures['insertions']
                deletions += measures['deletions']
            else:
                wer_list.append(measures)
        wer = sum(wer_list) / len(wer_list)

        print(f"Word error rate: {wer * 100:.2f}%")
        if return_dict:
            return {
                'wer': wer,
                'substitutions': substitutions,
                'insertions': insertions,
                'deletions': deletions
            }
        else:
            return wer
