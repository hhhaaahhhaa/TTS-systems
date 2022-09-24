from tqdm import tqdm
from typing import Dict

from dlhlp_lib.parsers.Interfaces import BaseDataParser


class SegmentationEvaluator(object):
    """
    Given boundaries, evaluate 4 metrics: recall, precision, os (over segmentation score), r-value.
    (https://arxiv.org/pdf/2007.13465.pdf)
    """
    def __init__(self):
        pass

    def exec(self,
            data_parser: BaseDataParser, 
            queries,
            ref_segment_featname: str, pred_segment_featname: str,
            threshold=0.04
        ) -> Dict[str, float]:
        ref_segment_feat = data_parser.get_feature(ref_segment_featname)
        pred_segment_feat = data_parser.get_feature(pred_segment_featname)

        # Double cursor algorithm, linear time complexity.
        tp, fn, fp = 0, 0, 0
        for query in tqdm(queries):
            try:
                ref_segment = ref_segment_feat.read_from_query(query)
                pred_segment = pred_segment_feat.read_from_query(query)
                ref_len, pred_len = len(ref_segment), len(pred_segment)
            except:
                continue
            
            ref_pos, pred_pos = 0, 0
            ref_offset, pred_offset = ref_segment[0][0], pred_segment[0][0]  # segment may not start from time 0 due to trimming.
            while 1:
                if ref_pos == ref_len:
                    fp += pred_len - pred_pos
                    break
                if pred_pos == pred_len:
                    fn += ref_len - ref_pos
                    break
                t_ref, t_pred = ref_segment[ref_pos][0] - ref_offset, pred_segment[pred_pos][0] - pred_offset
                if abs(t_ref - t_pred) <= threshold:  # matched!
                    tp += 1
                    ref_pos += 1
                    pred_pos += 1
                elif t_ref > t_pred:
                    pred_pos += 1
                    fp += 1
                else:
                    ref_pos += 1
                    fn += 1
        
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        os = recall / precision - 1
        r1 = (os ** 2 + (1 - recall) ** 2) ** 0.5
        r2 =  (recall / precision - recall) / (2 ** 0.5)
        r_val = (2 - r1 - r2) / 2

        print(f"Recall: {recall * 100:.2f}%")
        print(f"Precision: {precision * 100:.2f}%")
        print(f"OS: {os * 100:.2f}")
        print(f"R-val: {r_val * 100:.2f}")
        return {
            "recall": recall,
            "precision": precision,
            "os": os,
            "r-value": r_val,
        }
