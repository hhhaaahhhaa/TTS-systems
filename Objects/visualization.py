import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, TypedDict


class MatchingGraphInfo(TypedDict):
    title: str
    x_labels: List[str]
    y_labels: List[str]
    attn: np.array
    quantized: bool


class CodebookAnalyzer(object):
    def __init__(self, output_dir):
        self.root = output_dir

    def visualize_tsne(self):
        # TODO: Not done.
        os.makedirs(f"{self.root}/codebook/tsne", exist_ok=True)
    
    def plot_tsne(self):
        pass

    def plot_matching(self, info, quantized=False):
        if not quantized:
            fig = plt.figure(figsize=(32, 16))
            ax = fig.add_subplot(111)
            cax = ax.matshow(info["attn"])
            fig.colorbar(cax, ax=ax)

            ax.set_title(info["title"], fontsize=28)
            ax.set_xticks(np.arange(len(info["x_labels"])))
            ax.set_xticklabels(info["x_labels"], rotation=90, fontsize=8)
            ax.set_yticks(np.arange(len(info["y_labels"])))
            ax.set_yticklabels(info["y_labels"], fontsize=8)
        else:
            fig = plt.figure(figsize=(16, 12))
            ax = fig.add_subplot(111)
            column_labels=["Code Index", "Phonemes"]
            ax.axis('off')
            code2phn = {x: [] for x in info["x_labels"]}
            max_positions = np.argmax(info["attn"], axis=1)
            for phn, pos in zip(info["y_labels"], max_positions):
                code2phn[info["x_labels"][int(pos)]].append(phn)
            data = [[k, ", ".join(v)] for k, v in code2phn.items() if len(v) > 0]

            ax.set_title(info["title"], fontsize=28)
            ax.table(cellText=data,colLabels=column_labels, loc="center", fontsize=12)
        return fig

    def visualize_matching(self, idx, infos):
        os.makedirs(f"{self.root}/codebook/matching", exist_ok=True)
        for info in infos:
            with open(f"{self.root}/codebook/matching/{idx:03d}-{info['title']}.npy", 'wb') as f:
                print(info["attn"].shape)
                np.save(f, info["attn"])
            if not info["quantized"]:
                fig = self.plot_matching(info, False)
                fig.savefig(f"{self.root}/codebook/matching/{idx:03d}-{info['title']}.jpg")
            else:
                fig = self.plot_matching(info, True)
                fig.savefig(f"{self.root}/codebook/matching/{idx:03d}-{info['title']}-table.jpg")
            plt.close(fig)

    def visualize_phoneme_transfer(self, idx, infos):
        os.makedirs(f"{self.root}/codebook/phoneme-transfer", exist_ok=True)
        for info in infos:
            fig = self.plot_matching(info, False)
            fig.savefig(f"{self.root}/codebook/phoneme-transfer/{idx:03d}-{info['title']}.jpg")
            plt.close(fig)

    def visualize_phoneme_mapping(self, dst, attns_map):
        os.makedirs(dst, exist_ok=True)
        from text.define import LANG_ID2NAME
        for src_id, src_v in attns_map.items():
            for target_id, target_v in attns_map.items():
                assert src_v["attn"].shape[1] == 512, f"{src_v['attn'].shape}"
                assert target_v["attn"].shape[1] == 512, f"{target_v['attn'].shape}"
                cross_attn = np.matmul(src_v["attn"], target_v["attn"].T) / 4
                info = MatchingGraphInfo({
                    "title": f"{LANG_ID2NAME[src_id]}-{LANG_ID2NAME[target_id]}",
                    "y_labels": src_v["y-labels"],
                    "x_labels": target_v["y-labels"],
                    "attn": cross_attn,
                    "quantized": False,
                })
                
                if target_id == src_id:
                    pass
                else:
                    for i in range(cross_attn.shape[0]):
                        for j in range(cross_attn.shape[1]):
                            with open(f"{dst}/similarity.txt", 'a', encoding='utf-8') as f:       
                                x1 = src_v["y-labels"][i]
                                x2 = target_v["y-labels"][j]
                                f.write(f"{LANG_ID2NAME[src_id]}-{x1} {LANG_ID2NAME[target_id]}-{x2} {float(cross_attn[i][j])}\n")

                fig = self.plot_matching(info, False)
                fig.savefig(f"{dst}/{LANG_ID2NAME[src_id]}-{LANG_ID2NAME[target_id]}.jpg")
                plt.close(fig)
