import torch
import numpy as np
import json


DEBUG = True
CUDA_LAUNCH_BLOCKING = True
MAX_WORKERS = 2
DATAPARSERS = {}
ALLSTATS = {}
LOGGER = "tb"  # "tb", "comet", ""
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def merge_stats(stats_dict, keys):
    num = len(keys)
    pmi, pmx, pmu, pstd, emi, emx, emu, estd = \
        np.finfo(np.float64).max, np.finfo(np.float64).min, 0.0, 0.0, np.finfo(np.float64).max, np.finfo(np.float64).min, 0.0, 0.0
    for k in keys:
        pmu += stats_dict[k][2]
        pstd += stats_dict[k][3] ** 2
        emu += stats_dict[k][6]
        estd += stats_dict[k][7] ** 2
        pmi = min(pmi, stats_dict[k][0] * stats_dict[k][3] + stats_dict[k][2])
        pmx = max(pmx, stats_dict[k][1] * stats_dict[k][3] + stats_dict[k][2])
        emi = min(emi, stats_dict[k][4] * stats_dict[k][7] + stats_dict[k][6])
        emx = max(emx, stats_dict[k][5] * stats_dict[k][7] + stats_dict[k][6])

    pmu, pstd, emu, estd = pmu / num, (pstd / num) ** 0.5, emu / num, (estd / num) ** 0.5
    pmi, pmx, emi, emx = (pmi - pmu) / pstd, (pmx - pmu) / pstd, (emi - emu) / estd, (emx - emu) / estd
    
    return [pmi, pmx, pmu, pstd, emi, emx, emu, estd]
