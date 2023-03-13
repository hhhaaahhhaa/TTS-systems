import numpy as np
import json

import Define
from Parsers.parser import DataParser


def merge_stats(stats_dict, keys):
    num = len(keys)
    pmi, pmx, pmu, pstd, emi, emx, emu, estd = \
        np.finfo(np.float64).max, np.finfo(np.float64).min, 0.0, 0.0, np.finfo(np.float64).max, np.finfo(np.float64).min, 0.0, 0.0
    for k in keys:
        pmu += stats_dict[k][2]
        pstd += stats_dict[k][3] ** 2
        emu += stats_dict[k][6]
        estd += stats_dict[k][7] ** 2
        pmi = min(pmi, stats_dict[k][0])
        pmx = max(pmx, stats_dict[k][1])
        emi = min(emi, stats_dict[k][4])
        emx = max(emx, stats_dict[k][5])

    pmu, pstd, emu, estd = pmu / num, (pstd / num) ** 0.5, emu / num, (estd / num) ** 0.5
    
    return [pmi, pmx, pmu, pstd, emi, emx, emu, estd]


def setup_data(data_configs):
    keys = []
    for data_config in data_configs:
        Define.DATAPARSERS[data_config["name"]] = DataParser(data_config["data_dir"])
        data_parser = Define.DATAPARSERS[data_config["name"]]
        with open(data_parser.stats_path) as f:
            stats = json.load(f)
            stats = stats["pitch"] + stats["energy"]
            Define.ALLSTATS[data_config["name"]] = stats
            keys.append(data_config["name"])

    Define.ALLSTATS["global"] = merge_stats(Define.ALLSTATS, keys)
    if Define.DEBUG:
        print("Initialize data parsers and build normalization stats, done.")
        input()
