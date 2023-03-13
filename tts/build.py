"""
Global setup from data_configs.
"""
from text.define import LANG_ID2SYMBOLS
from text.symbols import common_symbols
import Define
from Parsers.parser import DataParser


def build_id2symbols(data_configs):
    id2symbols = {}
    
    for data_config in data_configs:
        if data_config["symbol_id"] not in id2symbols:
            if data_config["symbol_id"] in LANG_ID2SYMBOLS:
                id2symbols[data_config["symbol_id"]] = LANG_ID2SYMBOLS[data_config["symbol_id"]]
            else:  # units which are not phonemes
                id2symbols[data_config["symbol_id"]] = common_symbols + [str(idx) for idx in range(data_config["n_symbols"])]
    
    return id2symbols


def build_data_parsers(data_configs):
    for data_config in data_configs:
        Define.DATAPARSERS[data_config["name"]] = DataParser(data_config["data_dir"])


def build_all_speakers(data_configs):
    res = []
    for data_config in data_configs:
        dataset_name = data_config["name"]
        if dataset_name not in Define.DATAPARSERS:
            Define.DATAPARSERS[dataset_name] = DataParser(data_config["data_dir"])
        spks = Define.DATAPARSERS[dataset_name].get_all_speakers()
        res.extend(spks)
    return res
