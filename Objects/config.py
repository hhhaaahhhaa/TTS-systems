from typing import Dict
import yaml


class DataConfigReader(object):
    def __init__(self):
        pass

    def read(self, root):
        config = yaml.load(open(f"{root}/config.yaml", "r"), Loader=yaml.FullLoader)
        if "lang_id" not in config:
            config["lang_id"] = "en"
        if "symbol_id" not in config:
            if "n_symbols" in config:
                config["symbol_id"] = config["unit_name"]
                config["use_real_phoneme"] = False
            else:
                config["symbol_id"] = config["lang_id"]
                config["use_real_phoneme"] = True
        for k in config['subsets']:
            config['subsets'][k] = f"{root}/{config['subsets'][k]}"

        return config
