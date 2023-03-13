from typing import Dict
import yaml


class DataConfigReader(object):
    def __init__(self):
        pass

    def read(self, root):
        config = yaml.load(open(f"{root}/config.yaml", "r"), Loader=yaml.FullLoader)
        config["symbol_id"] = config["lang_id"]
        for k in config['subsets']:
            config['subsets'][k] = f"{root}/{config['subsets'][k]}"

        return config
