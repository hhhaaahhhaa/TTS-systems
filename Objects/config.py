from typing import Dict
import yaml


class DataConfigReader(object):
    def __init__(self):
        pass

    def read(self, root):
        config = yaml.load(open(f"{root}/config.yaml", "r"), Loader=yaml.FullLoader)
        for k in config['subsets']:
            config['subsets'][k] = f"{root}/{config['subsets'][k]}"

        return config
