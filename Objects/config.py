from typing import Dict
import yaml


class LanguageDataConfigReader(object):
    def __init__(self):
        pass

    def read(self, root):
        config = yaml.load(open(f"{root}/config.yaml", "r"), Loader=yaml.FullLoader)
        self.name = config["dataset"]
        self.lang_id = config["lang_id"]
        self.data_dir = config["preprocessed_path"]
        self.subsets = {
            "train": f"{root}/{config['subsets']['train']}",
            "val": f"{root}/{config['subsets']['val']}",
            "test": f"{root}/{config['subsets']['test']}",
        }
        return {
            "name": self.name,
            "lang_id": self.lang_id,
            "data_dir": self.data_dir,
            "subsets": self.subsets,
            "text_cleaners": ["transliteration_cleaners"],
        }
