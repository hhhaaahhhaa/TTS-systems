import os
from tqdm import tqdm


class LibriTTSRawParser(object):
    def __init__(self, root):
        self.root = root
        self.data = None
        self.dsets = [
            "train-clean-100",
            "train-clean-360",
            "train-other-500",
            "dev-clean",
            "dev-other",
            "test-clean",
            "test-other",
        ]

    def parse(self):
        self.data = {"data": [], "data_info": [], "all_speakers": []}
        for dset in self.dsets:
            if not os.path.isdir(f"{self.root}/{dset}"):
                continue
            for speaker in tqdm(os.listdir(f"{self.root}/{dset}"), desc=dset):
                self.data["all_speakers"].append(speaker)
                for chapter in os.listdir(f"{self.root}/{dset}/{speaker}"):
                    for filename in os.listdir(f"{self.root}/{dset}/{speaker}/{chapter}"):
                        if filename[-4:] != ".wav":
                            continue
                        basename = filename[:-4]
                        wav_path = f"{self.root}/{dset}/{speaker}/{chapter}/{basename}.wav"
                        text_path = f"{self.root}/{dset}/{speaker}/{chapter}/{basename}.normalized.txt"
                        with open(text_path, "r", encoding="utf-8") as f:
                            text = f.readline().strip("\n")
                        data = {
                            "wav_path": wav_path,
                            "text": text,
                        }
                        data_info = {
                            "spk": speaker,
                            "basename": basename,
                            "dset": dset,
                            "chapter": chapter,
                        }
                        self.data["data"].append(data)
                        self.data["data_info"].append(data_info)
