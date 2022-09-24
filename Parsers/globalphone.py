import os
import glob
from tqdm import tqdm


SPEAKERS = {
    "French": "gp-fr",
    "German": "gp-de",
    "Spanish": "gp-es",
    "Czech": "gp-cz",
}


class GlobalPhoneRawParser(object):
    def __init__(self, root):
        self.root = root
        self.data = None

    def parse(self):
        self.data = {"data": [], "data_info": [], "all_speakers": []}
        corpus_dir = f"{self.root}/corpus"
        wav_dir = f"{self.root}/wav"
        for wav_path in tqdm(glob.glob(f"{wav_dir}/*.wav")):
            basename = os.path.basename(wav_path)[:-4]
            speaker = basename.split('_')[0]
            if speaker not in self.data["all_speakers"]:
                self.data["all_speakers"].append(speaker)
            text_path = f"{corpus_dir}/{basename}.lab"
            with open(text_path, 'r', encoding='utf-8') as f:
                text = f.readline().strip("\n")
            data = {
                "wav_path": wav_path,
                "text": text,
            }
            data_info = {
                "spk": speaker,
                "basename": basename.replace('_', '-'),
            }
            self.data["data"].append(data)
            self.data["data_info"].append(data_info)
