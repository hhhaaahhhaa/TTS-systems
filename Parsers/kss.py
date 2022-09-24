import os
from tqdm import tqdm


class KSSRawParser(object):
    def __init__(self, root):
        self.root = root
        self.data = None

    def parse(self):
        self.data = {"data": [], "data_info": [], "all_speakers": ["kss"]}
        path = f"{self.root}/transcript.v.1.4.txt"
        speaker = "kss"
        with open(path, 'r', encoding="utf-8") as f:
            for line in tqdm(f):
                if line == "\n":
                    continue
                wav_name, _, text, _, _, en_text = line.strip().split("|")
                wav_path = f"{self.root}/{wav_name}"
                if os.path.isfile(wav_path):
                    basename = wav_name.split('/')[-1][:-4]
                    data = {
                        "wav_path": wav_path,
                        "text": text,
                        "en_text": en_text,
                    }
                    data_info = {
                        "spk": speaker,
                        "basename": f"kss-{basename.replace('_', '-')}",
                    }
                    self.data["data"].append(data)
                    self.data["data_info"].append(data_info)
                else:
                    print("transcript.txt should not contain non-exist wav files, data might be corrupted.")
                    print(f"Can not find {wav_path}.")
