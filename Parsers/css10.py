import os
from tqdm import tqdm


SPEAKERS = {
    "french": "css10-fr",
    "german": "css10-de",
    "spanish": "css10-es",
    "russian": "css10-ru",
}


class CSS10RawParser(object):
    def __init__(self, root):
        self.root = root
        self.data = None

    def parse(self):
        self.data = {"data": [], "data_info": [], "all_speakers": []}
        path = f"{self.root}/transcript.txt"
        speaker = SPEAKERS[self.root.split('/')[-1]]
        self.data["all_speakers"].append(speaker)
        with open(path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                if line == '\n':
                    continue
                wav_name, _, text, _ = line.strip().split('|')
                wav_path = f"{self.root}/{wav_name}"
                basename = wav_name.split('/')[-1][:-4]
                if os.path.isfile(wav_path):
                    data = {
                        "wav_path": wav_path,
                        "text": text,
                    }
                    data_info = {
                        "spk": speaker,
                        "basename": f"{speaker}-{basename.replace('_', '-')}",
                    }
                    self.data["data"].append(data)
                    self.data["data_info"].append(data_info)
                else:
                    print("transcript.txt should not contain non-exist wav files, data might be corrupted.")
                    print(f"Can not find {wav_path}.")
