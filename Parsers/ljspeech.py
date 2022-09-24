import os
from tqdm import tqdm


class LJSpeechRawParser(object):
    def __init__(self, root):
        self.root = root
        self.data = None

    def parse(self):
        self.data = {"data": [], "data_info": [], "all_speakers": ["LJSpeech"]}
        path = f"{self.root}/metadata.csv"
        speaker = "LJSpeech"
        with open(path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                if line == "\n":
                    continue
                wav_name, _, text = line.strip().split("|")
                if text[-1].isalpha():  # add missing periods
                    text += '.'
                wav_path = f"{self.root}/wavs/{wav_name}.wav"
                if os.path.isfile(wav_path):
                    basename = wav_name
                    data = {
                        "wav_path": wav_path,
                        "text": text,
                    }
                    data_info = {
                        "spk": speaker,
                        "basename": basename,
                    }
                    self.data["data"].append(data)
                    self.data["data_info"].append(data_info)
                else:
                    print("metadata.csv should not contain non-exist wav files, data might be corrupted.")
                    print(f"Can not find {wav_path}.")           
