import os
from tqdm import tqdm


class JSUTRawParser(object):
    def __init__(self, root):
        self.root = root
        self.data = None

    def parse(self):
        self.data = {"data": [], "data_info": [], "all_speakers": ["jsut"]}
        path = f"{self.root}/basic5000/transcript_utf8.txt"
        speaker = "jsut"
        with open(path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                if line == '\n':
                    continue
                basename, text = line.strip().split(":")
                wav_path = f"{self.root}/basic5000/wav/{basename}.wav"

                if os.path.isfile(wav_path):
                    data = {
                        "wav_path": wav_path,
                        "text": text,
                    }
                    data_info = {
                        "spk": speaker,
                        "basename": basename,
                        "dset": "basic5000",
                    }
                    self.data["data"].append(data)
                    self.data["data_info"].append(data_info)
                else:
                    print("transcript.txt should not contain non-exist wav files, data might be corrupted.")
                    print(f"Can not find {wav_path}.")
