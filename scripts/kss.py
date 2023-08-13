import os
from tqdm import tqdm

from dlhlp_lib.text.utils import remove_punctuation

from scripts.KoG2P.g2p import g2p_ko


def generate_dictionary(raw_dir, dictionary_path):
    dictionary_path = "MFA/kss/lexicon.txt"
    os.makedirs(os.path.dirname(dictionary_path), exist_ok=True)
    lexicons = {}
    with open(f"{raw_dir}/transcript.v.1.4.txt", 'r', encoding="utf-8") as f:
        for line in tqdm(f):
            if line == "\n":
                continue
            wav, _, text, _, _, _ = line.strip().split("|")
            text = remove_punctuation(text)
            for t in text.split(" "):
                if t not in lexicons:
                    lexicons[t] = g2p_ko(t)

    with open(dictionary_path, 'w', encoding="utf-8") as f:
        for k, v in lexicons.items():
            f.write(f"{k}\t{v}\n")
    print(f"Write {len(lexicons)} words.")

if __name__ == "__main__":
    raw_dir = "/work/u7663915/Data/kss"  # change to local path
    mfa_data_dir = "preprocessed_data/kss/mfa_data"
    dictionary_path = "MFA/kss/lexicon.txt"
    acoustic_model_path = "MFA/kss/acoustic_model.zip"

    generate_dictionary(raw_dir, dictionary_path)

    # Prepared by TA
    # cmd = f"mfa train {mfa_data_dir} {dictionary_path} {acoustic_model_path} -j 8 -v --clean"
    # os.system(cmd)
