import os
import glob
from tqdm import tqdm
from nnmnkwii.io import hts


label_dir = "/work/u7663915/Data/jsut-lab"  # change to local path
output_dir = "/work/u7663915/TTS-systems/preprocessed_data/JSUT/TextGrid/jsut"  # change to local path
os.makedirs(output_dir, exist_ok=True)
for d in os.listdir(label_dir):
    if not os.path.isdir(f"{label_dir}/{d}") or d[0] == '.':
        continue
    print(f"Parsing {d}...")
    for filename in tqdm(glob.glob(f"{label_dir}/{d}/lab/*.lab")):
        dst_path = f"{output_dir}/{os.path.basename(filename)[:-4]}.TextGrid"
        hts.write_textgrid(dst_path, hts.load(filename))
