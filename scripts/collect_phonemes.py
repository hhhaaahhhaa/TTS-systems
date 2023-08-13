import os
from typing import List, Set
from tqdm import tqdm

from Parsers.parser import DataParser


def collect_phonemes(data_dirs) -> Set[str]:
    all_phns = set()
    for data_dir in data_dirs:
        data_parser = DataParser(data_dir)
        queries = data_parser.get_all_queries()
        data_parser.units["mfa"].phoneme.read_all()
        for query in tqdm(queries):
            try:
                phn_seq = data_parser.units["mfa"].phoneme.read_from_query(query)
                phn_seq = phn_seq.strip().split()
                all_phns.update(phn_seq)
            except:
                pass
    return all_phns


def generate_phoneme_set(phns: Set[str], output_path: str) -> None:
    phns = list(phns)
    phns.sort()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for p in phns:
            f.write(f"{p}\n")


if __name__ == "__main__":
    phns = collect_phonemes([
        "preprocessed_data/JSUT"
    ])
    generate_phoneme_set(phns, "_phoneset/JSUT.txt")
