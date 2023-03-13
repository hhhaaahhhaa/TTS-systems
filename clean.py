import argparse
import os
import torch
from tqdm import tqdm
import json
import gc

from Parsers.parser import DataParser


class LengthChecker(object):
    def __init__(self, data_parser: DataParser, mi=1, mx=15):
        self.data_parser = data_parser
        self.data_parser.units["mfa"].segment.read_all()
        self.mi = mi
        self.mx = mx

    def check(self, query) -> bool:
        try:
            segment = self.data_parser.units["mfa"].segment.read_from_query(query)
            l = segment[-1][1] - segment[0][0]
            assert self.mi <= l and l <= self.mx 
        except:
            return False
        return True


class ExistenceChecker(object):
    def __init__(self, data_parser: DataParser):
        self.data_parser = data_parser

    def check(self, query) -> bool:
        try:
            filenames = [
                self.data_parser.mel.read_filename(query, raw=True),
                self.data_parser.units["mfa"].duration_avg_energy.read_filename(query, raw=True),
                self.data_parser.units["mfa"].duration_avg_pitch.read_filename(query, raw=True),
            ]
            for f in filenames:
                assert os.path.exists(f)
        except:
            return False
        return True


class UnknownTokenChecker(object):
    def __init__(self, data_parser: DataParser):
        self.data_parser = data_parser
        self.data_parser.units["mfa"].phoneme.read_all()

    def check(self, query) -> bool:
        try:
            phoneme = self.data_parser.units["mfa"].phoneme.read_from_query(query)
            assert "spn" not in phoneme.split(" ")
        except:
            return False
        return True


def clean(root: str, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data_parser = DataParser(root)
    res = data_parser.get_all_queries()

    print("Check existence...")
    filtered = []
    checker = ExistenceChecker(data_parser)
    for query in tqdm(res):
        if checker.check(query):
            filtered.append(query)
    res = filtered
    print(f"Remain samples: {len(res)}.")

    print("Check length...")
    filtered = []
    checker = LengthChecker(data_parser)
    for query in tqdm(res):
        if checker.check(query):
            filtered.append(query)
    res = filtered
    print(f"Remain samples: {len(res)}.")

    print("Check unknown tokens (spn)...")
    filtered = []
    checker = UnknownTokenChecker(data_parser)
    for query in tqdm(res):
        if checker.check(query):
            filtered.append(query)
    res = filtered
    print(f"Remain samples: {len(res)}.")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=4)


def main(args):
    """ Write results to args.output_path """
    clean(args.preprocessed_data_dir, args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("preprocessed_data_dir", type=str)
    parser.add_argument("output_path", type=str)
    args = parser.parse_args()

    main(args)
