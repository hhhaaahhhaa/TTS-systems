import os
import json
from typing import Dict

from dlhlp_lib.parsers.Interfaces import BaseDataParser
from dlhlp_lib.parsers.Feature import Feature
from dlhlp_lib.parsers.QueryParsers import SFQueryParser, NestSFQueryParser
from dlhlp_lib.parsers.IOObjects import NumpyIO, PickleIO, WavIO, TextGridIO, TextIO


class SSLUnitParser(BaseDataParser):
    def __init__(self, root):
        super().__init__(root)

        self.dp_segment = Feature(
            "dp_segment", root, SFQueryParser(f"{self.root}/dp_segment"), PickleIO(), enable_cache=True)
        self.phoneme = Feature(
            "phoneme", root, SFQueryParser(f"{self.root}/phoneme"), TextIO(), enable_cache=True)
        self.dp_duration = Feature(
            "dp_duration", root, SFQueryParser(f"{self.root}/dp_duration"), NumpyIO(), enable_cache=True)
        self.dp_duration_avg_pitch = Feature(
            "dp_duration_avg_pitch", root, SFQueryParser(f"{self.root}/dp_duration_avg_pitch"), NumpyIO(), enable_cache=True)
        self.dp_duration_avg_energy = Feature(
            "dp_duration_avg_energy", root, SFQueryParser(f"{self.root}/dp_duration_avg_energy"), NumpyIO(), enable_cache=True)

    def _init_structure(self):
        os.makedirs(f"{self.root}/dp_segment", exist_ok=True)
        os.makedirs(f"{self.root}/phoneme", exist_ok=True)
        os.makedirs(f"{self.root}/dp_duration", exist_ok=True)
        os.makedirs(f"{self.root}/dp_duration_avg_pitch", exist_ok=True)
        os.makedirs(f"{self.root}/dp_duration_avg_energy", exist_ok=True)

    def get_feature(self, query: str) -> Feature:
        return getattr(self, query)


class DataParser(BaseDataParser):

    ssl_units: Dict[str, SSLUnitParser]

    def __init__(self, root):
        super().__init__(root)
        self.__init_ssl_units()

        self.wav_16000 = Feature(
            "wav_16000", root, SFQueryParser(f"{self.root}/wav_16000"), WavIO(sr=16000))
        self.wav_22050 = Feature(
            "wav_22050", root, SFQueryParser(f"{self.root}/wav_22050"), WavIO(sr=22050))
        self.mel = Feature(
            "mel", root, SFQueryParser(f"{self.root}/mel"), NumpyIO())
        self.pitch = Feature(
            "pitch", root, SFQueryParser(f"{self.root}/pitch"), NumpyIO(), enable_cache=True)
        self.interpolate_pitch = Feature(
            "interpolate_pitch", root, SFQueryParser(f"{self.root}/interpolate_pitch"), NumpyIO(), enable_cache=True)
        self.energy = Feature(
            "energy", root, SFQueryParser(f"{self.root}/energy"), NumpyIO(), enable_cache=True)
        self.mfa_duration_avg_pitch = Feature(
            "mfa_duration_avg_pitch", root, SFQueryParser(f"{self.root}/mfa_duration_avg_pitch"), NumpyIO(), enable_cache=True)
        self.unsup_duration_avg_pitch = Feature(
            "unsup_duration_avg_pitch", root, SFQueryParser(f"{self.root}/unsup_duration_avg_pitch"), NumpyIO(), enable_cache=True)
        self.mfa_duration_avg_energy = Feature(
            "mfa_duration_avg_energy", root, SFQueryParser(f"{self.root}/mfa_duration_avg_energy"), NumpyIO(), enable_cache=True)
        self.unsup_duration_avg_energy = Feature(
            "unsup_duration_avg_energy", root, SFQueryParser(f"{self.root}/unsup_duration_avg_energy"), NumpyIO(), enable_cache=True)
        self.wav_trim_22050 = Feature(
            "wav_trim_22050", root, SFQueryParser(f"{self.root}/wav_trim_22050"), NumpyIO())
        self.wav_trim_16000 = Feature(
            "wav_trim_16000", root, SFQueryParser(f"{self.root}/wav_trim_16000"), NumpyIO())
        self.unsup_segment = Feature(
            "unsup_segment", root, SFQueryParser(f"{self.root}/unsup_segment"), PickleIO(), enable_cache=True)
        self.mfa_segment = Feature(
            "mfa_segment", root, SFQueryParser(f"{self.root}/mfa_segment"), PickleIO(), enable_cache=True)
        self.textgrid = Feature(
            "TextGrid", root, NestSFQueryParser(f"{self.root}/TextGrid"), TextGridIO())
        self.phoneme = Feature(
            "phoneme", root, SFQueryParser(f"{self.root}/phoneme"), TextIO(), enable_cache=True)
        self.text = Feature(
            "text", root, SFQueryParser(f"{self.root}/text"), TextIO(), enable_cache=True)
        self.spk_ref_mel_slices = Feature(
            "spk_ref_mel_slices", root, SFQueryParser(f"{self.root}/spk_ref_mel_slices"), NumpyIO())
        self.mfa_duration = Feature(
            "mfa_duration", root, SFQueryParser(f"{self.root}/mfa_duration"), NumpyIO(), enable_cache=True)
        self.unsup_duration = Feature(
            "unsup_duration", root, SFQueryParser(f"{self.root}/unsup_duration"), NumpyIO(), enable_cache=True)
        self.mfa_ssl_duration = Feature(
            "mfa_ssl_duration", root, SFQueryParser(f"{self.root}/mfa_ssl_duration"), NumpyIO(), enable_cache=True)
        self.unsup_ssl_duration = Feature(
            "unsup_ssl_duration", root, SFQueryParser(f"{self.root}/unsup_ssl_duration"), NumpyIO(), enable_cache=True)

        self.stats_path = f"{self.root}/stats.json"
        self.speakers_path = f"{self.root}/speakers.json"

    def _init_structure(self):
        os.makedirs(f"{self.root}/wav_16000", exist_ok=True)
        os.makedirs(f"{self.root}/wav_22050", exist_ok=True)
        os.makedirs(f"{self.root}/mel", exist_ok=True)
        os.makedirs(f"{self.root}/pitch", exist_ok=True)
        os.makedirs(f"{self.root}/interpolate_pitch", exist_ok=True)
        os.makedirs(f"{self.root}/energy", exist_ok=True)
        os.makedirs(f"{self.root}/wav_trim_22050", exist_ok=True)
        os.makedirs(f"{self.root}/wav_trim_16000", exist_ok=True)
        os.makedirs(f"{self.root}/unsup_segment", exist_ok=True)
        os.makedirs(f"{self.root}/mfa_segment", exist_ok=True)
        os.makedirs(f"{self.root}/phoneme", exist_ok=True)
        os.makedirs(f"{self.root}/text", exist_ok=True)
        os.makedirs(f"{self.root}/spk_ref_mel_slices", exist_ok=True)
        os.makedirs(f"{self.root}/mfa_duration", exist_ok=True)
        os.makedirs(f"{self.root}/unsup_duration", exist_ok=True)
        os.makedirs(f"{self.root}/mfa_ssl_duration", exist_ok=True)
        os.makedirs(f"{self.root}/unsup_ssl_duration", exist_ok=True)
        os.makedirs(f"{self.root}/mfa_duration_avg_pitch", exist_ok=True)
        os.makedirs(f"{self.root}/unsup_duration_avg_pitch", exist_ok=True)
        os.makedirs(f"{self.root}/mfa_duration_avg_energy", exist_ok=True)
        os.makedirs(f"{self.root}/unsup_duration_avg_energy", exist_ok=True)
    
    def get_all_queries(self):
        with open(f"{self.root}/data_info.json", "r", encoding="utf-8") as f:
            data_infos = json.load(f)
        res = []
        for data_info in data_infos:
            query = {
                "spk": data_info["spk"],
                "basename": data_info["basename"],
            }
            res.append(query)
        return res
    
    def __init_ssl_units(self):
        self.ssl_units = {}
        os.makedirs(f"{self.root}/ssl_units", exist_ok=True)
        unit_names = os.listdir(f"{self.root}/ssl_units")
        for unit_name in unit_names:
            self.ssl_units[unit_name] = SSLUnitParser(f"{self.root}/ssl_units/{unit_name}")

    def create_ssl_unit_feature(self, unit_name):
        if unit_name not in self.ssl_units:
            self.ssl_units[unit_name] = SSLUnitParser(f"{self.root}/ssl_units/{unit_name}")

    # def get_ssl_unit_feature(self, query) -> Feature:
    #     """
    #     Args:
    #         query: Dictionary with format
    #             {
    #                 "unit_name": str,
    #                 "feat_name": str,
    #             }
    #     """
    #     unit_parser = self.ssl_units.get(query["unit_name"], None)
    #     if unit_parser is None:
    #         print(f"Can not get SSL unit ({query['unit_name']})!")
    #         raise KeyError
    #     res = getattr(unit_parser, query["feat_name"], None)
    #     if res is None:
    #         print(f"Can not get feat name ({query['feat_name']})!")
    #         raise KeyError
    #     return res

    def get_feature(self, query: str) -> Feature:
        if "/" not in query:
            return getattr(self, query)
        prefix, subquery = query.split("/", 1)
        if prefix == "ssl_units":
            unit_name, subquery = subquery.split("/", 1)
            return self.ssl_units[unit_name].get_feature(subquery)
        else:
            raise NotImplementedError
