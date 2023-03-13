import os
import json
from typing import Dict, List

from dlhlp_lib.parsers.Interfaces import BaseDataParser
from dlhlp_lib.parsers.Feature import Feature
from dlhlp_lib.parsers.QueryParsers import SFQueryParser, NestSFQueryParser
from dlhlp_lib.parsers.IOObjects import NumpyIO, PickleIO, WavIO, TextGridIO, TextIO, JSONIO


class UnitParser(BaseDataParser):
    def __init__(self, root):
        super().__init__(root)

        self.segment = Feature(
            NestSFQueryParser(f"{self.root}/segment"), JSONIO(), enable_cache=True)
        self.phoneme = Feature(
            NestSFQueryParser(f"{self.root}/phoneme"), TextIO(), enable_cache=True)
        self.duration = Feature(
            NestSFQueryParser(f"{self.root}/duration"), NumpyIO(), enable_cache=True)
        self.duration_avg_pitch = Feature(
            NestSFQueryParser(f"{self.root}/duration_avg_pitch"), NumpyIO(), enable_cache=True)
        self.duration_avg_energy = Feature(
            NestSFQueryParser(f"{self.root}/duration_avg_energy"), NumpyIO(), enable_cache=True)
    
    def _init_structure(self):
        os.makedirs(self.root, exist_ok=True)

    def get_feature(self, query: str) -> Feature:
        return getattr(self, query)


class DataParser(BaseDataParser):

    units: Dict[str, UnitParser]

    def __init__(self, root):
        super().__init__(root)
        self.__init_units()
        self.create_unit_feature("mfa")

        self.wav_16000 = Feature(
            SFQueryParser(f"{self.root}/wav_16000"), WavIO(sr=16000))
        self.wav_22050 = Feature(
            SFQueryParser(f"{self.root}/wav_22050"), WavIO(sr=22050))
        self.mel = Feature(
            NestSFQueryParser(f"{self.root}/mel"), NumpyIO())
        self.pitch = Feature(
            NestSFQueryParser(f"{self.root}/pitch"), NumpyIO(), enable_cache=True)
        self.interpolate_pitch = Feature(
            NestSFQueryParser(f"{self.root}/interpolate_pitch"), NumpyIO(), enable_cache=True)
        self.energy = Feature(
            NestSFQueryParser(f"{self.root}/energy"), NumpyIO(), enable_cache=True)
        
        self.wav_trim_22050 = Feature(
            NestSFQueryParser(f"{self.root}/wav_trim_22050"), NumpyIO())
        self.wav_trim_16000 = Feature(
            NestSFQueryParser(f"{self.root}/wav_trim_16000"), NumpyIO())
        self.textgrid = Feature(
            NestSFQueryParser(f"{self.root}/TextGrid"), TextGridIO())
        self.text = Feature(
            SFQueryParser(f"{self.root}/text"), TextIO(), enable_cache=True)
        self.spk_ref_mel_slices = Feature(
            NestSFQueryParser(f"{self.root}/spk_ref_mel_slices"), NumpyIO())
        
        self.stats_path = f"{self.root}/stats.json"
        self.speakers_path = f"{self.root}/speakers.json"
        self.metadata_path = f"{self.root}/data_info.json"

    def _init_structure(self):
        os.makedirs(f"{self.root}/wav_22050", exist_ok=True)
        os.makedirs(f"{self.root}/text", exist_ok=True)
    
    def get_all_queries(self):
        with open(f"{self.root}/data_info.json", "r", encoding="utf-8") as f:
            data_infos = json.load(f)
        return data_infos
    
    def __init_units(self):
        self.units = {}
        os.makedirs(f"{self.root}/units", exist_ok=True)
        unit_names = os.listdir(f"{self.root}/units")
        for unit_name in unit_names:
            self.units[unit_name] = UnitParser(f"{self.root}/units/{unit_name}")

    def create_unit_feature(self, unit_name):
        if unit_name not in self.units:
            self.units[unit_name] = UnitParser(f"{self.root}/units/{unit_name}")

    def get_feature(self, query: str) -> Feature:
        if "/" not in query:
            return getattr(self, query)
        prefix, subquery = query.split("/", 1)
        if prefix == "units":
            unit_name, subquery = subquery.split("/", 1)
            return self.units[unit_name].get_feature(subquery)
        else:
            raise NotImplementedError
    
    def get_all_speakers(self) -> List[str]:
        with open(self.speakers_path, 'r', encoding='utf-8') as f:
            speakers = json.load(f)
        return speakers
