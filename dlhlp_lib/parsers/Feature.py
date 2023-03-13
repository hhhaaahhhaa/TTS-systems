import os
from typing import List
import pickle
from tqdm import tqdm

from .Interfaces import BaseFeature, BaseIOObject, BaseQueryParser


class Feature(BaseFeature):
    """
    Template class for single feature.
    """
    def __init__(self, parser: BaseQueryParser, io: BaseIOObject, enable_cache=False):
        self.query_parser = parser
        self.io = io
        self._data = None
        self._enable_cache = enable_cache

    def read_all(self, refresh=False):
        if self._data is not None:  # cache already loaded
            return
        if not self._enable_cache:
            self.log("Cache not supported...")
            raise NotImplementedError
        cache_path = self.query_parser.get_cache()
        if not os.path.isfile(cache_path) or refresh:
            self.log("Generating cache...")
            self.build_cache()
        
        self.log("Loading cache...")
        with open(cache_path, 'rb') as f:
            self._data = pickle.load(f)
    
    def build_cache(self):
        cache_path = self.query_parser.get_cache()
        data = {}
        filenames = self.query_parser.get_all(extension=self.io.extension)
        for filename in tqdm(filenames):
            data[filename] = self.read_from_filename(filename)
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)

    def clear_cache(self):
        self._data = None

    def read_filename(self, query, raw=False) -> str:
        filenames = self.read_filenames(query, raw=raw)
        assert len(filenames) == 1
        return filenames[0]

    def read_filenames(self, query, raw=False) -> List[str]:
        filenames = self.query_parser.get(query)
        if raw:
            filenames = [self.filename2rawpath(f) for f in filenames]
        return filenames
    
    def read_from_query(self, query):
        filename = self.read_filename(query)
        return self.read_from_filename(filename)

    def read_from_filename(self, filename):
        if self._data is not None:
            return self._data[filename]
        return self.io.readfile(self.filename2rawpath(filename))
    
    def save(self, input, query):
        path = self.read_filename(query, raw=True)
        self.io.savefile(input, path)

    def filename2rawpath(self, filename) -> str:
        return f"{self.query_parser.root}/{filename}{self.io.extension}"
    
    def log(self, msg):
        print(f"[Feature ({self.query_parser.root})]: ", msg)
