import os
import glob
from typing import List

from .Interfaces import BaseQueryParser


class SFQueryParser(BaseQueryParser):
    """
    [root]/[spk]-[basename][extension]
    """
    def __init__(self, root):
        super().__init__(root)
    
    def get(self, query) -> List[str]:
        return [f"{query['spk']}-{query['basename']}"]

    def get_all(self, extension) -> List[str]:
        return [os.path.splitext(os.path.basename(x))[0] for x in glob.glob(f"{self.root}/*{extension}")]

    def get_cache(self) -> str:
        return f"{self.root}/cache.pickle"


class NestSFQueryParser(BaseQueryParser):
    """
    [root]/[spk]/[basename][extension]
    """
    def __init__(self, root):
        super().__init__(root)
    
    def get(self, query) -> List[str]:
        return [f"{query['spk']}/{query['basename']}"]

    def get_all(self, extension) -> List[str]:
        res = []
        for d in os.listdir(self.root):
            if os.path.isdir(f"{self.root}/{d}"):
                res.extend([f"{d}/{os.path.splitext(os.path.basename(x))[0]}" 
                    for x in glob.glob(f"{self.root}/{d}/*{extension}")])
        return res

    def get_cache(self) -> str:
        return f"{self.root}/cache.pickle"
