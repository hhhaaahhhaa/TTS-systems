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
        return [os.path.basename(x).split(".")[0] for x in glob.glob(f"{self.root}/*{extension}")]

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
        return [os.path.basename(x).split(".")[0] for x in glob.glob(f"{self.root}/*/*{extension}")]

    def get_cache(self) -> str:
        return f"{self.root}/cache.pickle"
