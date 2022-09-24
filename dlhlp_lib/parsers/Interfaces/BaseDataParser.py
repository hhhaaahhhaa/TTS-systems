from typing import List
import abc

from .BaseIOObject import BaseIOObject
from .BaseQueryParser import BaseQueryParser


class BaseFeature(metaclass=abc.ABCMeta):
    """
    Feature is a template class.
    """
    def __init__(self, name: str, root: str, parser: BaseQueryParser, io: BaseIOObject, enable_cache=False, *args, **kwargs) -> None:
        raise NotImplementedError

    def read_all(self, refresh=False, *args, **kwargs) -> None:
        raise NotImplementedError

    def read_filename(self, query, raw=False, *args, **kwargs) -> str:
        raise NotImplementedError

    def read_filenames(self, query, raw=False, *args, **kwargs) -> List[str]:
        raise NotImplementedError
    
    def read_from_query(self, query, *args, **kwargs):
        raise NotImplementedError

    def read_from_filename(self, filename, *args, **kwargs):
        raise NotImplementedError
    
    def save(self, input, query, *args, **kwargs) -> None:
        raise NotImplementedError

    def filename2rawpath(self, filename: str, *args, **kwargs) -> str:
        raise NotImplementedError


class BaseDataParser(metaclass=abc.ABCMeta):
    def __init__(self, root, *args, **kwargs) -> None:
        self.root = root
        self._init_structure(*args, **kwargs)

    @abc.abstractmethod
    def _init_structure(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def get_feature(self, query: str) -> BaseFeature:
        raise NotImplementedError
