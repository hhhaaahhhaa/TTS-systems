from typing import List
import abc


class BaseQueryParser(metaclass=abc.ABCMeta):
    def __init__(self, root, *args, **kwargs):
        self.root = root
    
    @abc.abstractmethod
    def get(self, query) -> List[str]:
        raise NotImplementedError

    @abc.abstractmethod
    def get_all(self, extension: str) -> List[str]:
        raise NotImplementedError

    @abc.abstractmethod
    def get_cache(self) -> str:
        raise NotImplementedError
