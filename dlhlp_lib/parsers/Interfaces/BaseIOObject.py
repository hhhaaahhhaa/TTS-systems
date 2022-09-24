from typing import List
import abc


class BaseIOObject(metaclass=abc.ABCMeta):
    def __init__(self):
        self.extension = ""

    @abc.abstractmethod 
    def readfile(self, path):
        raise NotImplementedError

    @abc.abstractmethod
    def savefile(self, input, path):
        raise NotImplementedError
