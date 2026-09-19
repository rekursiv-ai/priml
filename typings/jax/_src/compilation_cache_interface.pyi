import abc

from jax._src import util as util

class CacheInterface(util.StrictABC, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get(self, key: str): ...
    @abc.abstractmethod
    def put(self, key: str, value: bytes): ...
