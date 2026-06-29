from abc import ABC, abstractmethod

class BaseEmbedder(ABC):

    @abstractmethod
    def embed(self, texts: list[str]):
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass