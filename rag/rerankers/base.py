from abc import ABC, abstractmethod
from rag.chunkers.base import Chunk


class BaseReranker(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the reranker."""
        pass

    @abstractmethod
    def rerank(
        self,
        question: str,
        chunks: list[Chunk],
    ) -> list[Chunk]:
        """
        Rerank a list of retrieved chunks according to their relevance
        to the question.
        """
        pass