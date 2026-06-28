from abc import ABC, abstractmethod
import numpy as np
from rag.chunkers.base import Chunk

class BaseRetriever(ABC):

    @abstractmethod
    def build(
        self,
        embeddings: np.ndarray,
        chunks: list[Chunk],
    ) -> None:
        """
        Build the retrieval index from the chunk embeddings.
        """
        pass

    @abstractmethod
    def retrieve(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
    ) -> list[Chunk]:
        """
        Retrieve the k most similar chunks for a query embedding.
        """
        pass