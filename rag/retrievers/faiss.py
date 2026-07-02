import faiss
import numpy as np

from rag.chunkers.base import Chunk
from rag.retrievers.base import BaseRetriever, RetrievalResult


class FAISSRetriever(BaseRetriever):

    def __init__(self) -> None:
        self.index = None
        self.chunks: list[Chunk] = []

    def build(
        self,
        embeddings: np.ndarray,
        chunks: list[Chunk],
    ) -> None:
        """
        Build a FAISS index from the chunk embeddings.
        """

        if len(embeddings) != len(chunks):
            raise ValueError(
                "The number of embeddings must match the number of chunks."
            )

        embeddings = embeddings.astype(np.float32)

        dimension = embeddings.shape[1]
        
        faiss.normalize_L2(embeddings)

        self.index = faiss.IndexFlatIP(dimension)

        self.index.add(embeddings)

        self.chunks = chunks

    def retrieve(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
    ) -> list[Chunk]:
        """
        Retrieve the k nearest chunks.
        """

        if self.index is None:
            raise RuntimeError(
                "The index has not been built. Call build() first."
            )

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32
        )

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(
            query_embedding,
            k
        )

        results = []

        for _, (idx, score) in enumerate(
            zip(indices[0], scores[0]),
            start=1,
        ):
            results.append(
                RetrievalResult(
                    chunk=self.chunks[idx],
                    score=float(score),
                )
            )

        return results