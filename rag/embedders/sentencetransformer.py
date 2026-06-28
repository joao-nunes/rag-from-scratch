from  rag.embedders.base import BaseEmbedder
from sentence_transformers import SentenceTransformer
import numpy as np


class SentenceTransformerEmbedder(BaseEmbedder):

    def __init__(self):
        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=False
        )
    
    @property
    def embedding_dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()