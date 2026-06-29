from  rag.embedders.base import BaseEmbedder
from sentence_transformers import SentenceTransformer
import numpy as np


class SentenceTransformerEmbedder(BaseEmbedder):

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(
            model_name
        )
        self.model_name = model_name

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=False
        )
    
    @property
    def embedding_dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()
    
    @property
    def name(self) -> str:
        return self.model_name

    