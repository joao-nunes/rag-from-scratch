from rag.chunkers.base import Chunk, BaseChunker
from rag.embedders.base import BaseEmbedder
from rag.generators.base import BaseGenerator, BasePromptBuilder
from rag.loaders.base import Document
from rag.retrievers.base import BaseRetriever
from rag.rerankers.base import BaseReranker
import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class IndexStats:
    num_documents: int
    num_chunks: int
    embedding_dimension: int


class RAGPipeline:

    def __init__(
        self,
        chunker: BaseChunker,
        embedder: BaseEmbedder,
        retriever: BaseRetriever,
        prompt_builder: BasePromptBuilder,
        generator: BaseGenerator,
        reranker: BaseReranker | None = None,
    ) -> None:

        self.chunker = chunker
        self.embedder = embedder
        self.retriever = retriever
        self.reranker = reranker
        self.prompt_builder = prompt_builder
        self.generator = generator
        self._num_chunks = 0
        self._num_documents = 0
        self._embedding_dim = 0

    def index(self, documents: list[Document]) -> None:
        """
        Index a collection of documents.
        """

        chunks = []

        for document in documents:
            chunks.extend(
                self.chunker.chunk(document)
            )
        
        self._num_chunks = len(chunks)
        self._num_documents = len(documents)

        texts = [chunk.text for chunk in chunks]

        embeddings = self.embedder.embed(texts)
        self._embedding_dim = self.embedder.embedding_dimension

        self.retriever.build(
            embeddings=embeddings,
            chunks=chunks,
        )

    def ask(
        self,
        question: str,
        k: int = 50,
        final_k: int = 5
    ) -> str:
        """
        Answer a question using Retrieval-Augmented Generation.
        """

        retrieved_chunks = self.search(question, k=k)

        if self.reranker is not None:
            retrieved_chunks = self.reranker.rerank(
                question,
                retrieved_chunks,
            )

        retrieved_chunks = retrieved_chunks[:final_k]

        prompt = self.prompt_builder.build(
            question=question,
            results=retrieved_chunks,
        )

        return self.generator.generate(prompt)
    

    def summary(self) -> str:
        """
        Return a summary of the RAG pipeline configuration.
        """

        lines = [
            "=" * 60,
            "RAG Pipeline Summary",
            "=" * 60,
            "",
            "Corpus",
            f"  Documents           : {self._num_documents}",
            f"  Chunks              : {self._num_chunks}",
            "",
            "Chunking",
            f"  Strategy            : {type(self.chunker).__name__}",
            "",
            "Embedding",
            f"  Model               : {self.embedder.name}",
            f"  Dimension           : {self.embedder.embedding_dimension}",
            "",
            "Retrieval",
            f"  Backend             : {type(self.retriever).__name__}",
            "",
            "Generation",
            f"  Model               : {self.generator.model}",
            "",
            "=" * 60,
        ]

        return "\n".join(lines)
    
    def search(
        self,
        question: str,
        k: int = 50,
    ) -> list[Chunk]:
        """
        Retrieve the k most relevant chunks for a question.
        """

        query_embedding = self.embedder.embed([question])

        return self.retriever.retrieve(
            query_embedding=query_embedding,
            k=k,
        )
