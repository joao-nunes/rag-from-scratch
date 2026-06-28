from rag.chunkers.base import BaseChunker
from rag.embedders.base import BaseEmbedder
from rag.generators.base import BaseGenerator, BasePromptBuilder
from rag.loaders.base import Document
from rag.retrievers.base import BaseRetriever


class RAGPipeline:

    def __init__(
        self,
        chunker: BaseChunker,
        embedder: BaseEmbedder,
        retriever: BaseRetriever,
        prompt_builder: BasePromptBuilder,
        generator: BaseGenerator,
    ) -> None:

        self.chunker = chunker
        self.embedder = embedder
        self.retriever = retriever
        self.prompt_builder = prompt_builder
        self.generator = generator

    def index(self, documents: list[Document]) -> None:
        """
        Index a collection of documents.
        """

        chunks = []

        for document in documents:
            chunks.extend(
                self.chunker.chunk(document)
            )

        texts = [chunk.text for chunk in chunks]

        embeddings = self.embedder.embed(texts)

        self.retriever.build(
            embeddings=embeddings,
            chunks=chunks,
        )

    def ask(
        self,
        question: str,
        k: int = 5,
    ) -> str:
        """
        Answer a question using Retrieval-Augmented Generation.
        """

        query_embedding = self.embedder.embed([question])

        retrieved_chunks = self.retriever.retrieve(
            query_embedding=query_embedding,
            k=k,
        )

        prompt = self.prompt_builder.build(
            question=question,
            chunks=retrieved_chunks,
        )

        return self.generator.generate(prompt)