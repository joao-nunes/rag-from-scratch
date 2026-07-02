from sentence_transformers import CrossEncoder

from rag.chunkers.base import Chunk
from rag.rerankers.base import BaseReranker


from sentence_transformers import CrossEncoder

from rag.retrievers.base import RetrievalResult
from rag.rerankers.base import BaseReranker


class CrossEncoderReranker(BaseReranker):

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:

        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    @property
    def name(self) -> str:
        return self.model_name

    def rerank(
        self,
        question: str,
        results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """
        Rerank retrieval results using a cross-encoder.
        """

        pairs = [
            (question, result.chunk.text)
            for result in results
        ]

        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(results, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        reranked_results = []

        for _, (result, score) in enumerate(
            ranked,
            start=1,
        ):
            reranked_results.append(
                RetrievalResult(
                    chunk=result.chunk,
                    score=float(score),
                )
            )

        return reranked_results