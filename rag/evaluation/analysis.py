from dataclasses import dataclass
from rag.retrievers.base import RetrievalResult


@dataclass
class RetrievalAnalysis:
    benchmark: str
    query_id: int
    claim: str
    relevant_document_ids: set[int]
    retrieved: list[RetrievalResult]
    reranked: list[RetrievalResult] | None = None