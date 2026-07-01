from dataclasses import dataclass

from rag.evaluation import (
    evaluate_retrieval,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


@dataclass(frozen=True)
class Result:
    chunk_id: str
    document_id: str


def test_precision_recall_and_reciprocal_rank_at_k() -> None:
    retrieved = ["doc-a", "doc-b", "doc-c", "doc-d"]
    relevant = ["doc-b", "doc-d"]

    assert precision_at_k(retrieved, relevant, k=3) == 1 / 3
    assert recall_at_k(retrieved, relevant, k=3) == 1 / 2
    assert reciprocal_rank(retrieved, relevant, k=3) == 1 / 2


def test_mean_reciprocal_rank() -> None:
    retrieved_results = [
        ["a", "b"],
        ["c", "d"],
        ["e", "f"],
    ]
    relevant_results = [
        ["b"],
        ["c"],
        ["x"],
    ]

    assert mean_reciprocal_rank(
        retrieved_results,
        relevant_results,
    ) == (1 / 2 + 1 + 0) / 3


def test_evaluate_retrieval_averages_query_metrics() -> None:
    scores = evaluate_retrieval(
        retrieved_results={
            "q1": ["a", "b", "c"],
            "q2": ["d", "e", "f"],
        },
        relevant_results={
            "q1": ["b", "c"],
            "q2": ["x"],
        },
        k=2,
    )

    assert scores.precision == (1 / 2 + 0) / 2
    assert scores.recall == (1 / 2 + 0) / 2
    assert scores.mrr == (1 / 2 + 0) / 2


def test_key_function_supports_document_level_evaluation() -> None:
    retrieved = [
        Result(chunk_id="chunk-1", document_id="rag"),
        Result(chunk_id="chunk-2", document_id="bert"),
        Result(chunk_id="chunk-3", document_id="rag"),
    ]
    relevant = [
        Result(chunk_id="anything", document_id="rag"),
    ]

    assert precision_at_k(
        retrieved,
        relevant,
        k=2,
        key=lambda result: result.document_id,
    ) == 1 / 2
    assert recall_at_k(
        retrieved,
        relevant,
        k=2,
        key=lambda result: result.document_id,
    ) == 1


def test_relevant_key_supports_plain_ground_truth_ids() -> None:
    retrieved = [
        Result(chunk_id="chunk-1", document_id="rag"),
        Result(chunk_id="chunk-2", document_id="bert"),
    ]

    assert reciprocal_rank(
        retrieved,
        relevant=["bert"],
        k=2,
        key=lambda result: result.document_id,
        relevant_key=lambda document_id: document_id,
    ) == 1 / 2
