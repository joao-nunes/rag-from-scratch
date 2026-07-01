from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

T = TypeVar("T")
KeyFn = Callable[[T], Any]


@dataclass(frozen=True)
class RetrievalMetrics:
    precision: float
    recall: float
    mrr: float
    map: float


def _identity(value: T) -> T:
    return value


def _as_key_set(
    items: Iterable[T],
    key: KeyFn[T] | None,
) -> set[Any]:
    key_fn = key or _identity
    return {key_fn(item) for item in items}


def _top_k(
    items: Sequence[T],
    k: int,
) -> Sequence[T]:
    if k <= 0:
        raise ValueError("k must be greater than 0.")
    return items[:k]


def precision_at_k(
    retrieved: Sequence[T],
    relevant: Iterable[Any],
    k: int,
    key: KeyFn[T] | None = None,
    relevant_key: KeyFn[Any] | None = None,
) -> float:
    """
    Compute precision@k for one ranked retrieval result.

    Precision@k is the fraction of the first k retrieved items that are
    relevant. The denominator is always k, even if fewer than k items were
    retrieved.
    """

    top_results = _top_k(retrieved, k)
    relevant_keys = _as_key_set(relevant, relevant_key or key)
    key_fn = key or _identity

    retrieved_keys = []
    seen = set()

    for item in top_results:
        key_value = key_fn(item)
        if key_value not in seen:
            seen.add(key_value)
            retrieved_keys.append(key_value)

    hits = sum(
        key in relevant_keys
        for key in retrieved_keys
    )

    return hits / k


def recall_at_k(
    retrieved: Sequence[T],
    relevant: Iterable[Any],
    k: int,
    key: KeyFn[T] | None = None,
    relevant_key: KeyFn[Any] | None = None,
) -> float:
    """
    Compute recall@k for one ranked retrieval result.

    Recall@k is the fraction of all relevant items that appear in the first k
    retrieved results.
    """

    relevant_keys = _as_key_set(relevant, relevant_key or key)
    if not relevant_keys:
        return 0.0

    top_results = _top_k(retrieved, k)
    key_fn = key or _identity
    
    retrieved_keys = {
    key_fn(item)
    for item in top_results}

    return len(retrieved_keys & relevant_keys) / len(relevant_keys)


def reciprocal_rank(
    retrieved: Sequence[T],
    relevant: Iterable[Any],
    k: int | None = None,
    key: KeyFn[T] | None = None,
    relevant_key: KeyFn[Any] | None = None,
) -> float:
    """
    Compute reciprocal rank for one ranked retrieval result.

    The score is 1 / rank of the first relevant item, or 0 when no relevant
    item is found. Pass k to limit the ranking depth.

    Retrieved items are deduplicated according to ``key`` before computing the
    rank so that multiple retrieved chunks from the same document count only
    once.
    """

    if k is not None:
        ranked_results = _top_k(retrieved, k)
    else:
        ranked_results = retrieved

    key_fn = key or _identity
    relevant_keys = _as_key_set(
        relevant,
        relevant_key or _identity,
    )

    seen: set[Any] = set()

    rank = 1
    for item in ranked_results:
        item_key = key_fn(item)

        if item_key in seen:
            continue

        seen.add(item_key)

        if item_key in relevant_keys:
            return 1 / rank

        rank += 1

    return 0.0


def mean_reciprocal_rank(
    retrieved_results: Sequence[Sequence[T]],
    relevant_results: Sequence[Iterable[Any]],
    k: int | None = None,
    key: KeyFn[T] | None = None,
    relevant_key: KeyFn[Any] | None = None,
) -> float:
    """
    Compute mean reciprocal rank over many queries.
    """

    if len(retrieved_results) != len(relevant_results):
        raise ValueError(
            "retrieved_results and relevant_results must have the same length."
        )

    if not retrieved_results:
        return 0.0

    return sum(
        reciprocal_rank(
            retrieved,
            relevant,
            k=k,
            key=key,
            relevant_key=relevant_key,
        )
        for retrieved, relevant in zip(
            retrieved_results,
            relevant_results,
            strict=True,
        )
    ) / len(retrieved_results)


def average_precision(
    retrieved: Sequence[T],
    relevant: Iterable[Any],
    k: int | None = None,
    key: KeyFn[T] | None = None,
    relevant_key: KeyFn[Any] | None = None,
) -> float:
    """
    Compute Average Precision (AP) for one ranked retrieval result.

    AP is the mean of the precision values computed at the ranks where a
    relevant document is retrieved.
    """

    if k is not None:
        ranked_results = _top_k(retrieved, k)
    else:
        ranked_results = retrieved

    key_fn = key or _identity
    relevant_key_fn = relevant_key or _identity

    relevant_keys = _as_key_set(
        relevant,
        relevant_key_fn,
    )

    if not relevant_keys:
        return 0.0

    seen: set[Any] = set()

    hits = 0
    precision_sum = 0.0
    rank = 0

    for item in ranked_results:
        item_key = key_fn(item)

        if item_key in seen:
            continue

        seen.add(item_key)
        rank += 1

        if item_key in relevant_keys:
            hits += 1
            precision_sum += hits / rank

    return precision_sum / len(relevant_keys)


def mean_average_precision(
    retrieved_results: Sequence[Sequence[T]],
    relevant_results: Sequence[Iterable[Any]],
    k: int | None = None,
    key: KeyFn[T] | None = None,
    relevant_key: KeyFn[Any] | None = None,
) -> float:
    """
    Compute Mean Average Precision (MAP).
    """

    if len(retrieved_results) != len(relevant_results):
        raise ValueError(
            "retrieved_results and relevant_results must have the same length."
        )

    if not retrieved_results:
        return 0.0

    return sum(
        average_precision(
            retrieved,
            relevant,
            k=k,
            key=key,
            relevant_key=relevant_key,
        )
        for retrieved, relevant in zip(
            retrieved_results,
            relevant_results,
            strict=True,
        )
    ) / len(retrieved_results)


def evaluate_retrieval(
    retrieved_results: Mapping[str, Sequence[T]],
    relevant_results: Mapping[str, Iterable[Any]],
    k: int,
    key: KeyFn[T] | None = None,
    relevant_key: KeyFn[Any] | None = None,
) -> RetrievalMetrics:
    """
    Average precision@k, recall@k, and MRR across multiple queries.

    Both mappings are keyed by query id. Every retrieved query id must have a
    matching set of relevant items.
    """

    if k <= 0:
        raise ValueError("k must be greater than 0.")

    missing_query_ids = set(retrieved_results) - set(relevant_results)
    if missing_query_ids:
        missing = ", ".join(sorted(missing_query_ids))
        raise ValueError(
            f"Missing relevant results for query id(s): {missing}"
        )

    if not retrieved_results:
        return RetrievalMetrics(
            precision=0.0,
            recall=0.0,
            mrr=0.0,
        )

    precision_sum = 0.0
    recall_sum = 0.0
    reciprocal_rank_sum = 0.0
    average_precision_sum = 0.0

    for query_id, retrieved in retrieved_results.items():
        relevant = relevant_results[query_id]
        precision_sum += precision_at_k(
            retrieved,
            relevant,
            k,
            key=key,
            relevant_key=relevant_key,
        )
        recall_sum += recall_at_k(
            retrieved,
            relevant,
            k,
            key=key,
            relevant_key=relevant_key,
        )
        reciprocal_rank_sum += reciprocal_rank(
            retrieved,
            relevant,
            k=None,
            key=key,
            relevant_key=relevant_key,
        )
        average_precision_sum += average_precision(
            retrieved,
            relevant,
            k,
            key=key,
            relevant_key=relevant_key,
        )


    query_count = len(retrieved_results)
    return RetrievalMetrics(
        precision=precision_sum / query_count,
        recall=recall_sum / query_count,
        mrr=reciprocal_rank_sum / query_count,
        map=average_precision_sum / query_count,
    )
