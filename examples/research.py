from beir import util
from beir.datasets.data_loader import GenericDataLoader
from rag.loaders.base import Document
from rag.pipeline.pipeline import RAGPipeline
from pathlib import Path
import argparse
from rag.generators.openai import OpenAIGenerator
from rag.chunkers.identity import IdentityChunker
from rag.embedders.sentencetransformer import SentenceTransformerEmbedder
from rag.retrievers.faiss import FAISSRetriever
from rag.rerankers.cross_encoder import CrossEncoderReranker

from rag.generators.prompts import SimplePromptBuilder
from dotenv import load_dotenv
import json
from rag.evaluation.metrics import RetrievalExperiment
from rag.evaluation.retrieval import evaluate_retrieval
from dataclasses import dataclass, asdict
from datetime import datetime
import csv
from time import perf_counter


def parse():

    parser = argparse.ArgumentParser()
    parser.add_argument(
    "--path",
    type=str,
    default="./benchmarks/research/corpus",
    help="Directory containing the documents."
    )
    parser.set_defaults()

    parser.add_argument(
    "--evaluation_queries",
    type=str,
    default="./benchmarks/research/queries.json",
    help="Path to file containing the evaluation queries."
    )
    parser.set_defaults()

    return parser.parse_args()


@dataclass(frozen=True)
class Query:
    id: str
    category: str
    difficulty: str
    question: str
    relevant_documents: list[str]
    requires_multiple_documents: bool
    key_facts: dict
    answer: str


def load_queries(path: Path) -> list[Query]:
    with open(path) as f:
        data = json.load(f)

    return [
        Query(**item)
        for item in data
    ]

def main():

    args = parse()
    root = Path(args.path)
    paths = [
    path
    for path in root.iterdir()
    if path.is_file() and not path.name.startswith(".")
]

    load_dotenv()

    pipeline = RAGPipeline(
        chunker=IdentityChunker(),
        embedder=SentenceTransformerEmbedder(model_name="BAAI/bge-base-en-v1.5"),
        retriever=FAISSRetriever(),
        reranker=CrossEncoderReranker(),
        prompt_builder=SimplePromptBuilder(),
        generator=OpenAIGenerator(),
    )


    dataset = "scifact"

    url = (
        "https://public.ukp.informatik.tu-darmstadt.de/"
        "thakur/BEIR/datasets/{}.zip".format(dataset)
    )

    data_path = util.download_and_unzip(url, "benchmarks")

    corpus, queries, qrels = GenericDataLoader(
        data_folder=data_path
    ).load(split="test")
   
    documents = [
    Document(
        id=doc_id,
        title=data["title"],
        text=data["title"] + "\n\n" + data["text"],
        metadata={},
    )
    for doc_id, data in corpus.items()
]

    start = perf_counter()
    pipeline.index(documents)
    index_time = perf_counter() - start
    print(pipeline.summary())
    
    question = "0-dimensional biomaterials lack inductive properties. Determine whether the claim is supported, contradicted, or not enough information based on the retrieved evidence."
   
    answer = pipeline.ask(
       question, k=50
    )
    print(answer)

    retrieved_chunks = pipeline.search(question, k=50)
    retrieved_chunks = pipeline.reranker.rerank(question, retrieved_chunks)
    retrieved_chunks = retrieved_chunks[:5]
    print(f"\nQuestion: {question}\n")

    for i, result in enumerate(retrieved_chunks, start=1):
        print(f"Rank {i}")
        print(f"Document: {result.chunk.document_id}")
        print(result.chunk.text[:300])
        print("-" * 80)

    retrieved_results = {}
    relevant_results = {}
    query_times = []
    for query_id, question in queries.items():
        start = perf_counter()
        retrieved = pipeline.search(
            question,
            k=50,
        )
        query_times.append(
        perf_counter() - start
        )

        if pipeline.reranker is not None:
            retrieved = pipeline.reranker.rerank(
                question,
                retrieved,
            )

        retrieved_results[query_id] = retrieved

        relevant_results[query_id] = list(
            qrels[query_id].keys()
        )

    query_time_ms = (
    sum(query_times)
    / len(query_times)
) * 1000

    for k in [1, 3, 5, 10, 20]:
        metrics = evaluate_retrieval(
            retrieved_results,
            relevant_results,
            k=k,
            key=lambda result: result.chunk.document_id,
            relevant_key= lambda doc_id: doc_id,
        )
        print(k, metrics)
        experiment_id=f"exp_{datetime.now():%Y%m%d_%H%M%S}"
        experiment = RetrievalExperiment(
                experiment_id=experiment_id,
                date=datetime.now().isoformat(timespec="seconds"),
                dataset="SciFact",
                embedding_model=pipeline.embedder.name,
                embedding_dimension=pipeline.embedder.embedding_dimension,
                chunker=type(pipeline.chunker).__name__,
                chunk_size=getattr(pipeline.chunker, "chunk_size", None),
                overlap=getattr(pipeline.chunker, "overlap", None),
                retriever=type(pipeline.retriever).__name__,
                similarity_metric="L2",
                k=k,
                precision=metrics.precision,
                recall=metrics.recall,
                mrr=metrics.mrr,
                map=metrics.map,
                index_time_s=index_time,
                query_time_ms=query_time_ms,
            )


        with open("./examples/results.csv", "a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=RetrievalExperiment.__dataclass_fields__.keys(),
            )

            if f.tell() == 0:
                writer.writeheader()

            writer.writerow(asdict(experiment))

if __name__=="__main__":
    main()