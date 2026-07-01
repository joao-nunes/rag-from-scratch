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
from rag.generators.prompts import SimplePromptBuilder
from dotenv import load_dotenv
import json
from rag.evaluation.retrieval import evaluate_retrieval
from dataclasses import dataclass


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
        embedder=SentenceTransformerEmbedder(),
        retriever=FAISSRetriever(),
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

    pipeline.index(documents)
    print(pipeline.summary())
    
    question = "0-dimensional biomaterials lack inductive properties. Determine whether the claim is supported, contradicted, or not enough information based on the retrieved evidence."
   
    answer = pipeline.ask(
       question, k=10
    )
    print(answer)

    retrieved_chunks = pipeline.search(question, k=10)

    print(f"\nQuestion: {question}\n")

    for i, chunk in enumerate(retrieved_chunks, start=1):
        print(f"Rank {i}")
        print(f"Document: {chunk.document_id}")
        print(chunk.text[:300])
        print("-" * 80)

    retrieved_results = {}
    relevant_results = {}

    for query_id, question in queries.items():

        retrieved = pipeline.search(
            question,
            k=10,
        )

        retrieved_results[query_id] = retrieved

        relevant_results[query_id] = list(
            qrels[query_id].keys()
        )
        
    metrics = evaluate_retrieval(
        retrieved_results,
        relevant_results,
        k=10,
        key=lambda chunk: chunk.document_id,
        relevant_key= lambda doc_id: doc_id,
    )
    
    for k in [1, 3, 5, 10, 20]:
        metrics = evaluate_retrieval(
            retrieved_results,
            relevant_results,
            k=k,
            key=lambda chunk: chunk.document_id,
            relevant_key= lambda doc_id: doc_id,
        )
        print(k, metrics)

if __name__=="__main__":
    main()