from rag.loaders.factory import DocumentLoaderFactory
from rag.pipeline.pipeline import RAGPipeline
from pathlib import Path
import argparse
from rag.generators.openai import OpenAIGenerator
from rag.chunkers.fixedsize import FixedSizeChunker
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
        chunker=FixedSizeChunker(chunk_size=1000, overlap=50),
        embedder=SentenceTransformerEmbedder(),
        retriever=FAISSRetriever(),
        prompt_builder=SimplePromptBuilder(),
        generator=OpenAIGenerator(),
    )


    documents = [
        DocumentLoaderFactory.load(path)
        for path in paths
    ]

    pipeline.index(documents)
    print(pipeline.summary())
    
    answer = pipeline.ask(
        "Why are Transformer models important for both BERT and RAG systems?", k=20
    )
    print(answer)

    retrieved_results = {}
    relevant_results = {}

    evaluation_queries = load_queries(args.evaluation_queries)

    for query in evaluation_queries:
        retrieved_chunks = pipeline.search(
        query.question,
        k=5,
        )

        retrieved_results[query.id] = retrieved_chunks
        relevant_results[query.id] = query.relevant_documents

    metrics = evaluate_retrieval(
        retrieved_results,
        relevant_results,
        k=5,
        key=lambda chunk: chunk.document_id,
        relevant_key=lambda doc_id: doc_id,
    )

    print(metrics)
    

if __name__=="__main__":
    main()