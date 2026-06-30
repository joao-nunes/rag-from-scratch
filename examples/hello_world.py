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


def parse():

    parser = argparse.ArgumentParser()
    parser.add_argument(
    "--path",
    type=str,
    default="./examples/data",
    help="Directory containing the documents."
    )
    parser.set_defaults()
    return parser.parse_args()




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
        "What is quality control in computational pathology?", k=20
    )
    print(answer)
    

if __name__=="__main__":
    main()