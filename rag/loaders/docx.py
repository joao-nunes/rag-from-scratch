from rag.loaders.base import BaseLoader, Document
from pathlib import Path


class DOCXLoader(BaseLoader):
    def load(self, path: Path)->Document:
        raise NotImplementedError