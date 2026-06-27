from loaders.base import BaseLoader, Document
from pathlib import Path


class HTMLLoader(BaseLoader):
    def load(self, path: Path)->Document:
        raise NotImplementedError