from pdf import PDFLoader
from docx import DOCXLoader
from html import HTMLLoader
from pathlib import Path


class DocumentLoaderFactory:

    _registry = {
        ".pdf": PDFLoader,
        ".docx": DOCXLoader,
        ".html": HTMLLoader,
    }

    @classmethod
    def load_document(self, path: Path):
        extension = path.suffix.lower()
        loader_cls = self._registry.get(extension)

        if loader_cls is None:
            raise ValueError(f"Unsupported file type: '{extension}'")

        loader = loader_cls()
        return loader.load(path)