from rag.loaders.base import BaseLoader, Document
from pathlib import Path
from pypdf import PdfReader


class PDFLoader(BaseLoader):
    def load(self, path: Path)->Document:

        document_id = path.stem
        reader = PdfReader(path)
        text = ""

        for page in reader.pages:
            text += page.extract_text() or ""

        return Document(
            id=document_id, 
            text=text,
            metadata={
                "path": str(path),
                "filename": path.name,
                "file_type": path.suffix.lstrip("."),
                "num_pages": len(reader.pages),
                }
            )