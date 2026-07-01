from rag.chunkers.base import BaseChunker, Chunk
from rag.loaders.base import Document

class IdentityChunker(BaseChunker):

    def chunk(
        self,
        document: Document,
    ) -> list[Chunk]:

        return [
            Chunk(
                id=f"chunk_0{document.id}",
                text=document.text,
                document_id=document.id,
                metadata={},
            )
        ]