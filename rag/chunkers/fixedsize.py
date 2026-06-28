from rag.chunkers.base import BaseChunker, Chunk
from rag.loaders.base import Document


class FixedSizeChunker(BaseChunker):

        def __init__(
            self,
            chunk_size: int = 500,
            overlap: int = 100
                ):
            
            if overlap >= chunk_size:
                raise ValueError(
                    "overlap must be smaller than chunk_size"
                )
            
            self.chunk_size = chunk_size
            self.overlap = overlap

            

        def chunk(self, document: Document)-> list[Chunk]:
            idx = 0
            start = 0
            chunks = []
            while start < len(document.text):
                
                end = min(start + self.chunk_size, len(document.text))
                
                text = document.text[start:end]
                chunk = Chunk(
                    id=f"{document.id}_chunk_{idx}",
                    document_id=document.id,
                    text=text,
                    metadata={
                         **document.metadata,
                        "chunk_index": idx,
                        "start_char": start,
                        "end_char": end
                    }
                    )
                idx+=1
                start += self.chunk_size - self.overlap
                chunks.append(chunk)
            return chunks

