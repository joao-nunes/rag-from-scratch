from abc import abstractmethod, ABC
from rag.loaders.base import Document
from dataclasses import dataclass
from typing import Any

@dataclass
class Chunk:
    id: str
    document_id: str
    text: str
    metadata: dict[str, Any]

class BaseChunker(ABC):

    @abstractmethod
    def chunk(
        self,
        document: Document
    ) -> list[Chunk]:
        pass