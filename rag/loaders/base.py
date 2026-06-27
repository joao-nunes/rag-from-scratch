from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Document:
    id: str
    text: str
    metadata: dict[str, Any]


class BaseLoader(ABC):

    @abstractmethod
    def load(self, source: Path) -> list[Document]:
        pass

