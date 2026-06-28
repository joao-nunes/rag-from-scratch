from abc import ABC, abstractmethod

from rag.chunkers.base import Chunk


class BasePromptBuilder(ABC):

    @abstractmethod
    def build(
        self,
        question: str,
        chunks: list[Chunk]
    ) -> str:
        """Build a prompt for the language model."""
        pass


class BaseGenerator(ABC):

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a response from a prompt."""
        pass