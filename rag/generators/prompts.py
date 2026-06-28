from rag.chunkers.base import Chunk
from rag.generators.base import BasePromptBuilder


class SimplePromptBuilder(BasePromptBuilder):

    def __init__(self, include_sources: bool = True):
        self.include_sources = include_sources

    def build(
        self,
        question: str,
        chunks: list[Chunk],
    ) -> str:
        """
        Build a prompt for the language model using the retrieved chunks.
        """

        context_parts = []

        for chunk in chunks:

            if self.include_sources:
                source = chunk.metadata.get("path", chunk.document_id)

                context_parts.append(
                    f"""Source: {source}

{chunk.text}"""
                )
            else:
                context_parts.append(chunk.text)

        context = "\n\n" + "-" * 80 + "\n\n".join(context_parts)

        prompt = f"""
You are a helpful AI assistant.

Answer the user's question using ONLY the provided context.

If the answer cannot be found in the context, say that you do not know.

Context
=======

{context}

Question
========

{question}

Answer
======
"""

        return prompt.strip()