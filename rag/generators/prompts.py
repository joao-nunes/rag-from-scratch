from rag.generators.base import BasePromptBuilder
from rag.retrievers.base import RetrievalResult


class SimplePromptBuilder(BasePromptBuilder):

    def __init__(self, include_sources: bool = True):
        self.include_sources = include_sources

    def build(
        self,
        question: str,
        results: list[RetrievalResult],
    ) -> str:
        """
        Build a prompt for the language model using the retrieved chunks.
        """

        context_parts = []

        for result in results:

            chunk = result.chunk

            if self.include_sources:
                source = chunk.metadata.get("path", chunk.document_id)

                context_parts.append(
                    f"""Source: {source}

{chunk.text}"""
                )
            else:
                context_parts.append(chunk.text)

        separator = "\n\n" + "-" * 80 + "\n\n"
        context = separator.join(context_parts)

        prompt = f"""
You are a helpful AI assistant.

Answer the user's question using only the provided context.

If the context does not contain enough information, reply that you do not know.
Do not use outside knowledge.

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