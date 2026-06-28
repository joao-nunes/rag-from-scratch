from openai import OpenAI
from rag.generators.base import BaseGenerator


class OpenAIGenerator(BaseGenerator):

    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        temperature: float = 0.0,
        api_key: str | None = None,
    ) -> None:
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature

    def generate(self, prompt: str) -> str:
        response = self.client.responses.create(
            model=self.model,
            input=prompt,
            temperature=self.temperature,
        )

        return response.output_text