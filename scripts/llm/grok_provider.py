import os

from openai import OpenAI

from .base import LLMProvider


class GrokProvider(LLMProvider):

    def __init__(self):

        self.api_key = os.getenv("XAI_API_KEY")

        if not self.api_key:
            raise RuntimeError(
                "XAI_API_KEY is not set."
            )

        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.x.ai/v1"
        )

        self.model_name = "grok-4.6"

    def generate(self, prompt: str) -> str:

        response = self.client.responses.create(
            model=self.model_name,
            input=prompt
        )

        if not response or not response.output_text:
            raise RuntimeError(
                "Grok returned an empty response."
            )

        return response.output_text