import os
import time

from google import genai

from .base import LLMProvider


class GeminiProvider(LLMProvider):

    def __init__(self):

        self.api_key = os.getenv("GEMINI_API_KEY")

        if not self.api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set in the environment."
            )

        self.client = genai.Client(
            api_key=self.api_key
        )

        self.model_name = "gemini-3.6-flash"

    def generate(self, prompt: str) -> str:

        try:

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )

            if response is None:
                raise RuntimeError(
                    "Gemini returned no response."
                )

            text = getattr(
                response,
                "text",
                None
            )

            if not text or not text.strip():
                raise RuntimeError(
                    "Gemini returned an empty response."
                )

            return text.strip()

        except Exception as error:

            error_text = str(error)

            # ------------------------------------------------
            # IMPORTANT:
            # Do NOT retry quota exhaustion.
            # ProviderManager will immediately try Grok.
            # ------------------------------------------------

            if (
                "429" in error_text
                or "RESOURCE_EXHAUSTED" in error_text
                or "quota" in error_text.lower()
                or "rate limit" in error_text.lower()
            ):

                raise RuntimeError(
                    f"Gemini quota/rate limit error: "
                    f"{error_text}"
                )

            # ------------------------------------------------
            # Other Gemini errors are also passed to the
            # ProviderManager so it can try Grok.
            # ------------------------------------------------

            raise RuntimeError(
                f"Gemini generation failed: "
                f"{error_text}"
            )