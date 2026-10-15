import os

from groq import Groq

from .base import LLMProvider


class GroqProvider(LLMProvider):
    """
    Groq LLM provider for the legal RAG system.

    Used as a fallback when the primary Gemini provider
    is unavailable because of quota/rate-limit problems.
    """

    def __init__(self):

        # ====================================================
        # LOAD API KEY
        # ====================================================

        self.api_key = os.getenv("GROQ_API_KEY")

        if not self.api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set in the .env file."
            )

        # ====================================================
        # CREATE GROQ CLIENT
        # ====================================================

        self.client = Groq(
            api_key=self.api_key
        )

        # ====================================================
        # MODEL
        # ====================================================

        self.model_name = "openai/gpt-oss-120b"
    # ========================================================
    # GENERATE
    # ========================================================

    def generate(self, prompt):

        try:

            response = self.client.chat.completions.create(

                model=self.model_name,

                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],

                temperature=0.2,

                max_tokens=3000
            )

            # ------------------------------------------------
            # VALIDATE RESPONSE
            # ------------------------------------------------

            if not response.choices:

                raise RuntimeError(
                    "Groq returned no choices."
                )

            choice = response.choices[0]

            message = choice.message

            # ------------------------------------------------
            # EXTRACT TEXT
            # ------------------------------------------------

            text = getattr(
                message,
                "content",
                None
            )

            if isinstance(text, str):

                text = text.strip()

                if text:
                    return text

            # ------------------------------------------------
            # NO USABLE TEXT
            # ------------------------------------------------

            finish_reason = getattr(
                choice,
                "finish_reason",
                None
            )

            raise RuntimeError(
                "Groq returned no usable text. "
                f"finish_reason={finish_reason}"
            )

        except Exception as error:

            raise RuntimeError(
                f"Groq generation failed: {error}"
            )