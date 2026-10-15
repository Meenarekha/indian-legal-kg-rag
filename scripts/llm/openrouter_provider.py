import os
from openai import OpenAI

from .base import LLMProvider


class OpenRouterProvider(LLMProvider):
    """
    OpenRouter fallback provider for the legal RAG system.

    Uses a specific instruction-tuned free model rather than
    OpenRouter's automatic free-model router.

    This gives us more predictable behavior across the
    evaluation questions.
    """

    def __init__(self):

        # ====================================================
        # LOAD API KEY
        # ====================================================

        self.api_key = os.getenv(
            "OPENROUTER_API_KEY"
        )

        if not self.api_key:

            raise RuntimeError(
                "OPENROUTER_API_KEY is not set in the .env file."
            )

        # ====================================================
        # CREATE CLIENT
        # ====================================================

        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1"
        )

        # ====================================================
        # SPECIFIC FREE MODEL
        # ====================================================

        self.model_name = (
            "google/gemma-4-31b-it:free"
        )

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
                    "OpenRouter returned no choices."
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

            # Normal response
            if isinstance(text, str):

                text = text.strip()

                if text:

                    return text

            # ------------------------------------------------
            # HANDLE STRUCTURED CONTENT
            # ------------------------------------------------

            if isinstance(text, list):

                parts = []

                for item in text:

                    if isinstance(item, dict):

                        item_text = item.get(
                            "text"
                        )

                    else:

                        item_text = getattr(
                            item,
                            "text",
                            None
                        )

                    if item_text:

                        parts.append(
                            str(item_text)
                        )

                combined_text = "\n".join(
                    parts
                ).strip()

                if combined_text:

                    return combined_text

            # ------------------------------------------------
            # NO USABLE TEXT
            # ------------------------------------------------

            finish_reason = getattr(
                choice,
                "finish_reason",
                None
            )

            model_used = getattr(
                response,
                "model",
                self.model_name
            )

            raise RuntimeError(
                "OpenRouter returned no usable text. "
                f"model={model_used}, "
                f"finish_reason={finish_reason}"
            )

        except Exception as error:

            raise RuntimeError(
                f"OpenRouter generation failed: {error}"
            )