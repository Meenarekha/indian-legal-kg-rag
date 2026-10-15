from .gemini_provider import GeminiProvider
from .groq_provider import GroqProvider
from .openrouter_provider import OpenRouterProvider
from .grok_provider import GrokProvider


class ProviderManager:

    def __init__(self):

        self.providers = {}

        # ====================================================
        # GEMINI
        # ====================================================

        try:

            self.providers["gemini"] = GeminiProvider()

            print(
                "Provider initialized: Gemini"
            )

        except Exception as error:

            print(
                f"Gemini unavailable: {error}"
            )

        # ====================================================
        # GROQ
        # ====================================================

        try:

            self.providers["groq"] = GroqProvider()

            print(
                "Provider initialized: Groq"
            )

        except Exception as error:

            print(
                f"Groq unavailable: {error}"
            )

        # ====================================================
        # OPENROUTER
        # ====================================================

        try:

            self.providers["openrouter"] = (
                OpenRouterProvider()
            )

            print(
                "Provider initialized: OpenRouter"
            )

        except Exception as error:

            print(
                f"OpenRouter unavailable: {error}"
            )

        # ====================================================
        # XAI GROK
        # ====================================================

        try:

            self.providers["grok"] = GrokProvider()

            print(
                "Provider initialized: xAI Grok"
            )

        except Exception as error:

            print(
                f"xAI Grok unavailable: {error}"
            )

    # ========================================================
    # GENERATE
    # ========================================================

    def generate(
        self,
        prompt,
        primary_provider=None,
        use_fallback=True
    ):

        if not self.providers:

            raise RuntimeError(
                "No LLM providers are available."
            )

        # ====================================================
        # BUILD PROVIDER ORDER
        # ====================================================

        if primary_provider:

            provider_order = []

            if primary_provider in self.providers:

                provider_order.append(
                    primary_provider
                )

            if use_fallback:

                fallback_order = [
                    "gemini",
                    "groq",
                    "openrouter",
                    "grok"
                ]

                for provider_name in fallback_order:

                    if (
                        provider_name in self.providers
                        and
                        provider_name not in provider_order
                    ):

                        provider_order.append(
                            provider_name
                        )

        else:

            provider_order = [
                provider_name
                for provider_name in [
                    "gemini",
                    "groq",
                    "openrouter",
                    "grok"
                ]
                if provider_name in self.providers
            ]

        # ====================================================
        # TRY PROVIDERS
        # ====================================================

        errors = []

        for provider_name in provider_order:

            provider = self.providers[
                provider_name
            ]

            print(
                f"\nTrying LLM provider: "
                f"{provider_name}"
            )

            try:

                answer = provider.generate(
                    prompt
                )

                print(
                    f"Provider SUCCESS: "
                    f"{provider_name}"
                )

                return answer

            except Exception as error:

                print(
                    f"Provider FAILED: "
                    f"{provider_name}"
                )

                print(
                    f"Reason: {error}"
                )

                errors.append(
                    f"{provider_name}: {error}"
                )

                if not use_fallback:

                    break

        # ====================================================
        # ALL FAILED
        # ====================================================

        raise RuntimeError(
            "All configured LLM providers failed.\n"
            + "\n".join(errors)
        )