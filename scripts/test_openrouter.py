import os
from pathlib import Path

from dotenv import load_dotenv

from llm.openrouter_provider import OpenRouterProvider


PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(
    PROJECT_ROOT / ".env"
)


def main():

    print("=" * 70)
    print("OPENROUTER CONNECTION TEST")
    print("=" * 70)

    provider = OpenRouterProvider()

    print(
        "\nProvider initialized successfully."
    )

    print(
        "Model:",
        provider.model_name
    )

    prompt = """
Answer this simple question in two sentences:

What is bail in criminal law?

This is only an API connection test.
""".strip()

    print(
        "\nSending test request..."
    )

    answer = provider.generate(
        prompt
    )

    print(
        "\nOPENROUTER SUCCESS"
    )

    print(
        "=" * 70
    )

    print(answer)

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()