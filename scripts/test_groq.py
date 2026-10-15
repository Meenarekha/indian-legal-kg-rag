import os
import sys
from pathlib import Path

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)

# ============================================================
# LOAD ENVIRONMENT
# ============================================================

from dotenv import load_dotenv

load_dotenv(
    PROJECT_ROOT / ".env"
)

# ============================================================
# IMPORT PROVIDER
# ============================================================

from scripts.llm.groq_provider import GroqProvider


def main():

    print("=" * 70)
    print("GROQ CONNECTION TEST")
    print("=" * 70)

    try:

        provider = GroqProvider()

        print(
            "\nProvider initialized successfully."
        )

        print(
            f"Model: {provider.model_name}"
        )

        print(
            "\nSending test request..."
        )

        answer = provider.generate(
            """
You are testing an LLM connection.

Answer in one short sentence:

What is the purpose of a legal case retrieval system?
"""
        )

        print("\n" + "=" * 70)
        print("GROQ SUCCESS")
        print("=" * 70)

        print(answer)

    except Exception as error:

        print("\n" + "=" * 70)
        print("GROQ FAILED")
        print("=" * 70)

        print(error)


if __name__ == "__main__":
    main()