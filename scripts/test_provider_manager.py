import sys
from pathlib import Path


# ============================================================
# PROJECT PATH
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

# Make sure Python can find scripts/
sys.path.insert(0, str(SCRIPT_DIR))


# ============================================================
# IMPORT OUR LOCAL PROVIDER MANAGER
# ============================================================

from llm.provider_manager import ProviderManager


# ============================================================
# TEST QUESTION
# ============================================================

QUESTION = """
What is the purpose of bail in criminal proceedings?

Give a concise answer in 3 to 5 sentences.
Do not provide legal advice.
"""


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("LLM PROVIDER MANAGER TEST")
    print("=" * 70)

    try:

        manager = ProviderManager()

        answer = manager.generate(
            prompt=QUESTION,
            primary_provider="gemini",
            use_fallback=True
        )

        print("\n" + "=" * 70)
        print("FINAL ANSWER")
        print("=" * 70)

        print(answer)

    except Exception as e:

        print("\n" + "=" * 70)
        print("ALL PROVIDERS FAILED")
        print("=" * 70)

        print(f"\nError type: {type(e).__name__}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()