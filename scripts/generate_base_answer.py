import json
import sys
import os
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
# ============================================================
# PROJECT PATHS
# ============================================================

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "generation_outputs"
)

# ============================================================
# TEST QUESTION
# ============================================================

QUESTION = (
    "What are the conditions for granting anticipatory bail "
    "under Section 438 CrPC?"
)


# ============================================================
# BUILD BASE LLM PROMPT
# ============================================================

def build_prompt(question):

    return f"""
You are a general-purpose language model answering a legal
information question for an academic evaluation.

Answer the following question clearly and concisely:

{question}

Important instructions:

1. Provide a general informational answer.
2. Do not provide personalized legal advice.
3. Do not assume any documents or external evidence have
   been supplied.
4. Do not claim that the answer is based on our legal RAG
   corpus.
5. If appropriate, mention important legal principles and
   relevant statutory provisions that you know.
"""


# ============================================================
# GENERATE BASE ANSWER
# ============================================================

def generate_answer(question):

    scripts_dir = (
        PROJECT_ROOT
        / "scripts"
    )

    if str(scripts_dir) not in sys.path:

        sys.path.insert(
            0,
            str(scripts_dir)
        )

    from llm.provider_manager import ProviderManager

    print("=" * 70)
    print("GENERATING BASE LLM ANSWER")
    print("=" * 70)

    print("\nProvider: Gemini")
    print("External RAG context: NONE")
    print("Legal corpus retrieval: NONE")

    prompt = build_prompt(question)

    manager = ProviderManager()

    answer = manager.generate(
        prompt=prompt,
        primary_provider="gemini",
        use_fallback=False
    )

    return answer


# ============================================================
# SAVE RESULT
# ============================================================

def save_result(
    question,
    answer
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_file = (
        OUTPUT_DIR
        / f"base_answer_q001_{timestamp}.json"
    )

    result = {

        "query_id": "Q001",

        "question": question,

        "generation_type": "BASE_LLM",

        "provider": "gemini",

        "retrieval_used": False,

        "external_legal_context_used": False,

        "answer": answer,

        "generated_at": datetime.now().isoformat()
    }

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            indent=4,
            ensure_ascii=False
        )

    return output_file


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")

    print("=" * 70)
    print("INDIAN LEGAL RAG — BASELINE GENERATION TEST")
    print("=" * 70)

    print("\nQuestion:")
    print(QUESTION)

    try:

        # ----------------------------------------------------
        # Generate answer directly from Gemini
        # ----------------------------------------------------

        answer = generate_answer(
            QUESTION
        )

        # ----------------------------------------------------
        # Save result
        # ----------------------------------------------------

        output_file = save_result(
            QUESTION,
            answer
        )

        # ----------------------------------------------------
        # Display result
        # ----------------------------------------------------

        print("\n")

        print("=" * 70)
        print("BASE LLM ANSWER")
        print("=" * 70)

        print("\n")

        print(answer)

        print("\n")

        print("=" * 70)
        print("BASELINE GENERATION SUCCESSFUL")
        print("=" * 70)

        print(
            f"\nSaved result:\n"
            f"{output_file}"
        )

    except Exception as e:

        print("\n")

        print("=" * 70)
        print("BASELINE GENERATION FAILED")
        print("=" * 70)

        print(
            f"\nError type: "
            f"{type(e).__name__}"
        )

        print(
            f"Error: {e}"
        )


if __name__ == "__main__":
    main()