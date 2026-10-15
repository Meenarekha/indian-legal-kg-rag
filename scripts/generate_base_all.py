"""
generate_base_all.py

Generate normal/Base LLM answers for the 30 fixed evaluation questions.

IMPORTANT:
- Uses the questions from data/evaluation/reference_answers.json
- Does NOT use BM25, FAISS, RRF, case ranking, or legal retrieval.
- Uses ProviderManager for:
      Gemini -> OpenAI -> Anthropic
- No retry loop is implemented here.
- If all providers fail for a query, completed answers are preserved and
  the batch stops cleanly.
- Existing valid Base answers are skipped.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(SCRIPT_DIR))

from llm.provider_manager import ProviderManager


# ============================================================
# CONFIGURATION
# ============================================================

START_QUERY = 1
END_QUERY = 30

PRIMARY_PROVIDER = "gemini"
USE_FALLBACK = True


# ============================================================
# PATHS
# ============================================================

EVALUATION_DIR = PROJECT_ROOT / "data" / "evaluation"

REFERENCE_FILE = EVALUATION_DIR / "reference_answers.json"

OUTPUT_DIR = EVALUATION_DIR / "generation_outputs"
FAILURE_DIR = EVALUATION_DIR / "generation_failures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FAILURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def qid(number):
    return f"Q{number:03d}"


def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_questions():
    if not REFERENCE_FILE.exists():
        raise FileNotFoundError(
            f"Reference file not found:\n{REFERENCE_FILE}"
        )

    with open(REFERENCE_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    answers = data.get("answers")

    if not isinstance(answers, list):
        raise RuntimeError(
            "reference_answers.json does not contain "
            "an 'answers' list."
        )

    questions = {}

    for item in answers:
        if not isinstance(item, dict):
            continue

        query_id = item.get("query_id")
        question = item.get("question")

        if query_id and question:
            questions[str(query_id).upper()] = {
                "query_id": str(query_id).upper(),
                "question": question,
                "reference_answer": item.get(
                    "reference_answer", ""
                )
            }

    if len(questions) != 30:
        print(
            f"WARNING: Expected 30 questions, "
            f"found {len(questions)}."
        )

    return questions


def find_existing_base(query_id):
    pattern = f"base_answer_{query_id}_*.json"

    files = list(OUTPUT_DIR.glob(pattern))

    if not files:
        return None

    files.sort(
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            answer = data.get("answer")

            if isinstance(answer, str) and answer.strip():
                return path

        except Exception:
            continue

    return None


def build_base_prompt(question):
    return f"""
You are a general-purpose language model answering an Indian
legal research and education question.

Question:
{question}

Provide a clear, concise and informative answer.

Important instructions:

1. Answer the question directly.
2. Use your general pretrained knowledge.
3. Do not use or claim to have searched any external legal
   corpus or retrieval system.
4. Do not invent case names, statutes, sections, legal rules,
   citations or facts.
5. If you are uncertain about a legal detail, clearly indicate
   the uncertainty.
6. Keep the answer focused on the question.
7. This answer is for an academic evaluation experiment and
   is not personalized legal advice.
""".strip()


def save_answer(query_id, question, answer, provider):
    path = (
        OUTPUT_DIR /
        f"base_answer_{query_id}_{timestamp()}.json"
    )

    result = {
        "query_id": query_id,
        "question": question,
        "generation_type": "BASE_LLM",
        "provider": provider,
        "retrieval_used": False,
        "external_legal_context_used": False,
        "answer": answer,
        "generated_at": datetime.now().isoformat()
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            result,
            f,
            indent=2,
            ensure_ascii=False
        )

    return path


def save_failure(query_id, question, error):
    path = (
        FAILURE_DIR /
        f"base_failure_{query_id}_{timestamp()}.json"
    )

    result = {
        "query_id": query_id,
        "question": question,
        "error_type": type(error).__name__,
        "error": str(error),
        "failed_at": datetime.now().isoformat()
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            result,
            f,
            indent=2,
            ensure_ascii=False
        )

    return path


def generate_answer(manager, question):
    result = manager.generate(
        prompt=build_base_prompt(question),
        primary_provider=PRIMARY_PROVIDER,
        use_fallback=USE_FALLBACK
    )

    if isinstance(result, dict):
        provider = result.get("provider", "unknown")
        answer = result.get("text", "")
    else:
        provider = PRIMARY_PROVIDER
        answer = str(result)

    if not answer or not answer.strip():
        raise RuntimeError(
            "LLM returned an empty answer."
        )

    return provider, answer.strip()


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("NORMAL / BASE LLM GENERATION")
    print("=" * 70)

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Reference    : {REFERENCE_FILE}")
    print(
        f"Query range  : {qid(START_QUERY)} "
        f"to {qid(END_QUERY)}"
    )
    print(
        "Provider     : "
        "Gemini -> OpenAI -> Anthropic"
    )
    print("Retrieval    : DISABLED")

    questions = load_questions()

    print()
    print(
        f"Loaded questions: {len(questions)}"
    )

    manager = ProviderManager()

    completed = []
    stopped_at = None

    for number in range(
        START_QUERY,
        END_QUERY + 1
    ):

        query_id = qid(number)

        if query_id not in questions:
            print()
            print(
                f"{query_id} not found in reference_answers.json."
            )
            stopped_at = query_id
            break

        item = questions[query_id]
        question = item["question"]

        print()
        print("=" * 70)
        print(f"PROCESSING {query_id}")
        print("=" * 70)
        print(f"Question: {question}")

        # ----------------------------------------------------
        # CHECKPOINT
        # ----------------------------------------------------

        existing = find_existing_base(query_id)

        if existing:
            print()
            print(
                f"BASE already exists: {existing.name}"
            )
            completed.append(query_id)
            continue

        try:

            print()
            print("Generating normal Base LLM answer...")

            provider, answer = generate_answer(
                manager,
                question
            )

            path = save_answer(
                query_id,
                question,
                answer,
                provider
            )

            completed.append(query_id)

            print()
            print(
                f"SUCCESS: {query_id}"
            )
            print(
                f"Provider: {provider}"
            )
            print(
                f"Saved: {path.name}"
            )

        except Exception as error:

            stopped_at = query_id

            print()
            print("=" * 70)
            print(f"{query_id} FAILED")
            print("=" * 70)
            print(f"Error type: {type(error).__name__}")
            print(f"Error: {error}")

            try:
                failure_path = save_failure(
                    query_id,
                    question,
                    error
                )

                print(
                    f"Failure log: {failure_path.name}"
                )

            except Exception as save_error:
                print(
                    f"Could not save failure log: "
                    f"{save_error}"
                )

            print()
            print(
                "Completed answers are preserved."
            )
            print(
                "Stopping batch without retrying this query."
            )

            break

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print()
    print("=" * 70)
    print("BASE LLM BATCH SUMMARY")
    print("=" * 70)

    print(
        f"Completed: {len(completed)}"
    )

    if completed:
        print(
            "Queries: "
            + ", ".join(completed)
        )

    if stopped_at:
        print(
            f"Stopped at: {stopped_at}"
        )
    else:
        print(
            "All requested Base LLM answers generated."
        )

    print()
    print(
        f"Output directory:\n{OUTPUT_DIR}"
    )

    print()
    print("Batch finished.")


if __name__ == "__main__":
    main()
