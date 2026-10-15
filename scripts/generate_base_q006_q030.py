"""
generate_base_q006_q030.py

Generate NORMAL / BASE LLM answers for Q006-Q030.

IMPORTANT:
The reference answers are NOT used for generation.

The reference file is used ONLY as a source of:
    - query_id
    - question

The "reference_answer" field is deliberately ignored.

Generation:

    Question
       |
       v
    Gemini
       |
       | if Gemini fails
       v
    Grok
       |
       | if Grok fails
       v
    STOP

No:
    - BM25
    - FAISS
    - RRF
    - Case aggregation
    - Authority ranking
    - Retrieved legal context
    - Reference answer context
    - Retry loops
    - Waiting

Q001-Q005 are already completed and are preserved.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

load_dotenv(PROJECT_ROOT / ".env")

sys.path.insert(0, str(SCRIPT_DIR))
# ============================================================
# PROJECT PATH
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(SCRIPT_DIR))


# ============================================================
# FILE PATHS
# ============================================================

REFERENCE_FILE = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "reference_answers.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "generation_outputs"
)

FAILURE_DIR = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "generation_failures"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FAILURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# GENERATION RANGE
# ============================================================

START_QUERY = 6
END_QUERY = 30


# ============================================================
# LLM CONFIGURATION
# ============================================================

PRIMARY_PROVIDER = "gemini"

USE_FALLBACK = True


# ============================================================
# PROVIDER MANAGER
# ============================================================

from llm.provider_manager import ProviderManager


# ============================================================
# QUERY ID
# ============================================================

def make_query_id(number):

    return f"Q{number:03d}"


# ============================================================
# TIMESTAMP
# ============================================================

def timestamp():

    return datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )


# ============================================================
# LOAD QUESTIONS ONLY
# ============================================================

def load_questions():

    if not REFERENCE_FILE.exists():

        raise FileNotFoundError(
            f"Reference file not found:\n"
            f"{REFERENCE_FILE}"
        )

    with open(
        REFERENCE_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

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

        if not query_id or not question:
            continue

        # ----------------------------------------------------
        # IMPORTANT
        #
        # We deliberately extract ONLY:
        #
        #   query_id
        #   question
        #
        # reference_answer is NOT loaded.
        # ----------------------------------------------------

        questions[str(query_id).upper()] = {
            "query_id": str(query_id).upper(),
            "question": question
        }

    if not questions:

        raise RuntimeError(
            "No questions found in reference_answers.json."
        )

    return questions


# ============================================================
# FIND EXISTING BASE ANSWER
# ============================================================

def find_existing_answer(query_id):

    pattern = (
        f"base_answer_{query_id}_*.json"
    )

    files = list(
        OUTPUT_DIR.glob(pattern)
    )

    if not files:

        return None

    files.sort(
        key=lambda path: path.stat().st_mtime,
        reverse=True
    )

    for path in files:

        try:

            with open(
                path,
                "r",
                encoding="utf-8"
            ) as file:

                data = json.load(file)

            answer = data.get("answer")

            if (
                isinstance(answer, str)
                and answer.strip()
            ):

                return path

        except Exception:

            continue

    return None


# ============================================================
# BASE LLM PROMPT
# ============================================================

def build_base_prompt(question):

    return f"""
You are a general-purpose language model answering an
Indian legal research and education question.

Question:

{question}

Answer the question clearly, accurately, and concisely
using your own pretrained/general knowledge.

Important instructions:

1. Answer ONLY the question provided above.

2. Do NOT use any retrieved documents.

3. Do NOT use any legal context from a RAG system.

4. Do NOT use a reference answer.

5. Do NOT assume that a reference answer has been provided.

6. Do NOT claim that you searched the project's legal corpus.

7. Do NOT invent case names, statutes, sections,
   legal principles, citations, or facts.

8. If you are uncertain about a legal detail,
   clearly state the uncertainty.

9. This response is being generated as a BASELINE
   for an academic experiment.

10. This is not personalized legal advice.
""".strip()


# ============================================================
# GENERATE USING PROVIDER MANAGER
# ============================================================

def generate_base_answer(
    manager,
    question
):

    prompt = build_base_prompt(
        question
    )

    result = manager.generate(
        prompt=prompt,
        primary_provider=PRIMARY_PROVIDER,
        use_fallback=USE_FALLBACK
    )

    if isinstance(result, dict):

        provider = result.get(
            "provider",
            "unknown"
        )

        answer = result.get(
            "text",
            ""
        )

    else:

        provider = PRIMARY_PROVIDER

        answer = str(result)

    if not answer or not answer.strip():

        raise RuntimeError(
            "LLM returned an empty answer."
        )

    return provider, answer.strip()


# ============================================================
# SAVE BASE ANSWER
# ============================================================

def save_base_answer(
    query_id,
    question,
    answer,
    provider
):

    output_path = (
        OUTPUT_DIR
        / f"base_answer_{query_id}_{timestamp()}.json"
    )

    data = {

        "query_id": query_id,

        "question": question,

        "generation_type": "BASE_LLM",

        "provider": provider,

        "retrieval_used": False,

        "external_legal_context_used": False,

        "reference_answer_used": False,

        "answer": answer,

        "generated_at": datetime.now().isoformat()
    }

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False
        )

    return output_path


# ============================================================
# SAVE FAILURE
# ============================================================

def save_failure(
    query_id,
    question,
    error
):

    failure_path = (
        FAILURE_DIR
        / f"base_failure_{query_id}_{timestamp()}.json"
    )

    data = {

        "query_id": query_id,

        "question": question,

        "generation_type": "BASE_LLM",

        "error_type": type(error).__name__,

        "error": str(error),

        "failed_at": datetime.now().isoformat()
    }

    with open(
        failure_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False
        )

    return failure_path


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("NORMAL / BASE LLM GENERATION")
    print("=" * 70)

    print()
    print(
        f"Generation range: "
        f"{make_query_id(START_QUERY)} "
        f"to "
        f"{make_query_id(END_QUERY)}"
    )

    print()
    print(
        f"Primary provider: {PRIMARY_PROVIDER}"
    )

    print(
        "Fallback: Gemini -> Grok"
    )

    print()
    print(
        "Retrieval: DISABLED"
    )

    print(
        "Reference answers: NOT USED FOR GENERATION"
    )

    # --------------------------------------------------------
    # LOAD QUESTIONS
    # --------------------------------------------------------

    questions = load_questions()

    print()
    print(
        f"Questions loaded: {len(questions)}"
    )

    # --------------------------------------------------------
    # CREATE PROVIDER MANAGER
    # --------------------------------------------------------

    manager = ProviderManager()

    # --------------------------------------------------------
    # TRACKING
    # --------------------------------------------------------

    completed = []

    stopped_at = None

    # --------------------------------------------------------
    # PROCESS Q006-Q030
    # --------------------------------------------------------

    for number in range(
        START_QUERY,
        END_QUERY + 1
    ):

        query_id = make_query_id(
            number
        )

        print()
        print()
        print("=" * 70)
        print(
            f"PROCESSING {query_id}"
        )
        print("=" * 70)

        # ----------------------------------------------------
        # CHECK QUESTION
        # ----------------------------------------------------

        if query_id not in questions:

            print()
            print(
                f"ERROR: {query_id} not found."
            )

            stopped_at = query_id

            break

        question = questions[
            query_id
        ]["question"]

        print()
        print(
            f"Question: {question}"
        )

        # ----------------------------------------------------
        # CHECKPOINT
        # ----------------------------------------------------

        existing = find_existing_answer(
            query_id
        )

        if existing:

            print()
            print(
                f"BASE already exists: "
                f"{existing.name}"
            )

            completed.append(
                query_id
            )

            continue

        # ----------------------------------------------------
        # GENERATE
        # ----------------------------------------------------

        try:

            print()
            print(
                "Generating Base LLM answer..."
            )

            provider, answer = (
                generate_base_answer(
                    manager,
                    question
                )
            )

            # ------------------------------------------------
            # SAVE IMMEDIATELY
            # ------------------------------------------------

            output_path = save_base_answer(
                query_id=query_id,
                question=question,
                answer=answer,
                provider=provider
            )

            completed.append(
                query_id
            )

            print()
            print(
                f"{query_id} SUCCESS"
            )

            print(
                f"Provider: {provider}"
            )

            print(
                f"Saved: {output_path.name}"
            )

        except Exception as error:

            # ------------------------------------------------
            # STOP
            # ------------------------------------------------

            stopped_at = query_id

            print()
            print("=" * 70)
            print(
                f"{query_id} FAILED"
            )
            print("=" * 70)

            print(
                f"Error type: "
                f"{type(error).__name__}"
            )

            print(
                f"Error: {error}"
            )

            # ------------------------------------------------
            # SAVE FAILURE LOG
            # ------------------------------------------------

            try:

                failure_path = save_failure(
                    query_id,
                    question,
                    error
                )

                print()
                print(
                    f"Failure log saved:"
                )

                print(
                    failure_path.name
                )

            except Exception as save_error:

                print()
                print(
                    f"Could not save failure log:"
                )

                print(
                    save_error
                )

            print()
            print(
                "All previously completed answers "
                "have been preserved."
            )

            print(
                "Stopping batch."
            )

            break

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print()
    print("=" * 70)
    print("BASE LLM GENERATION SUMMARY")
    print("=" * 70)

    print()
    print(
        f"Requested: "
        f"{make_query_id(START_QUERY)} "
        f"to "
        f"{make_query_id(END_QUERY)}"
    )

    print(
        f"Completed: {len(completed)}"
    )

    if completed:

        print()
        print(
            "Completed queries:"
        )

        print(
            ", ".join(completed)
        )

    if stopped_at:

        print()
        print(
            f"Stopped at: {stopped_at}"
        )

    else:

        print()
        print(
            "All Base LLM answers generated."
        )

    print()
    print(
        "Reference answers were NOT used "
        "during generation."
    )

    print(
        "Retrieval was NOT used."
    )

    print()
    print(
        f"Output directory:"
    )

    print(
        OUTPUT_DIR
    )

    print()
    print(
        "Batch finished."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()