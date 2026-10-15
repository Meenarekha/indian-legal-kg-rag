import os
import sys
import json
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCRIPTS_DIR = PROJECT_ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv(PROJECT_ROOT / ".env")


# ============================================================
# IMPORT RETRIEVAL PIPELINE
# ============================================================

from authority_rerank import (
    load_bm25,
    load_faiss,
    load_embedding_model,
    search,
)

from context_builder import (
    build_context_package,
    create_llm_context,
)

from llm.provider_manager import ProviderManager


# ============================================================
# CONFIGURATION
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

CONTEXT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "contexts"
)

FAILURE_DIR = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "generation_failures"
)

START_QUERY = 1
END_QUERY = 30

PRIMARY_PROVIDER = "gemini"
USE_FALLBACK = True


# ============================================================
# CREATE DIRECTORIES
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CONTEXT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FAILURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# QUERY ID
# ============================================================

def qid(number):
    return f"Q{number:03d}"


# ============================================================
# LOAD QUESTIONS
# ============================================================

def load_questions():

    if not REFERENCE_FILE.exists():
        raise FileNotFoundError(
            f"Reference file not found:\n{REFERENCE_FILE}"
        )

    with open(
        REFERENCE_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    questions = {}

    for item in data["answers"]:

        query_id = item["query_id"]

        questions[query_id] = {
            "question": item["question"]
        }

    return questions


# ============================================================
# CHECK EXISTING RAG ANSWER
# ============================================================

def existing_rag_answer(query_id):

    prefix = f"rag_answer_{query_id.lower()}_"

    candidates = list(
        OUTPUT_DIR.glob(
            f"{prefix}*.json"
        )
    )

    if not candidates:
        # Also support uppercase filenames
        candidates = list(
            OUTPUT_DIR.glob(
                f"rag_answer_{query_id}_*.json"
            )
        )

    if candidates:

        candidates.sort(
            key=lambda path: path.stat().st_mtime,
            reverse=True
        )

        return candidates[0]

    return None


# ============================================================
# BUILD RAG PROMPT
# ============================================================

def build_rag_prompt(
    question,
    llm_context
):

    prompt = f"""
You are an academic legal question-answering assistant.

You must answer the user's question using ONLY the
retrieved legal evidence provided below.

USER QUESTION:
{question}

RETRIEVED LEGAL EVIDENCE:
{llm_context}

INSTRUCTIONS:

1. Answer the question directly and clearly.

2. Base the answer primarily on the retrieved evidence.

3. Do not invent facts, cases, statutes, sections,
   legal principles, citations, or authorities that are
   not supported by the retrieved evidence.

4. If the retrieved evidence is insufficient to answer
   an important part of the question, explicitly state
   that the available retrieved evidence is insufficient.

5. Distinguish between what the retrieved cases establish
   and what cannot be established from the supplied evidence.

6. Mention relevant case names, courts, statutory provisions,
   or principles only when they appear in the retrieved
   evidence.

7. Do not use the reference answer.

8. Do not claim that you performed an external web search.

9. This is an academic RAG evaluation experiment and
   not personalized legal advice.

10. Prefer a concise, structured answer with short
    paragraphs or bullet points where appropriate.

ANSWER:
""".strip()

    return prompt


# ============================================================
# SAVE CONTEXT PACKAGE
# ============================================================

def save_context(
    query_id,
    context_package
):

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    filename = (
        f"rag_context_{query_id}_"
        f"{timestamp}.json"
    )

    output_path = CONTEXT_DIR / filename

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            context_package,
            file,
            indent=2,
            ensure_ascii=False
        )

    return output_path


# ============================================================
# SAVE RAG ANSWER
# ============================================================

def save_answer(
    query_id,
    question,
    answer,
    context_package
):

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    filename = (
        f"rag_answer_{query_id}_"
        f"{timestamp}.json"
    )

    output_path = OUTPUT_DIR / filename

    result = {
        "query_id": query_id,
        "question": question,
        "generation_type": "RAG",
        "provider": "provider_manager",
        "primary_provider": PRIMARY_PROVIDER,
        "fallback_enabled": USE_FALLBACK,

        "retrieval_used": True,

        "reference_answer_used": False,

        "retrieval_configuration": {
            "retrieval_pipeline":
                "BM25 + Dense FAISS + RRF + "
                "case aggregation + relevance filtering + "
                "authority-aware reranking",

            "max_cases":
                context_package["configuration"][
                    "max_cases"
                ],

            "supporting_chunks_per_case":
                context_package["configuration"][
                    "supporting_chunks_per_case"
                ],

            "max_context_characters":
                context_package["configuration"][
                    "max_context_characters"
                ],
        },

        "cases_considered":
            context_package["cases_considered"],

        "evidence_records":
            len(
                context_package["evidence_records"]
            ),

        "context_characters":
            context_package[
                "total_context_characters"
            ],

        "context_truncated":
            context_package["truncated"],

        "answer": answer,

        "generated_at":
            datetime.now().isoformat()
    }

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            result,
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

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    filename = (
        f"rag_failure_{query_id}_"
        f"{timestamp}.json"
    )

    output_path = FAILURE_DIR / filename

    result = {
        "query_id": query_id,
        "question": question,
        "error_type": type(error).__name__,
        "error": str(error),
        "failed_at": datetime.now().isoformat()
    }

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            result,
            file,
            indent=2,
            ensure_ascii=False
        )

    return output_path


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print("CHECKPOINTED LEGAL RAG BATCH GENERATION")
    print("=" * 80)

    print(
        f"\nProject root : {PROJECT_ROOT}"
    )

    print(
        f"Start query  : {qid(START_QUERY)}"
    )

    print(
        f"End query    : {qid(END_QUERY)}"
    )

    print(
        "\nProvider     : Gemini -> OpenRouter -> Grok"
    )

    print(
        "Reference answers used for generation: NO"
    )

    print(
        "Retrieval    : ENABLED"
    )

    # ========================================================
    # LOAD QUESTIONS
    # ========================================================

    questions = load_questions()

    print(
        f"\nReference questions loaded: "
        f"{len(questions)}"
    )

    # ========================================================
    # LOAD RETRIEVAL SYSTEM
    # ========================================================

    print()
    print("=" * 80)
    print("LOADING RETRIEVAL SYSTEM")
    print("=" * 80)

    print("\nLoading BM25 index...")

    bm25, bm25_metadata = load_bm25()

    print(
        f"BM25 documents loaded: "
        f"{len(bm25_metadata)}"
    )

    print("\nLoading FAISS index...")

    faiss_index, faiss_metadata = load_faiss()

    print(
        f"FAISS vectors loaded: "
        f"{faiss_index.ntotal}"
    )

    print(
        "\nLoading embedding model..."
    )

    embedding_model = load_embedding_model()

    print(
        "Embedding model loaded."
    )

    # ========================================================
    # VALIDATE INDEXES
    # ========================================================

    if len(bm25_metadata) != bm25.corpus_size:

        raise ValueError(
            "BM25 metadata count does not match "
            "BM25 corpus size."
        )

    if len(faiss_metadata) != faiss_index.ntotal:

        raise ValueError(
            "FAISS metadata count does not match "
            "FAISS index size."
        )

    print(
        "\nIndex validation passed."
    )

    # ========================================================
    # LOAD LLM MANAGER
    # ========================================================

    print()
    print("=" * 80)
    print("LOADING LLM PROVIDER MANAGER")
    print("=" * 80)

    manager = ProviderManager()

    print(
        "Provider manager loaded successfully."
    )

    # ========================================================
    # BATCH GENERATION
    # ========================================================

    completed = []
    skipped = []
    failures = []

    for number in range(
        START_QUERY,
        END_QUERY + 1
    ):

        query_id = qid(number)

        # ----------------------------------------------------
        # CHECK QUESTION
        # ----------------------------------------------------

        if query_id not in questions:

            print()
            print(
                f"{query_id} missing from "
                "reference_answers.json."
            )

            failures.append(query_id)
            continue

        question = questions[
            query_id
        ]["question"]

        print()
        print("#" * 80)
        print(
            f"PROCESSING {query_id}: {question}"
        )
        print("#" * 80)

        # ----------------------------------------------------
        # CHECKPOINT
        # ----------------------------------------------------

        existing = existing_rag_answer(
            query_id
        )

        if existing:

            print()
            print(
                f"RAG already exists: "
                f"{existing.name}"
            )

            skipped.append(query_id)
            continue

        # ----------------------------------------------------
        # RETRIEVAL
        # ----------------------------------------------------

        try:

            print(
                "\nBuilding legal retrieval context..."
            )

            (
                original_cases,
                filtered_cases,
                reranked_cases
            ) = search(
                query=question,
                bm25=bm25,
                bm25_metadata=bm25_metadata,
                faiss_index=faiss_index,
                faiss_metadata=faiss_metadata,
                embedding_model=embedding_model
            )

            print(
                f"Retrieved cases: "
                f"{len(original_cases)}"
            )

            print(
                f"Relevant cases after filtering: "
                f"{len(filtered_cases)}"
            )

            print(
                f"Final reranked cases: "
                f"{len(reranked_cases)}"
            )

        except Exception as error:

            print(
                "\nRETRIEVAL FAILED."
            )

            print(
                f"Error: {type(error).__name__}: {error}"
            )

            path = save_failure(
                query_id,
                question,
                error
            )

            print(
                f"Failure saved: {path}"
            )

            failures.append(query_id)

            continue

        # ----------------------------------------------------
        # CONTEXT BUILDING
        # ----------------------------------------------------

        try:

            print(
                "\nBuilding evidence package..."
            )

            context_package = (
                build_context_package(
                    question,
                    reranked_cases
                )
            )

            print(
                "Evidence package created."
            )

            print(
                f"Cases considered: "
                f"{context_package['cases_considered']}"
            )

            print(
                f"Evidence records: "
                f"{len(context_package['evidence_records'])}"
            )

            print(
                f"Context characters: "
                f"{context_package['total_context_characters']}"
            )

            print(
                f"Context truncated: "
                f"{context_package['truncated']}"
            )

            context_path = save_context(
                query_id,
                context_package
            )

            print(
                f"Context saved: {context_path}"
            )

        except Exception as error:

            print(
                "\nCONTEXT BUILDING FAILED."
            )

            print(
                f"Error: {type(error).__name__}: {error}"
            )

            path = save_failure(
                query_id,
                question,
                error
            )

            print(
                f"Failure saved: {path}"
            )

            failures.append(query_id)

            continue

        # ----------------------------------------------------
        # CREATE LLM CONTEXT
        # ----------------------------------------------------

        try:

            llm_context = (
                create_llm_context(
                    context_package
                )
            )

            print(
                "\nLLM evidence context created."
            )

        except Exception as error:

            print(
                "\nLLM CONTEXT CREATION FAILED."
            )

            print(
                f"Error: {type(error).__name__}: {error}"
            )

            path = save_failure(
                query_id,
                question,
                error
            )

            print(
                f"Failure saved: {path}"
            )

            failures.append(query_id)

            continue

        # ----------------------------------------------------
        # GENERATION
        # ----------------------------------------------------

        print()
        print(
            f"RAG GENERATION: {query_id}"
        )

        prompt = build_rag_prompt(
            question,
            llm_context
        )

        try:

            result = manager.generate(
                prompt=prompt,
                primary_provider=PRIMARY_PROVIDER,
                use_fallback=USE_FALLBACK
            )

            # Current ProviderManager returns text.
            # Support dict results too, if used later.

            if isinstance(result, dict):

                answer = result.get(
                    "text",
                    ""
                )

                provider = result.get(
                    "provider",
                    "unknown"
                )

            else:

                answer = str(result)

                provider = "provider_manager"

            if not answer.strip():

                raise RuntimeError(
                    "LLM returned an empty answer."
                )

            # ------------------------------------------------
            # SAVE ANSWER
            # ------------------------------------------------

            output_path = save_answer(
                query_id,
                question,
                answer.strip(),
                context_package
            )

            completed.append(query_id)

            print()
            print(
                "RAG generation successful."
            )

            print(
                f"Provider result: {provider}"
            )

            print(
                f"Saved: {output_path}"
            )

        except Exception as error:

            print()
            print(
                "RAG GENERATION FAILED."
            )

            print(
                f"Error: {type(error).__name__}: {error}"
            )

            path = save_failure(
                query_id,
                question,
                error
            )

            print(
                f"Failure saved: {path}"
            )

            failures.append(query_id)

            # Stop on generation failure so we can inspect
            # the provider problem rather than silently
            # continuing through the batch.

            print(
                "\nStopping batch."
            )

            break

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 80)
    print("RAG GENERATION SUMMARY")
    print("=" * 80)

    print(
        f"\nRequested: {qid(START_QUERY)} "
        f"to {qid(END_QUERY)}"
    )

    print(
        f"Generated: {len(completed)}"
    )

    print(
        f"Skipped existing: {len(skipped)}"
    )

    print(
        f"Failures: {len(failures)}"
    )

    if completed:

        print(
            "\nGenerated queries:"
        )

        print(
            ", ".join(completed)
        )

    if skipped:

        print(
            "\nSkipped queries:"
        )

        print(
            ", ".join(skipped)
        )

    if failures:

        print(
            "\nFailed queries:"
        )

        print(
            ", ".join(failures)
        )

    else:

        print(
            "\nAll RAG answers generated successfully."
        )

    print(
        "\nReference answers were NOT used during generation."
    )

    print(
        "Retrieval WAS used."
    )

    print(
        f"\nOutput directory:\n{OUTPUT_DIR}"
    )

    print(
        f"\nContext directory:\n{CONTEXT_DIR}"
    )

    print(
        "\nBatch finished."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()