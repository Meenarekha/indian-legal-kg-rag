"""
generate_all_answers.py

Checkpointed batch generation for the Legal Query RAG project.

Pipeline:

    Reference Questions
            |
            +----------------------+
            |                      |
            v                      v
       BASE LLM                RAG SYSTEM
       No retrieval              |
            |                    v
            |             Hybrid Retrieval
            |                    |
            |             Case Aggregation
            |                    |
            |             Authority Reranking
            |                    |
            |             Evidence Packaging
            |                    |
            +---------> LLM Generation
                         |
                         v
                    Save Answers

LLM provider fallback:

    Gemini
       |
       v
    OpenAI
       |
       v
    Anthropic

Important behavior:

    - No provider retry loops.
    - No waiting after API failure.
    - If Gemini fails, immediately try OpenAI.
    - If OpenAI fails, immediately try Anthropic.
    - If all providers fail, stop the batch.
    - All completed answers remain saved.
    - Restarting the batch skips already completed valid answers.
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any


# ============================================================
# PROJECT PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(SCRIPT_DIR))


# ============================================================
# CONFIGURATION
# ============================================================

# Change these when you want to process a different range.

START_QUERY = 2
END_QUERY = 30

# Primary provider.
PRIMARY_PROVIDER = "gemini"

# Enable fallback:
# Gemini -> OpenAI -> Anthropic
USE_FALLBACK = True


# ============================================================
# DIRECTORIES
# ============================================================

EVALUATION_DIR = PROJECT_ROOT / "data" / "evaluation"

REFERENCE_FILE = (
    EVALUATION_DIR /
    "generation_reference_answers.json"
)

GENERATION_OUTPUT_DIR = (
    EVALUATION_DIR /
    "generation_outputs"
)

FAILURE_DIR = (
    EVALUATION_DIR /
    "generation_failures"
)

FAILURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

GENERATION_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# IMPORT PROJECT MODULES
# ============================================================

from authority_rerank import (
    load_bm25,
    load_faiss,
    load_embedding_model,
    search
)

from context_builder import (
    build_context_package,
    create_llm_context,
    save_context_package
)

from llm.provider_manager import ProviderManager


# ============================================================
# GENERAL HELPERS
# ============================================================

def now_string() -> str:
    """
    Return a timestamp suitable for filenames.
    """

    return datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )


def query_id_string(number: int) -> str:
    """
    Convert integer query number into Q001 format.
    """

    return f"Q{number:03d}"


def normalize_query_id(value: Any) -> str:
    """
    Normalize different possible query ID formats.

    Examples:

        1      -> Q001
        "1"    -> Q001
        "Q001" -> Q001
        "q001" -> Q001
    """

    if isinstance(value, int):
        return query_id_string(value)

    value = str(value).strip().upper()

    if value.startswith("Q"):
        numeric_part = value[1:]
    else:
        numeric_part = value

    try:
        number = int(numeric_part)
        return query_id_string(number)
    except ValueError:
        return value


# ============================================================
# REFERENCE QUESTIONS
# ============================================================

def load_reference_questions() -> Dict[str, Dict[str, Any]]:
    """
    Load the evaluation/reference question file.

    The project has 30 evaluation questions.
    """

    if not REFERENCE_FILE.exists():

        raise FileNotFoundError(
            f"Reference file not found:\n"
            f"{REFERENCE_FILE}"
        )

    with open(
        REFERENCE_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    questions = {}

    # --------------------------------------------------------
    # Case 1:
    # Reference file is a list.
    # --------------------------------------------------------

    if isinstance(data, list):

        for item in data:

            if not isinstance(item, dict):
                continue

            raw_id = (
                item.get("query_id")
                or item.get("id")
                or item.get("question_id")
            )

            if raw_id is None:
                continue

            qid = normalize_query_id(raw_id)

            question = (
                item.get("question")
                or item.get("query")
                or item.get("user_question")
            )

            if question:

                questions[qid] = item

    # --------------------------------------------------------
    # Case 2:
    # Reference file is a dictionary.
    # --------------------------------------------------------

    elif isinstance(data, dict):

        # Direct mapping:
        #
        # {
        #   "Q001": {...},
        #   "Q002": {...}
        # }

        for key, value in data.items():

            qid = normalize_query_id(key)

            if isinstance(value, dict):

                question = (
                    value.get("question")
                    or value.get("query")
                    or value.get("user_question")
                )

                if question:

                    questions[qid] = value

            elif isinstance(value, str):

                questions[qid] = {
                    "query_id": qid,
                    "question": value
                }

        # ----------------------------------------------------
        # Some datasets use:
        #
        # {
        #   "questions": [...]
        # }
        # ----------------------------------------------------

        if "questions" in data:

            question_list = data["questions"]

            if isinstance(question_list, list):

                for item in question_list:

                    if not isinstance(item, dict):
                        continue

                    raw_id = (
                        item.get("query_id")
                        or item.get("id")
                        or item.get("question_id")
                    )

                    if raw_id is None:
                        continue

                    qid = normalize_query_id(raw_id)

                    question = (
                        item.get("question")
                        or item.get("query")
                        or item.get("user_question")
                    )

                    if question:

                        questions[qid] = item

    if not questions:

        raise RuntimeError(
            "No evaluation questions could be loaded "
            f"from {REFERENCE_FILE}"
        )

    return questions


# ============================================================
# OUTPUT FILE DISCOVERY
# ============================================================

def find_existing_answer(
    query_id: str,
    answer_type: str
) -> Optional[Path]:
    """
    Find an existing valid answer file.

    answer_type:
        "base"
        "rag"
    """

    pattern = f"{answer_type}_answer_{query_id}_*.json"

    files = list(
        GENERATION_OUTPUT_DIR.glob(pattern)
    )

    if not files:
        return None

    # Newest first.
    files.sort(
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    for path in files:

        try:

            with open(
                path,
                "r",
                encoding="utf-8"
            ) as f:

                data = json.load(f)

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
# SAVE BASE ANSWER
# ============================================================

def save_base_answer(
    query_id: str,
    question: str,
    answer: str,
    provider: str
) -> Path:
    """
    Save one Base LLM answer.
    """

    timestamp = now_string()

    output_path = (
        GENERATION_OUTPUT_DIR /
        f"base_answer_{query_id}_{timestamp}.json"
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

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            indent=2,
            ensure_ascii=False
        )

    return output_path


# ============================================================
# SAVE RAG ANSWER
# ============================================================

def save_rag_answer(
    query_id: str,
    question: str,
    answer: str,
    provider: str,
    context_package: Dict[str, Any]
) -> Path:
    """
    Save one RAG answer.

    The evidence records are included so that the generated
    answer remains traceable to the retrieved judicial evidence.
    """

    timestamp = now_string()

    output_path = (
        GENERATION_OUTPUT_DIR /
        f"rag_answer_{query_id}_{timestamp}.json"
    )

    result = {

        "query_id": query_id,

        "question": question,

        "generation_type": "RAG",

        "provider": provider,

        "retrieval_used": True,

        "external_legal_context_used": True,

        "answer": answer,

        "evidence_records": context_package.get(
            "evidence_records",
            []
        ),

        "cases_considered": context_package.get(
            "cases_considered",
            []
        ),

        "total_context_characters": context_package.get(
            "total_context_characters",
            0
        ),

        "truncated": context_package.get(
            "truncated",
            False
        ),

        "generated_at": datetime.now().isoformat()
    }

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            indent=2,
            ensure_ascii=False
        )

    return output_path


# ============================================================
# BUILD BASE PROMPT
# ============================================================

def build_base_prompt(
    question: str
) -> str:
    """
    Prompt for the no-retrieval Base LLM experiment.
    """

    return f"""
You are answering an Indian legal research and education question.

Question:
{question}

Provide a clear and concise informational answer.

Important instructions:

1. Answer the question directly.
2. Use your general pretrained knowledge.
3. Do not claim that you searched the project's legal corpus.
4. Do not invent case names, statutes, sections or legal principles.
5. If you are uncertain about a legal detail, clearly state the uncertainty.
6. This is an academic/legal research system, not personalized legal advice.
7. Do not present yourself as a lawyer or court.
8. Keep the answer focused on the question.
""".strip()


# ============================================================
# BUILD RAG PROMPT
# ============================================================

def build_rag_prompt(
    question: str,
    llm_context: str
) -> str:
    """
    Prompt for evidence-grounded RAG generation.
    """

    return f"""
You are an Indian legal research and education assistant.

You must answer the user's question using ONLY the
judicial evidence supplied below.

USER QUESTION:
{question}

RETRIEVED JUDICIAL EVIDENCE:
{llm_context}

INSTRUCTIONS:

1. Answer the user's question directly and clearly.

2. Use only information supported by the retrieved judicial
   evidence.

3. Do NOT introduce legal facts, cases, statutes, sections,
   principles or dates from outside the supplied evidence.

4. Do NOT invent citations or case names.

5. When referring to a judgment, identify the case using the
   case information provided in the evidence.

6. Where page or paragraph information is available, preserve
   that provenance when explaining the supporting material.

7. If the retrieved evidence is insufficient to answer part
   of the question, explicitly say that the supplied evidence
   is insufficient for that part.

8. Do not fill missing information using unsupported assumptions.

9. Distinguish between what the retrieved judgments state and
   any limitations in the available evidence.

10. This system is intended for legal research and education.
    It is NOT a substitute for professional legal advice.

11. Do not make personalized legal recommendations.

12. Keep the final answer reasonably concise while covering
    the supported legal principles.
""".strip()


# ============================================================
# PROVIDER GENERATION
# ============================================================

def generate_with_provider_fallback(
    manager: ProviderManager,
    prompt: str
) -> Dict[str, str]:
    """
    Generate using ProviderManager.

    ProviderManager is responsible for:

        Gemini -> OpenAI -> Anthropic

    No retry loop exists here.

    If every provider fails, an exception is raised immediately.
    """

    result = manager.generate(
        prompt=prompt,
        primary_provider=PRIMARY_PROVIDER,
        use_fallback=USE_FALLBACK
    )

    # ProviderManager currently returns:
    #
    # {
    #     "provider": "...",
    #     "text": "..."
    # }

    if isinstance(result, dict):

        provider = result.get(
            "provider",
            "unknown"
        )

        text = result.get(
            "text",
            ""
        )

    else:

        # Defensive support in case ProviderManager
        # returns a plain string.

        provider = PRIMARY_PROVIDER
        text = str(result)

    if not text or not text.strip():

        raise RuntimeError(
            "LLM provider returned an empty answer."
        )

    return {
        "provider": provider,
        "text": text.strip()
    }


# ============================================================
# GENERATE BASE ANSWER
# ============================================================

def generate_base(
    query_id: str,
    question: str,
    manager: ProviderManager
) -> Path:
    """
    Generate and save the Base LLM answer.
    """

    existing = find_existing_answer(
        query_id,
        "base"
    )

    if existing:

        print(
            f"BASE already exists: "
            f"{existing.name}"
        )

        return existing

    print()
    print("-" * 70)
    print(f"BASE LLM GENERATION: {query_id}")
    print("-" * 70)

    prompt = build_base_prompt(
        question
    )

    result = generate_with_provider_fallback(
        manager,
        prompt
    )

    output_path = save_base_answer(
        query_id=query_id,
        question=question,
        answer=result["text"],
        provider=result["provider"]
    )

    print(
        f"BASE ANSWER SAVED: "
        f"{output_path.name}"
    )

    return output_path


# ============================================================
# GENERATE RAG ANSWER
# ============================================================

def generate_rag(
    query_id: str,
    question: str,
    manager: ProviderManager,
    bm25,
    bm25_metadata,
    faiss_index,
    faiss_metadata,
    embedding_model
) -> Path:
    """
    Generate and save the RAG answer.

    Steps:

        Query
          ↓
        Hybrid retrieval
          ↓
        Case aggregation
          ↓
        Authority reranking
          ↓
        Context construction
          ↓
        LLM generation
          ↓
        Save
    """

    existing = find_existing_answer(
        query_id,
        "rag"
    )

    if existing:

        print(
            f"RAG already exists: "
            f"{existing.name}"
        )

        return existing

    print()
    print("-" * 70)
    print(f"RAG GENERATION: {query_id}")
    print("-" * 70)

    # --------------------------------------------------------
    # STEP 1: RETRIEVAL
    # --------------------------------------------------------

    print()
    print("Running legal retrieval...")

    reranked_cases = search(
        question,
        bm25,
        bm25_metadata,
        faiss_index,
        faiss_metadata,
        embedding_model
    )

    if not reranked_cases:

        raise RuntimeError(
            f"No legal cases were retrieved for {query_id}."
        )

    print(
        f"Retrieved ranked cases: "
        f"{len(reranked_cases)}"
    )

    # --------------------------------------------------------
    # STEP 2: CONTEXT CONSTRUCTION
    # --------------------------------------------------------

    print()
    print("Building evidence context...")

    context_package = build_context_package(
        question,
        reranked_cases
    )

    evidence_records = context_package.get(
        "evidence_records",
        []
    )

    if not evidence_records:

        raise RuntimeError(
            f"No evidence records were created for {query_id}."
        )

    print(
        f"Evidence records: "
        f"{len(evidence_records)}"
    )

    print(
        f"Context characters: "
        f"{context_package.get('total_context_characters', 0)}"
    )

    # --------------------------------------------------------
    # STEP 3: SAVE CONTEXT
    # --------------------------------------------------------

    print()
    print("Saving context package...")

    context_path = save_context_package(
        context_package
    )

    print(
        f"Context saved: "
        f"{Path(context_path).name}"
    )

    # --------------------------------------------------------
    # STEP 4: CREATE LLM CONTEXT
    # --------------------------------------------------------

    llm_context = create_llm_context(
        context_package
    )

    if not llm_context or not llm_context.strip():

        raise RuntimeError(
            f"LLM context is empty for {query_id}."
        )

    # --------------------------------------------------------
    # STEP 5: RAG PROMPT
    # --------------------------------------------------------

    prompt = build_rag_prompt(
        question,
        llm_context
    )

    # --------------------------------------------------------
    # STEP 6: LLM GENERATION
    # --------------------------------------------------------

    print()
    print("Generating evidence-grounded answer...")

    result = generate_with_provider_fallback(
        manager,
        prompt
    )

    # --------------------------------------------------------
    # STEP 7: SAVE ANSWER
    # --------------------------------------------------------

    output_path = save_rag_answer(
        query_id=query_id,
        question=question,
        answer=result["text"],
        provider=result["provider"],
        context_package=context_package
    )

    print(
        f"RAG ANSWER SAVED: "
        f"{output_path.name}"
    )

    return output_path


# ============================================================
# SAVE FAILURE INFORMATION
# ============================================================

def save_failure(
    query_id: str,
    question: str,
    error: Exception
) -> Path:
    """
    Save failure information so that debugging is reproducible.
    """

    timestamp = now_string()

    path = (
        FAILURE_DIR /
        f"failure_{query_id}_{timestamp}.json"
    )

    result = {

        "query_id": query_id,

        "question": question,

        "error_type": type(error).__name__,

        "error": str(error),

        "failed_at": datetime.now().isoformat()
    }

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            indent=2,
            ensure_ascii=False
        )

    return path


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("CHECKPOINTED LEGAL RAG BATCH GENERATION")
    print("=" * 70)

    print()
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Start query  : {query_id_string(START_QUERY)}")
    print(f"End query    : {query_id_string(END_QUERY)}")
    print(f"Provider     : {PRIMARY_PROVIDER}")
    print(
        f"Fallback     : "
        f"{'Gemini -> OpenAI -> Anthropic' if USE_FALLBACK else 'Disabled'}"
    )

    # --------------------------------------------------------
    # LOAD QUESTIONS
    # --------------------------------------------------------

    print()
    print("Loading reference questions...")

    questions = load_reference_questions()

    print(
        f"Reference questions loaded: "
        f"{len(questions)}"
    )

    # --------------------------------------------------------
    # LOAD RETRIEVAL SYSTEM
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("LOADING RETRIEVAL SYSTEM")
    print("=" * 70)

    print()
    print("Loading BM25 index...")

    bm25, bm25_metadata = load_bm25()

    print(
        f"BM25 documents loaded: "
        f"{len(bm25_metadata)}"
    )

    print()
    print("Loading FAISS index...")

    faiss_index, faiss_metadata = load_faiss()

    print(
        f"FAISS vectors loaded: "
        f"{faiss_index.ntotal}"
    )

    print()
    print("Loading embedding model...")

    embedding_model = load_embedding_model()

    print(
        "Embedding model loaded successfully."
    )

    # --------------------------------------------------------
    # PROVIDER MANAGER
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("INITIALIZING LLM PROVIDERS")
    print("=" * 70)

    manager = ProviderManager()

    print()
    print("LLM provider manager initialized.")

    # --------------------------------------------------------
    # TRACKING
    # --------------------------------------------------------

    completed = []

    stopped_query = None

    # --------------------------------------------------------
    # PROCESS QUESTIONS
    # --------------------------------------------------------

    for number in range(
        START_QUERY,
        END_QUERY + 1
    ):

        query_id = query_id_string(
            number
        )

        # ----------------------------------------------------
        # CHECK QUESTION
        # ----------------------------------------------------

        if query_id not in questions:

            print()
            print("=" * 70)
            print(
                f"{query_id} NOT FOUND IN REFERENCE DATASET"
            )
            print("=" * 70)

            stopped_query = query_id

            break

        item = questions[query_id]

        question = (
            item.get("question")
            or item.get("query")
            or item.get("user_question")
        )

        if not question:

            print()
            print(
                f"{query_id} has no question text."
            )

            stopped_query = query_id

            break

        # ----------------------------------------------------
        # QUERY HEADER
        # ----------------------------------------------------

        print()
        print()
        print("=" * 70)
        print(f"PROCESSING {query_id}")
        print("=" * 70)

        print()
        print("Question:")
        print(question)

        # ----------------------------------------------------
        # GENERATE BASE + RAG
        # ----------------------------------------------------

        try:

            # =================================================
            # BASE
            # =================================================

            base_path = generate_base(
                query_id,
                question,
                manager
            )

            # =================================================
            # RAG
            # =================================================

            rag_path = generate_rag(
                query_id,
                question,
                manager,
                bm25,
                bm25_metadata,
                faiss_index,
                faiss_metadata,
                embedding_model
            )

            # =================================================
            # SUCCESS
            # =================================================

            completed.append(
                query_id
            )

            print()
            print("=" * 70)
            print(f"{query_id} COMPLETED SUCCESSFULLY")
            print("=" * 70)

            print(
                f"Base: {base_path.name}"
            )

            print(
                f"RAG : {rag_path.name}"
            )

        except Exception as error:

            # ------------------------------------------------
            # FAILURE
            # ------------------------------------------------

            stopped_query = query_id

            print()
            print("=" * 70)
            print(f"{query_id} FAILED")
            print("=" * 70)

            print()
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
                    f"Failure log saved: "
                    f"{failure_path.name}"
                )

            except Exception as save_error:

                print(
                    f"Could not save failure log: "
                    f"{save_error}"
                )

            # ------------------------------------------------
            # IMPORTANT:
            #
            # DO NOT RETRY THE QUESTION.
            # DO NOT WAIT.
            # DO NOT CONTINUE.
            #
            # Everything completed before this query
            # is already safely saved.
            # ------------------------------------------------

            print()
            print(
                "Saving completed results and stopping batch."
            )

            break

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print()
    print("=" * 70)
    print("BATCH GENERATION SUMMARY")
    print("=" * 70)

    print()
    print(
        f"Requested range: "
        f"{query_id_string(START_QUERY)} "
        f"to "
        f"{query_id_string(END_QUERY)}"
    )

    print(
        f"Completed queries: "
        f"{len(completed)}"
    )

    if completed:

        print()
        print("Completed:")

        print(
            ", ".join(completed)
        )

    if stopped_query:

        print()
        print(
            f"Stopped at: "
            f"{stopped_query}"
        )

        print()
        print(
            "All queries completed before the failure "
            "have been preserved."
        )

    else:

        print()
        print(
            "All requested queries completed."
        )

    print()
    print(
        f"Generation output directory:"
    )

    print(
        GENERATION_OUTPUT_DIR
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