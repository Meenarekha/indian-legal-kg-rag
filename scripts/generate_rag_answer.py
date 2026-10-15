import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")


CONTEXT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "contexts"
)


OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "generation_outputs"
)


CONTEXT_BUILDER = (
    PROJECT_ROOT
    / "scripts"
    / "context_builder.py"
)


# ============================================================
# TEST QUESTION
# ============================================================

QUESTION = (
    "What are the conditions for granting anticipatory bail "
    "under Section 438 CrPC?"
)


# ============================================================
# BUILD LLM PROMPT
# ============================================================

def build_prompt(question, legal_context):
    """
    Build the prompt sent to the LLM.

    The retrieved judicial evidence is the primary source
    for the generated answer.
    """

    return f"""
You are a legal research assistant for an academic
Indian legal information retrieval system.

Answer the user's question using ONLY the judicial evidence
provided below.

IMPORTANT RULES:

1. Use the provided judicial evidence as the primary and
   controlling source for your answer.

2. Do NOT invent or introduce:
   - facts
   - cases
   - statutes
   - constitutional provisions
   - legal principles
   - citations
   - quotations
   - case outcomes
   - legal tests

3. Every important legal claim must be supported by the
   provided judicial evidence.

4. Do NOT expand or interpret an Article, Section, Act,
   constitutional provision, or legal term unless that
   interpretation is explicitly supported by the provided
   evidence.

5. If the evidence mentions an Article or Section but does
   not explain its meaning, do not guess its meaning.
   State only what the evidence establishes.

6. Do NOT use general knowledge to fill missing legal
   information.

7. If the provided evidence is insufficient to answer any
   part of the question, explicitly say:

   "The retrieved judicial evidence is insufficient to
   establish this point."

8. Mention case names only when they appear in the
   provided evidence.

9. Do not create citations or case references that are not
   present in the provided evidence.

10. Do not present assumptions or inferences as established
    law.

11. Keep the answer concise, clear, and structured.

12. Where useful, organize the answer as:
    - Direct answer
    - Legal basis / principle
    - Supporting cases
    - Evidence limitation

13. Do not provide personalized legal advice.

14. Do not claim that the answer represents the complete
    law of India. The answer is based only on the retrieved
    judicial evidence.

15. IMPORTANT ACCURACY RULE:

    If there is any conflict between your general knowledge
    and the retrieved judicial evidence, do NOT use your
    general knowledge to override the evidence.

    Follow the retrieved evidence and clearly mention any
    limitation.

USER QUESTION:

{question}

JUDICIAL EVIDENCE:

============================================================

{legal_context}

============================================================

Provide the answer based strictly on the judicial evidence above.
"""


# ============================================================
# CREATE SAFE FILE NAME
# ============================================================

def make_safe_filename(text, prefix, extension=".json"):
    """
    Convert arbitrary user text into a Windows-safe filename.

    Windows does not allow characters such as angle brackets,
    colon, quotes, slash, backslash, asterisk, pipe,
    question mark, and similar special characters.

    This function replaces every non-alphanumeric character
    with an underscore.
    """

    safe_text = "".join(
        character
        if character.isalnum()
        else "_"
        for character in text.lower()
    )

    while "__" in safe_text:
        safe_text = safe_text.replace(
            "__",
            "_"
        )

    safe_text = safe_text.strip("_")

    if not safe_text:
        safe_text = "legal_query"

    safe_text = safe_text[:80]

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    return (
        f"{prefix}_{safe_text}_{timestamp}{extension}"
    )


# ============================================================
# RUN CONTEXT BUILDER
# ============================================================

def run_context_builder(question):
    """
    Run context_builder.py and return the NEW context file.

    IMPORTANT:

    This function deliberately does NOT fall back to an old
    context file.

    An old context file could belong to a completely different
    legal query and would therefore cause incorrect RAG answers.
    """

    if not CONTEXT_BUILDER.exists():

        raise FileNotFoundError(
            f"Context builder not found:\n"
            f"{CONTEXT_BUILDER}"
        )


    CONTEXT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # Remember existing context files
    # --------------------------------------------------------

    before = {
        path.resolve()
        for path in CONTEXT_DIR.glob("*.json")
    }


    print(
        "=" * 70
    )

    print(
        "STEP 1: BUILDING LEGAL CONTEXT"
    )

    print(
        "=" * 70
    )


    # --------------------------------------------------------
    # Run context builder
    # --------------------------------------------------------

    process = subprocess.run(
        [
            sys.executable,
            str(CONTEXT_BUILDER)
        ],
        input=f"{question}\nexit\n",
        text=True,
        capture_output=True,
        cwd=str(PROJECT_ROOT)
    )


    # --------------------------------------------------------
    # Display context-builder output
    # --------------------------------------------------------

    if process.stdout:

        print(
            process.stdout
        )


    if process.stderr:

        print(
            process.stderr
        )


    # --------------------------------------------------------
    # Check process status
    # --------------------------------------------------------

    if process.returncode != 0:

        raise RuntimeError(
            "Context builder failed."
        )


    # --------------------------------------------------------
    # Find files after context construction
    # --------------------------------------------------------

    after = {
        path.resolve()
        for path in CONTEXT_DIR.glob("*.json")
    }


    new_files = after - before


    # --------------------------------------------------------
    # IMPORTANT:
    # A new context file MUST exist.
    # --------------------------------------------------------

    if not new_files:

        raise FileNotFoundError(
            "Context builder completed, but no new "
            "context JSON file was created."
        )


    # --------------------------------------------------------
    # Select newest newly-created file
    # --------------------------------------------------------

    new_context_file = max(
        new_files,
        key=lambda path: path.stat().st_mtime
    )


    # --------------------------------------------------------
    # Verify the file actually contains the requested query
    # --------------------------------------------------------

    try:

        with open(
            new_context_file,
            "r",
            encoding="utf-8"
        ) as file:

            package = json.load(file)


    except Exception as e:

        raise RuntimeError(
            "The newly-created context file could not "
            f"be loaded:\n{e}"
        )


    saved_query = package.get(
        "query"
    )


    if saved_query != question:

        raise ValueError(
            "Context query mismatch detected.\n\n"
            f"Requested query:\n{question}\n\n"
            f"Saved context query:\n{saved_query}\n\n"
            "The system will NOT use this context because "
            "it may belong to another query."
        )


    # --------------------------------------------------------
    # Validate evidence
    # --------------------------------------------------------

    evidence_records = package.get(
        "evidence_records",
        []
    )


    if not evidence_records:

        raise ValueError(
            "The newly-created context file contains "
            "no evidence records."
        )


    print(
        "\nVerified NEW context file:"
    )

    print(
        new_context_file
    )

    print(
        f"Evidence records: "
        f"{len(evidence_records)}"
    )


    return new_context_file


# ============================================================
# BUILD LLM-READABLE CONTEXT
# ============================================================

def build_legal_context(package):
    """
    Convert the structured context package into a readable
    evidence block for the LLM.
    """

    evidence_records = package.get(
        "evidence_records",
        []
    )


    if not evidence_records:

        raise ValueError(
            "The context package contains no evidence records."
        )


    sections = []


    for index, record in enumerate(
        evidence_records,
        start=1
    ):

        case_id = record.get(
            "case_id",
            "Unknown"
        )


        court = record.get(
            "court",
            "Unknown"
        )


        source_file = record.get(
            "source_file",
            "Unknown"
        )


        pages = record.get(
            "pages",
            "N/A"
        )


        paragraphs = record.get(
            "paragraphs",
            "N/A"
        )


        case_rank = record.get(
            "case_rank",
            "N/A"
        )


        passage_rank = record.get(
            "passage_rank",
            "N/A"
        )


        rrf_score = record.get(
            "rrf_score",
            "N/A"
        )


        text = record.get(
            "text",
            ""
        )


        if not text.strip():

            continue


        section = f"""
[EVIDENCE {index}]

Case ID:
{case_id}

Court:
{court}

Source File:
{source_file}

Pages:
{pages}

Paragraphs:
{paragraphs}

Case Rank:
{case_rank}

Passage Rank:
{passage_rank}

RRF Score:
{rrf_score}

Judicial Text:
{text}
"""


        sections.append(
            section
        )


    if not sections:

        raise ValueError(
            "The context package contains no usable "
            "judicial evidence."
        )


    return "\n".join(
        sections
    )


# ============================================================
# LOAD CONTEXT PACKAGE
# ============================================================

def load_context(context_file):
    """
    Load a context JSON file and convert its evidence into
    LLM-readable context.
    """

    context_file = Path(
        context_file
    )


    if not context_file.exists():

        raise FileNotFoundError(
            f"Context file not found:\n"
            f"{context_file}"
        )


    print(
        "=" * 70
    )

    print(
        "STEP 2: LOADING LEGAL EVIDENCE"
    )

    print(
        "=" * 70
    )


    print(
        f"Context file:\n"
        f"{context_file}"
    )


    with open(
        context_file,
        "r",
        encoding="utf-8"
    ) as file:

        package = json.load(
            file
        )


    # --------------------------------------------------------
    # Validate package
    # --------------------------------------------------------

    if not isinstance(
        package,
        dict
    ):

        raise ValueError(
            "Context JSON must contain an object."
        )


    if "query" not in package:

        raise ValueError(
            "Context package does not contain a query."
        )


    legal_context = build_legal_context(
        package
    )


    return (
        package,
        legal_context
    )


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    question,
    legal_context
):
    """
    Generate a RAG answer using Gemini as the primary provider.
    """

    print(
        "=" * 70
    )

    print(
        "STEP 3: GENERATING RAG ANSWER"
    )

    print(
        "=" * 70
    )


    scripts_dir = (
        PROJECT_ROOT
        / "scripts"
    )


    if str(scripts_dir) not in sys.path:

        sys.path.insert(
            0,
            str(scripts_dir)
        )


    from llm.provider_manager import (
        ProviderManager
    )


    manager = ProviderManager()


    prompt = build_prompt(
        question,
        legal_context
    )


    print(
        "\nProvider: Gemini"
    )

    print(
        "Retrieval context: PROVIDED"
    )


    answer = manager.generate(
        prompt=prompt,
        primary_provider="gemini",
        use_fallback=True
    )


    if not answer:

        raise RuntimeError(
            "The LLM returned an empty answer."
        )


    return answer


# ============================================================
# SAVE RESULT
# ============================================================

def save_result(
    question,
    context_file,
    package,
    answer
):
    """
    Save the generated RAG answer and retrieval metadata.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    output_filename = make_safe_filename(
        question,
        "rag_answer"
    )


    output_file = (
        OUTPUT_DIR
        / output_filename
    )


    # --------------------------------------------------------
    # Context path relative to project root
    # --------------------------------------------------------

    try:

        relative_context_path = str(
            Path(context_file).resolve().relative_to(
                PROJECT_ROOT.resolve()
            )
        )


    except ValueError:

        relative_context_path = str(
            context_file
        )


    # --------------------------------------------------------
    # Result object
    # --------------------------------------------------------

    result = {

        "query_id": "INTERACTIVE",

        "question": question,

        "generation_type": "RAG",

        "provider": "gemini",

        "context_file":
            relative_context_path,

        "retrieval_summary": {

            "cases_considered":
                package.get(
                    "cases_considered"
                ),

            "evidence_records":
                len(
                    package.get(
                        "evidence_records",
                        []
                    )
                ),

            "context_characters":
                package.get(
                    "total_context_characters"
                ),

            "context_truncated":
                package.get(
                    "truncated"
                )
        },

        "answer":
            answer,

        "generated_at":
            datetime.now().isoformat()
    }


    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            result,
            file,
            indent=4,
            ensure_ascii=False
        )


    return output_file


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n"
    )


    print(
        "=" * 70
    )

    print(
        "INDIAN LEGAL RAG — FIRST GENERATION TEST"
    )

    print(
        "=" * 70
    )


    print(
        "\nQuestion:"
    )

    print(
        QUESTION
    )


    try:

        # ----------------------------------------------------
        # 1. Retrieve and build evidence
        # ----------------------------------------------------

        context_file = run_context_builder(
            QUESTION
        )


        print(
            "\nContext generated successfully:"
        )

        print(
            context_file
        )


        # ----------------------------------------------------
        # 2. Load evidence
        # ----------------------------------------------------

        package, legal_context = (
            load_context(
                context_file
            )
        )


        print(
            f"\nEvidence records loaded: "
            f"{len(package['evidence_records'])}"
        )


        print(
            f"Cases considered: "
            f"{package.get('cases_considered')}"
        )


        print(
            f"Context characters: "
            f"{package.get('total_context_characters')}"
        )


        # ----------------------------------------------------
        # 3. Generate answer
        # ----------------------------------------------------

        answer = generate_answer(
            QUESTION,
            legal_context
        )


        # ----------------------------------------------------
        # 4. Save answer
        # ----------------------------------------------------

        output_file = save_result(
            QUESTION,
            context_file,
            package,
            answer
        )


        # ----------------------------------------------------
        # 5. Display answer
        # ----------------------------------------------------

        print(
            "\n"
        )


        print(
            "=" * 70
        )

        print(
            "FINAL RAG ANSWER"
        )

        print(
            "=" * 70
        )


        print(
            "\n"
        )


        print(
            answer
        )


        print(
            "\n"
        )


        print(
            "=" * 70
        )

        print(
            "RAG GENERATION SUCCESSFUL"
        )

        print(
            "=" * 70
        )


        print(
            f"\nSaved result:\n"
            f"{output_file}"
        )


    except Exception as e:

        print(
            "\n"
        )


        print(
            "=" * 70
        )

        print(
            "RAG GENERATION FAILED"
        )

        print(
            "=" * 70
        )


        print(
            f"\nError type: "
            f"{type(e).__name__}"
        )


        print(
            f"Error: {e}"
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()