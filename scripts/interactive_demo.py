import sys
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


# ============================================================
# IMPORT EXISTING WORKING FUNCTIONS
# ============================================================

from generate_base_answer import (
    generate_answer as generate_base_answer
)

from generate_rag_answer import (
    run_context_builder,
    load_context,
    generate_answer as generate_rag_answer,
)


# ============================================================
# DISPLAY FUNCTIONS
# ============================================================

def print_header():

    print("\n")
    print("=" * 80)
    print("             INDIAN LEGAL RAG — INTERACTIVE DEMO")
    print("=" * 80)

    print()
    print("System:")
    print("  Base LLM        : Gemini without legal retrieval")
    print("  RAG             : BM25 + FAISS + RRF + case ranking")
    print("  Evidence        : Indian judicial corpus")
    print()

    print("Enter a legal question to compare:")
    print("  1. Base LLM answer")
    print("  2. RAG answer using retrieved Indian court evidence")
    print()

    print("Type 'exit' to quit.")
    print("=" * 80)


def print_section(title):

    print("\n")
    print("-" * 80)
    print(title)
    print("-" * 80)


# ============================================================
# RUN BASE LLM
# ============================================================

def run_base_llm(question):

    print_section("BASE LLM — NO RETRIEVAL")

    print("Question:")
    print(question)

    print("\nGenerating answer using Gemini...")
    print("External legal context: NONE")
    print("Legal corpus retrieval: NONE")

    answer = generate_base_answer(question)

    print("\n")
    print("BASE LLM ANSWER")
    print("-" * 80)
    print(answer)

    return answer


# ============================================================
# RUN RAG
# ============================================================

def run_rag(question):

    print_section("RAG PIPELINE")

    print("Question:")
    print(question)

    # --------------------------------------------------------
    # 1. Retrieve and build legal context
    # --------------------------------------------------------

    print("\n[1] RETRIEVAL AND CONTEXT CONSTRUCTION")

    print("-" * 80)

    print("Retrieving relevant judicial evidence...")

    context_file = run_context_builder(question)

    print("\nContext generated successfully:")
    print(context_file)

    # --------------------------------------------------------
    # 2. Load evidence
    # --------------------------------------------------------

    print("\n[2] LOADING RETRIEVED LEGAL EVIDENCE")

    print("-" * 80)

    package, legal_context = load_context(context_file)

    evidence_records = package.get(
        "evidence_records",
        []
    )

    cases_considered = package.get(
        "cases_considered",
        "N/A"
    )

    context_characters = package.get(
        "total_context_characters",
        "N/A"
    )

    truncated = package.get(
        "truncated",
        False
    )

    print(
        f"Evidence records loaded : {len(evidence_records)}"
    )

    print(
        f"Cases considered        : {cases_considered}"
    )

    print(
        f"Context characters      : {context_characters}"
    )

    print(
        f"Context truncated       : {truncated}"
    )

    # --------------------------------------------------------
    # 3. Display supporting cases BEFORE generation
    # --------------------------------------------------------

    print("\n[3] RETRIEVED SUPPORTING CASES")

    print("-" * 80)

    shown_cases = set()

    for record in evidence_records:

        case_id = record.get(
            "case_id",
            "Unknown"
        )

        court = record.get(
            "court",
            "Unknown"
        )

        if case_id not in shown_cases:

            shown_cases.add(case_id)

            print(
                f"\nCase {len(shown_cases)}:"
            )

            print(
                f"  Case ID : {case_id}"
            )

            print(
                f"  Court   : {court}"
            )

    # --------------------------------------------------------
    # 4. Generate RAG answer
    # --------------------------------------------------------

    print("\n[4] GENERATING RAG ANSWER")

    print("-" * 80)

    print("Generating answer using Gemini...")
    print("Retrieval context: PROVIDED")

    answer = generate_rag_answer(
        question,
        legal_context
    )

    # --------------------------------------------------------
    # 5. Display answer
    # --------------------------------------------------------

    print("\n")
    print("RAG ANSWER")
    print("-" * 80)
    print(answer)

    # --------------------------------------------------------
    # 6. Display evidence summary
    # --------------------------------------------------------

    print("\n")
    print("SUPPORTING JUDICIAL EVIDENCE")
    print("-" * 80)

    shown_cases = set()

    for record in evidence_records:

        case_id = record.get(
            "case_id",
            "Unknown"
        )

        court = record.get(
            "court",
            "Unknown"
        )

        if case_id not in shown_cases:

            shown_cases.add(case_id)

            print(
                f"\nCase: {case_id}"
            )

            print(
                f"Court: {court}"
            )

    return answer


# ============================================================
# DISPLAY FINAL COMPARISON
# ============================================================

def display_comparison(
    base_answer,
    rag_answer
):

    print("\n")
    print("=" * 80)
    print("                         FINAL COMPARISON")
    print("=" * 80)

    print()

    if base_answer is not None:

        print("BASE LLM")
        print("-" * 80)

        print(
            "Generates an answer using the LLM's internal "
            "knowledge without retrieving evidence from "
            "the legal corpus."
        )

    else:

        print("BASE LLM")
        print("-" * 80)
        print("Generation failed.")


    print()

    if rag_answer is not None:

        print("RAG SYSTEM")
        print("-" * 80)

        print(
            "Retrieves relevant Indian judicial evidence "
            "before generating the answer."
        )

        print(
            "The answer is grounded in the retrieved "
            "judicial context."
        )

    else:

        print("RAG SYSTEM")
        print("-" * 80)
        print("Generation failed.")


    print()
    print("=" * 80)


# ============================================================
# MAIN INTERACTIVE DEMO
# ============================================================

def main():

    print_header()

    while True:

        print("\n")

        question = input(
            "Enter your legal query: "
        ).strip()


        # ----------------------------------------------------
        # EXIT
        # ----------------------------------------------------

        if question.lower() == "exit":

            print(
                "\nExiting Indian Legal RAG demo."
            )

            break


        # ----------------------------------------------------
        # EMPTY INPUT
        # ----------------------------------------------------

        if not question:

            print(
                "\nPlease enter a legal question."
            )

            continue


        # ----------------------------------------------------
        # RUN BASE LLM
        # ----------------------------------------------------

        try:

            base_answer = run_base_llm(
                question
            )

        except Exception as e:

            print("\n")
            print("BASE LLM FAILED")
            print("-" * 80)

            print(
                f"Error type: {type(e).__name__}"
            )

            print(
                f"Error: {e}"
            )

            base_answer = None


        # ----------------------------------------------------
        # RUN RAG
        # ----------------------------------------------------

        try:

            rag_answer = run_rag(
                question
            )

        except Exception as e:

            print("\n")
            print("RAG PIPELINE FAILED")
            print("-" * 80)

            print(
                f"Error type: {type(e).__name__}"
            )

            print(
                f"Error: {e}"
            )

            rag_answer = None


        # ----------------------------------------------------
        # FINAL COMPARISON
        # ----------------------------------------------------

        display_comparison(
            base_answer,
            rag_answer
        )


        # ----------------------------------------------------
        # DEMO STATUS
        # ----------------------------------------------------

        print("\n")
        print("=" * 80)
        print("                         DEMO COMPLETE")
        print("=" * 80)

        if base_answer is not None:

            print(
                "\nBase LLM: SUCCESS"
            )

        else:

            print(
                "\nBase LLM: FAILED"
            )


        if rag_answer is not None:

            print(
                "RAG: SUCCESS"
            )

        else:

            print(
                "RAG: FAILED"
            )

        print("=" * 80)


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()