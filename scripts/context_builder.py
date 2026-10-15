import os
import sys
import json
from datetime import datetime


# ============================================================
# ADD SCRIPTS DIRECTORY TO PYTHON PATH
# ============================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)


# ============================================================
# IMPORT EXISTING RETRIEVAL PIPELINE
# ============================================================

from authority_rerank import (
    load_bm25,
    load_faiss,
    load_embedding_model,
    search,
)


# ============================================================
# CONFIGURATION
# ============================================================

# Maximum number of cases sent to the context builder
MAX_CASES = 5

# Maximum supporting passages per case
SUPPORTING_CHUNKS_PER_CASE = 3

# Maximum total context size
MAX_CONTEXT_CHARACTERS = 30000

# Output directory
OUTPUT_DIR = os.path.join(
    os.path.dirname(SCRIPT_DIR),
    "data",
    "processed",
    "contexts"
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# SAFE TEXT CLEANING
# ============================================================

def clean_context_text(text):
    """
    Prepare retrieved text for LLM context.

    This does NOT modify the original dataset.
    It only normalizes whitespace for the generated
    context package.
    """

    if not text:
        return ""

    text = text.replace(
        "\r\n",
        "\n"
    )

    text = text.replace(
        "\r",
        "\n"
    )

    # Collapse repeated spaces
    text = " ".join(
        text.split()
    )

    return text.strip()


# ============================================================
# FORMAT PAGE INFORMATION
# ============================================================

def format_pages(chunk):
    """
    Convert page metadata into a human-readable form.
    """

    start_page = chunk.get(
        "start_page"
    )

    end_page = chunk.get(
        "end_page"
    )

    if (
        start_page is not None
        and end_page is not None
    ):

        if start_page == end_page:

            return str(
                start_page
            )

        return (
            f"{start_page}-{end_page}"
        )

    pages = chunk.get(
        "pages",
        []
    )

    if isinstance(
        pages,
        list
    ):

        return ", ".join(
            str(page)
            for page in pages
        )

    return str(
        pages
    )


# ============================================================
# FORMAT PARAGRAPH INFORMATION
# ============================================================

def format_paragraphs(chunk):
    """
    Convert paragraph metadata into a readable form.
    """

    start = chunk.get(
        "paragraph_start"
    )

    end = chunk.get(
        "paragraph_end"
    )

    if (
        start is not None
        and end is not None
    ):

        if start == end:

            return str(
                start
            )

        return (
            f"{start}-{end}"
        )

    return "N/A"


# ============================================================
# BUILD EVIDENCE RECORD
# ============================================================

def build_evidence_record(
    case_rank,
    passage_rank,
    case,
    chunk,
):
    """
    Build one structured evidence record.

    Every evidence record retains provenance.
    """

    text = clean_context_text(
        chunk.get(
            "text",
            ""
        )
    )

    return {

        "case_rank": case_rank,

        "passage_rank": passage_rank,

        "case_id": case.get(
            "case_id"
        ),

        "court": case.get(
            "court"
        ),

        "source_file": case.get(
            "source_file"
        ),

        "case_score": case.get(
            "case_score"
        ),

        "final_score": case.get(
            "final_score"
        ),

        "relevance_score": case.get(
            "normalized_relevance"
        ),

        "authority_score": case.get(
            "authority_score"
        ),

        "citation_score": case.get(
            "citation_score"
        ),

        "temporal_score": case.get(
            "temporal_score"
        ),

        "rrf_score": chunk.get(
            "rrf_score"
        ),

        "bm25_rank": chunk.get(
            "bm25_rank"
        ),

        "faiss_rank": chunk.get(
            "faiss_rank"
        ),

        "pages": format_pages(
            chunk
        ),

        "paragraphs": format_paragraphs(
            chunk
        ),

        "text": text,
    }


# ============================================================
# BUILD CONTEXT PACKAGE
# ============================================================

def build_context_package(
    query,
    reranked_cases,
):
    """
    Construct the final evidence package.

    Structure:

        Query
          ↓
        Ranked cases
          ↓
        Supporting passages
          ↓
        Provenance-preserving context
    """

    selected_cases = (
        reranked_cases[
            :MAX_CASES
        ]
    )

    context_records = []

    total_characters = 0

    # ========================================================
    # PROCESS CASES
    # ========================================================

    for case_rank, case in enumerate(
        selected_cases,
        start=1
    ):

        supporting_chunks = (
            case.get(
                "supporting_chunks",
                []
            )
        )

        supporting_chunks = (
            supporting_chunks[
                :SUPPORTING_CHUNKS_PER_CASE
            ]
        )

        for passage_rank, chunk in enumerate(
            supporting_chunks,
            start=1
        ):

            evidence = (
                build_evidence_record(
                    case_rank,
                    passage_rank,
                    case,
                    chunk,
                )
            )

            text_length = len(
                evidence["text"]
            )

            # ------------------------------------------------
            # Context size check
            # ------------------------------------------------

            if (
                total_characters
                + text_length
                > MAX_CONTEXT_CHARACTERS
            ):

                return {

                    "query": query,

                    "created_at": (
                        datetime.now()
                        .isoformat()
                    ),

                    "configuration": {

                        "max_cases":
                            MAX_CASES,

                        "supporting_chunks_per_case":
                            SUPPORTING_CHUNKS_PER_CASE,

                        "max_context_characters":
                            MAX_CONTEXT_CHARACTERS,
                    },

                    "cases_considered":
                        len(selected_cases),

                    "evidence_records":
                        context_records,

                    "total_context_characters":
                        total_characters,

                    "truncated":
                        True,
                }

            context_records.append(
                evidence
            )

            total_characters += (
                text_length
            )

    # ========================================================
    # RETURN COMPLETE PACKAGE
    # ========================================================

    return {

        "query": query,

        "created_at": (
            datetime.now()
            .isoformat()
        ),

        "configuration": {

            "max_cases":
                MAX_CASES,

            "supporting_chunks_per_case":
                SUPPORTING_CHUNKS_PER_CASE,

            "max_context_characters":
                MAX_CONTEXT_CHARACTERS,
        },

        "cases_considered":
            len(selected_cases),

        "evidence_records":
            context_records,

        "total_context_characters":
            total_characters,

        "truncated":
            False,
    }


# ============================================================
# CONVERT PACKAGE TO LLM CONTEXT
# ============================================================

def create_llm_context(
    context_package
):
    """
    Convert structured evidence into a readable context
    that can later be supplied to Gemini.

    The function explicitly labels each source and its
    provenance.
    """

    lines = []

    lines.append(
        "LEGAL EVIDENCE CONTEXT"
    )

    lines.append(
        "=" * 80
    )

    lines.append(
        f"USER QUERY: "
        f"{context_package['query']}"
    )

    lines.append(
        "=" * 80
    )

    lines.append("")

    # ========================================================
    # EVIDENCE
    # ========================================================

    for record in context_package[
        "evidence_records"
    ]:

        lines.append(
            f"[CASE {record['case_rank']}]"
        )

        lines.append(
            f"Case ID: "
            f"{record['case_id']}"
        )

        lines.append(
            f"Court: "
            f"{record['court']}"
        )

        lines.append(
            f"Source File: "
            f"{record['source_file']}"
        )

        lines.append(
            f"Pages: "
            f"{record['pages']}"
        )

        lines.append(
            f"Paragraphs: "
            f"{record['paragraphs']}"
        )

        lines.append(
            f"Case Rank: "
            f"{record['case_rank']}"
        )

        lines.append(
            f"Passage Rank: "
            f"{record['passage_rank']}"
        )

        lines.append(
            f"RRF Score: "
            f"{record['rrf_score']}"
        )

        lines.append(
            "Evidence:"
        )

        lines.append(
            record["text"]
        )

        lines.append(
            "-" * 80
        )

    return "\n".join(
        lines
    )


# ============================================================
# SAVE CONTEXT PACKAGE
# ============================================================

def save_context_package(
    package
):
    """
    Save structured evidence as JSON.

    The query is sanitized before being used in the filename
    so that Windows-invalid characters cannot cause a failure.
    """

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    # --------------------------------------------------------
    # Get original query
    # --------------------------------------------------------

    raw_query = package.get(
        "query",
        "legal_query"
    )

    # --------------------------------------------------------
    # Make query Windows-safe
    # --------------------------------------------------------

    safe_query = "".join(
        character
        if character.isalnum()
        else "_"
        for character in raw_query.lower()
    )

    # Remove repeated underscores
    while "__" in safe_query:
        safe_query = safe_query.replace(
            "__",
            "_"
        )

    # Remove underscores from beginning/end
    safe_query = safe_query.strip(
        "_"
    )

    # Keep filename manageable
    safe_query = safe_query[:60]

    # Fallback if query contains no usable characters
    if not safe_query:
        safe_query = "legal_query"

    # --------------------------------------------------------
    # Create final filename
    # --------------------------------------------------------

    filename = (
        f"context_"
        f"{safe_query}_"
        f"{timestamp}.json"
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            package,
            file,
            ensure_ascii=False,
            indent=2
        )

    return output_path


# ============================================================
# DISPLAY CONTEXT
# ============================================================

def display_context(
    context_package,
    llm_context,
):
    """
    Display a concise summary and the actual LLM context.
    """

    print("\n")

    print(
        "=" * 100
    )

    print(
        "CONTEXT CONSTRUCTION RESULTS"
    )

    print(
        "=" * 100
    )

    print(
        f"\nQuery: "
        f"{context_package['query']}"
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
        f"Truncated: "
        f"{context_package['truncated']}"
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "LLM CONTEXT"
    )

    print(
        "=" * 100
    )

    print(
        llm_context
    )

    print(
        "\n"
        + "=" * 100
    )


# ============================================================
# LOAD RETRIEVAL COMPONENTS
# ============================================================

def load_retrieval_system():

    print(
        "=" * 100
    )

    print(
        "STEP 11 - CONTEXT CONSTRUCTION "
        "AND EVIDENCE PACKAGING"
    )

    print(
        "=" * 100
    )

    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

    bm25, bm25_metadata = load_bm25()

    print(
        f"BM25 documents: "
        f"{len(bm25_metadata)}"
    )

    # --------------------------------------------------------
    # FAISS
    # --------------------------------------------------------

    faiss_index, faiss_metadata = load_faiss()

    print(
        f"FAISS vectors: "
        f"{faiss_index.ntotal}"
    )

    # --------------------------------------------------------
    # Embedding model
    # --------------------------------------------------------

    embedding_model = (
        load_embedding_model()
    )

    print(
        "Embedding model loaded."
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if (
        len(bm25_metadata)
        != bm25.corpus_size
    ):

        raise ValueError(
            "BM25 metadata count does not "
            "match BM25 corpus size."
        )

    if (
        len(faiss_metadata)
        != faiss_index.ntotal
    ):

        raise ValueError(
            "FAISS metadata count does not "
            "match FAISS index size."
        )

    print(
        "\nIndex validation passed."
    )

    return (
        bm25,
        bm25_metadata,
        faiss_index,
        faiss_metadata,
        embedding_model,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    try:

        (
            bm25,
            bm25_metadata,
            faiss_index,
            faiss_metadata,
            embedding_model,
        ) = load_retrieval_system()

    except Exception as e:

        print(
            "\nERROR WHILE LOADING SYSTEM:"
        )

        print(
            type(e).__name__,
            ":",
            e
        )

        return

    print(
        "\nContext construction system ready."
    )

    # ========================================================
    # INTERACTIVE LOOP
    # ========================================================

    while True:

        print(
            "\n"
            + "=" * 100
        )

        query = input(
            "Enter legal query "
            "(or type 'exit' to quit): "
        ).strip()

        # ----------------------------------------------------
        # EXIT
        # ----------------------------------------------------

        if query.lower() == "exit":

            print(
                "\nExiting context builder."
            )

            break

        # ----------------------------------------------------
        # EMPTY QUERY
        # ----------------------------------------------------

        if not query:

            print(
                "Please enter a legal query."
            )

            continue

        # ====================================================
        # SEARCH
        # ====================================================

        print(
            "\nRetrieving and ranking legal cases..."
        )

        try:

            (
                original_cases,
                filtered_cases,
                reranked_cases
            ) = search(
                query,
                bm25,
                bm25_metadata,
                faiss_index,
                faiss_metadata,
                embedding_model,
            )

        except Exception as e:

            print(
                "\nERROR DURING RETRIEVAL:"
            )

            print(
                type(e).__name__,
                ":",
                e
            )

            continue

        # ====================================================
        # BUILD CONTEXT
        # ====================================================

        print(
            "Building evidence context..."
        )

        context_package = (
            build_context_package(
                query,
                reranked_cases
            )
        )

        # ====================================================
        # CREATE LLM CONTEXT
        # ====================================================

        llm_context = (
            create_llm_context(
                context_package
            )
        )

        # ====================================================
        # SAVE JSON
        # ====================================================

        try:

            output_path = (
                save_context_package(
                    context_package
                )
            )

            print(
                f"\nContext package saved to:"
            )

            print(
                output_path
            )

        except Exception as e:

            print(
                "\nWARNING:"
            )

            print(
                "Could not save context package."
            )

            print(
                type(e).__name__,
                ":",
                e
            )

        # ====================================================
        # DISPLAY
        # ====================================================

        display_context(
            context_package,
            llm_context
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()