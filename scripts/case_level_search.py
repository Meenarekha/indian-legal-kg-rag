import os
import sys
from collections import defaultdict


# ============================================================
# ADD SCRIPTS DIRECTORY TO PYTHON PATH
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)


# ============================================================
# IMPORT FUNCTIONS FROM HYBRID SEARCH
# ============================================================

from hybrid_search import (
    load_bm25,
    load_faiss,
    load_embedding_model,
    bm25_search,
    faiss_search,
    reciprocal_rank_fusion,
)


# ============================================================
# CONFIGURATION
# ============================================================

# Number of candidates retrieved from BM25
BM25_TOP_K = 20

# Number of candidates retrieved from FAISS
FAISS_TOP_K = 20

# Reciprocal Rank Fusion constant
RRF_K = 60

# Number of final cases to display
FINAL_CASES = 5

# Number of supporting passages displayed for each case
SUPPORTING_CHUNKS = 3


# ============================================================
# CASE-LEVEL AGGREGATION
# ============================================================

def aggregate_by_case(fused_results):
    """
    Convert chunk-level hybrid retrieval results
    into case-level ranked results.

    Each retrieved chunk contains a case_id.

    Chunks are grouped using case_id.

    Case score:

        Case Score =
            Maximum RRF score
            + 0.25 * Mean Top-3 RRF scores

    This avoids simply summing every chunk score,
    which could favor longer judgments.
    """

    # --------------------------------------------------------
    # Group retrieved chunks by case
    # --------------------------------------------------------

    grouped_cases = defaultdict(list)

    for result in fused_results:

        case_id = result.get("case_id")

        # Ignore results without a case ID
        if not case_id:
            continue

        grouped_cases[case_id].append(result)

    # --------------------------------------------------------
    # Calculate case-level scores
    # --------------------------------------------------------

    case_results = []

    for case_id, chunks in grouped_cases.items():

        # Sort chunks from strongest to weakest
        chunks = sorted(
            chunks,
            key=lambda x: x["rrf_score"],
            reverse=True
        )

        # Keep only the strongest supporting passages
        top_chunks = chunks[:SUPPORTING_CHUNKS]

        # ----------------------------------------------------
        # Strongest evidence from this case
        # ----------------------------------------------------

        max_rrf = top_chunks[0]["rrf_score"]

        # ----------------------------------------------------
        # Average score of top supporting passages
        # ----------------------------------------------------

        mean_top_rrf = sum(
            chunk["rrf_score"]
            for chunk in top_chunks
        ) / len(top_chunks)

        # ----------------------------------------------------
        # Final case-level score
        # ----------------------------------------------------

        case_score = (
            max_rrf
            + (0.25 * mean_top_rrf)
        )

        # Strongest retrieved passage
        best_chunk = top_chunks[0]

        # ----------------------------------------------------
        # Store case-level result
        # ----------------------------------------------------

        case_result = {

            "case_id": case_id,

            "court": best_chunk.get(
                "court"
            ),

            "source_file": best_chunk.get(
                "source_file"
            ),

            "case_score": case_score,

            "best_rrf": max_rrf,

            "candidate_chunk_count": len(
                chunks
            ),

            "supporting_chunks": top_chunks,
        }

        case_results.append(
            case_result
        )

    # --------------------------------------------------------
    # Sort cases by case score
    # --------------------------------------------------------

    case_results.sort(
        key=lambda x: x["case_score"],
        reverse=True
    )

    return case_results


# ============================================================
# DISPLAY CASE RESULTS
# ============================================================

def display_case_results(
    query,
    case_results
):
    """
    Display ranked cases and their strongest
    supporting passages.
    """

    print("\n")
    print("=" * 90)
    print("CASE-LEVEL SEARCH RESULTS")
    print("=" * 90)

    print(
        f"\nQuery: {query}"
    )

    print(
        f"Retrieved cases: "
        f"{len(case_results)}"
    )

    # --------------------------------------------------------
    # Display top cases
    # --------------------------------------------------------

    for rank, case in enumerate(
        case_results[:FINAL_CASES],
        start=1
    ):

        print("\n" + "-" * 90)

        print(
            f"CASE RANK      : {rank}"
        )

        print(
            f"CASE ID        : "
            f"{case['case_id']}"
        )

        print(
            f"COURT          : "
            f"{case['court']}"
        )

        print(
            f"CASE SCORE     : "
            f"{case['case_score']:.6f}"
        )

        print(
            f"BEST RRF       : "
            f"{case['best_rrf']:.6f}"
        )

        print(
            f"CANDIDATE "
            f"CHUNKS        : "
            f"{case['candidate_chunk_count']}"
        )

        print(
            f"SOURCE FILE    : "
            f"{case['source_file']}"
        )

        print(
            "\nTOP SUPPORTING PASSAGES:"
        )

        # ----------------------------------------------------
        # Display supporting passages
        # ----------------------------------------------------

        for chunk_rank, chunk in enumerate(
            case["supporting_chunks"],
            start=1
        ):

            print(
                "\n"
                + " " * 2
                + f"Passage {chunk_rank}"
            )

            print(
                " " * 4
                + "RRF Score : "
                + f"{chunk['rrf_score']:.6f}"
            )

            # ------------------------------------------------
            # Pages
            # ------------------------------------------------

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

                    page_text = str(
                        start_page
                    )

                else:

                    page_text = (
                        f"{start_page}-{end_page}"
                    )

            else:

                pages = chunk.get(
                    "pages",
                    []
                )

                page_text = str(
                    pages
                )

            print(
                " " * 4
                + "Pages     : "
                + page_text
            )

            # ------------------------------------------------
            # Paragraph
            # ------------------------------------------------

            paragraph_start = chunk.get(
                "paragraph_start"
            )

            paragraph_end = chunk.get(
                "paragraph_end"
            )

            if (
                paragraph_start is not None
                and paragraph_end is not None
            ):

                if (
                    paragraph_start
                    == paragraph_end
                ):

                    paragraph_text = str(
                        paragraph_start
                    )

                else:

                    paragraph_text = (
                        f"{paragraph_start}-"
                        f"{paragraph_end}"
                    )

            else:

                paragraph_text = "N/A"

            print(
                " " * 4
                + "Paragraph : "
                + paragraph_text
            )

            # ------------------------------------------------
            # BM25 information
            # ------------------------------------------------

            print(
                " " * 4
                + "BM25 Rank : "
                + str(
                    chunk.get(
                        "bm25_rank"
                    )
                )
            )

            # ------------------------------------------------
            # FAISS information
            # ------------------------------------------------

            print(
                " " * 4
                + "FAISS Rank: "
                + str(
                    chunk.get(
                        "faiss_rank"
                    )
                )
            )

            # ------------------------------------------------
            # Text
            # ------------------------------------------------

            print(
                " " * 4
                + "Text:"
            )

            text = chunk.get(
                "text",
                ""
            )

            # Remove line breaks for terminal display
            text = text.replace(
                "\n",
                " "
            )

            # Keep terminal output readable
            if len(text) > 500:

                text = (
                    text[:500]
                    + "..."
                )

            print(
                " " * 8
                + text
            )

    print(
        "\n"
        + "=" * 90
    )


# ============================================================
# CASE-LEVEL SEARCH PIPELINE
# ============================================================

def search_cases(
    query,
    bm25,
    bm25_metadata,
    faiss_index,
    faiss_metadata,
    embedding_model,
):
    """
    Complete retrieval pipeline:

        User Query
             |
             v
          BM25
             |
             +----------------+
                              |
          FAISS              |
             |                |
             +-------+--------+
                     |
                     v
                    RRF
                     |
                     v
              Ranked Chunks
                     |
                     v
             Case Aggregation
                     |
                     v
               Ranked Cases
                     |
                     v
          Supporting Passages
    """

    # --------------------------------------------------------
    # STEP 1: BM25 retrieval
    # --------------------------------------------------------

    bm25_results = bm25_search(
        query,
        bm25,
        bm25_metadata,
        BM25_TOP_K
    )

    # --------------------------------------------------------
    # STEP 2: FAISS semantic retrieval
    # --------------------------------------------------------

    faiss_results = faiss_search(
        query,
        faiss_index,
        faiss_metadata,
        embedding_model,
        FAISS_TOP_K
    )

    # --------------------------------------------------------
    # STEP 3: Reciprocal Rank Fusion
    # --------------------------------------------------------
    #
    # IMPORTANT:
    #
    # The existing hybrid_search.py defines:
    #
    # reciprocal_rank_fusion(
    #     bm25_results,
    #     faiss_results,
    #     rrf_k=60
    # )
    #
    # Therefore RRF_K is passed positionally here.
    # --------------------------------------------------------

    fused_results = reciprocal_rank_fusion(
        bm25_results,
        faiss_results,
        RRF_K
    )

    # --------------------------------------------------------
    # STEP 4: Case-level aggregation
    # --------------------------------------------------------

    case_results = aggregate_by_case(
        fused_results
    )

    return case_results


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print(
        "=" * 90
    )

    print(
        "CASE-LEVEL LEGAL SEARCH"
    )

    print(
        "=" * 90
    )

    # ========================================================
    # LOAD BM25
    # ========================================================

    # load_bm25() already prints its own loading message.
    bm25, bm25_metadata = load_bm25()

    print(
        f"BM25 documents: "
        f"{len(bm25_metadata)}"
    )

    # ========================================================
    # LOAD FAISS
    # ========================================================

    # load_faiss() already prints its own loading message.
    faiss_index, faiss_metadata = load_faiss()

    print(
        f"FAISS vectors: "
        f"{faiss_index.ntotal}"
    )

    # ========================================================
    # LOAD EMBEDDING MODEL
    # ========================================================

    # load_embedding_model() already prints its own message.
    embedding_model = load_embedding_model()

    print(
        "Embedding model loaded."
    )

    # ========================================================
    # VALIDATE BM25
    # ========================================================

    if (
        len(bm25_metadata)
        != bm25.corpus_size
    ):

        print(
            "\nERROR:"
        )

        print(
            "BM25 metadata count does not "
            "match BM25 corpus size."
        )

        print(
            f"Metadata: "
            f"{len(bm25_metadata)}"
        )

        print(
            f"Corpus: "
            f"{bm25.corpus_size}"
        )

        return

    # ========================================================
    # VALIDATE FAISS
    # ========================================================

    if (
        len(faiss_metadata)
        != faiss_index.ntotal
    ):

        print(
            "\nERROR:"
        )

        print(
            "FAISS metadata count does not "
            "match FAISS index size."
        )

        print(
            f"Metadata: "
            f"{len(faiss_metadata)}"
        )

        print(
            f"Index vectors: "
            f"{faiss_index.ntotal}"
        )

        return

    # ========================================================
    # VALIDATION SUCCESS
    # ========================================================

    print(
        "\nIndex validation passed."
    )

    print(
        "\nCase-level retrieval system ready."
    )

    print(
        "\nEnter a legal query."
    )

    print(
        "Type 'exit' to stop."
    )

    # ========================================================
    # INTERACTIVE SEARCH LOOP
    # ========================================================

    while True:

        print(
            "\n"
            + "=" * 90
        )

        query = input(
            "Enter legal query "
            "(or type 'exit' to quit): "
        ).strip()

        # ----------------------------------------------------
        # Exit
        # ----------------------------------------------------

        if query.lower() == "exit":

            print(
                "\nExiting case-level search."
            )

            break

        # ----------------------------------------------------
        # Empty query
        # ----------------------------------------------------

        if not query:

            print(
                "Please enter a legal query."
            )

            continue

        # ----------------------------------------------------
        # Search
        # ----------------------------------------------------

        print(
            "\nSearching..."
        )

        try:

            case_results = search_cases(
                query,
                bm25,
                bm25_metadata,
                faiss_index,
                faiss_metadata,
                embedding_model,
            )

        except Exception as e:

            print(
                "\nERROR DURING SEARCH:"
            )

            print(
                type(e).__name__,
                ":",
                e
            )

            continue

        # ----------------------------------------------------
        # Display results
        # ----------------------------------------------------

        display_case_results(
            query,
            case_results
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()