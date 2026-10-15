import os
import sys
import re
from datetime import datetime
from collections import defaultdict


# ============================================================
# ADD SCRIPTS DIRECTORY TO PYTHON PATH
# ============================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)


# ============================================================
# IMPORT EXISTING RETRIEVAL FUNCTIONS
# ============================================================

from hybrid_search import (
    load_bm25,
    load_faiss,
    load_embedding_model,
    bm25_search,
    faiss_search,
    reciprocal_rank_fusion,
)

from case_level_search import (
    aggregate_by_case,
)


# ============================================================
# CONFIGURATION
# ============================================================

# Number of chunks retrieved from BM25
BM25_TOP_K = 20

# Number of chunks retrieved from FAISS
FAISS_TOP_K = 20

# Reciprocal Rank Fusion constant
RRF_K = 60

# Number of cases shown before/after reranking
FINAL_CASES = 5

# Number of supporting passages retained per case
SUPPORTING_CHUNKS = 3


# ============================================================
# RELEVANCE-FIRST FILTER
# ============================================================

# A candidate must achieve at least this fraction
# of the strongest case relevance score.

RELEVANCE_FLOOR_RATIO = 0.45


# ============================================================
# FINAL RERANKING WEIGHTS
# ============================================================

RELEVANCE_WEIGHT = 0.80
AUTHORITY_WEIGHT = 0.15
CITATION_WEIGHT = 0.03
TEMPORAL_WEIGHT = 0.02

# ============================================================
# COURT AUTHORITY SIGNALS
# ============================================================

COURT_AUTHORITY = {
    "supreme court": 1.00,
    "high court": 0.70,
    "district court": 0.40,
}


# ============================================================
# NORMALIZE COURT NAME
# ============================================================

def normalize_court(court):
    """
    Convert court names into a standard form.
    """

    if not court:
        return ""

    return court.strip().lower()


# ============================================================
# AUTHORITY SCORE
# ============================================================

def get_authority_score(court):
    """
    Return a bounded authority signal based on court level.

    This is a retrieval/ranking signal only.
    It is not a legal determination of binding authority.
    """

    normalized = normalize_court(court)

    return COURT_AUTHORITY.get(
        normalized,
        0.20
    )


# ============================================================
# EXTRACT DATE FROM CASE ID / SOURCE FILE
# ============================================================

def extract_case_date(case_id):
    """
    Extract judgment date from case identifiers such as:

        Case_Name_on_25_September_2024

    Returns:
        datetime object
        or None if a date cannot be extracted.
    """

    if not case_id:
        return None

    pattern = (
        r"_on_"
        r"(\d{1,2})_"
        r"([A-Za-z]+)_"
        r"(\d{4})"
        r"$"
    )

    match = re.search(
        pattern,
        case_id
    )

    if not match:
        return None

    day = match.group(1)
    month = match.group(2)
    year = match.group(3)

    date_text = (
        f"{day} "
        f"{month} "
        f"{year}"
    )

    try:

        return datetime.strptime(
            date_text,
            "%d %B %Y"
        )

    except ValueError:

        return None


# ============================================================
# CITATION SIGNAL
# ============================================================

def citation_signal(case):
    """
    Estimate how strongly the retrieved supporting passages
    contain legal case/statutory references.

    This is a simple baseline proxy.

    It does NOT represent a true citation count or
    legal precedential strength.
    """

    supporting_chunks = case.get(
        "supporting_chunks",
        []
    )

    combined_text = " ".join(
        chunk.get("text", "")
        for chunk in supporting_chunks
    )

    if not combined_text:
        return 0.0

    # --------------------------------------------------------
    # Patterns commonly appearing in legal references
    # --------------------------------------------------------

    patterns = [

        # Case-versus-case references
        r"\bv\.\s+[A-Z]",

        r"\bvs\.\s+[A-Z]",

        r"\bversus\b",

        # SCC citations
        r"\bSCC\b",

        # AIR citations
        r"\bAIR\b",

        # Criminal Law Journal
        r"\bCr\.?\s*LJ\b",

        # Criminal Law Reports / Crimes
        r"\bCrimes\b",

        # Law Commission references
        r"\bLaw Commission\b",

        # Constitutional articles
        r"\bArticle\s+\d+",

        # Statutory sections
        r"\bSection\s+\d+",
    ]

    count = 0

    for pattern in patterns:

        matches = re.findall(
            pattern,
            combined_text,
            flags=re.IGNORECASE
        )

        count += len(matches)

    # --------------------------------------------------------
    # Normalize the signal
    #
    # We cap the count so a case cannot dominate simply
    # because its retrieved passages contain many references.
    # --------------------------------------------------------

    MAX_CITATION_SIGNAL = 20

    normalized = min(
        count,
        MAX_CITATION_SIGNAL
    ) / MAX_CITATION_SIGNAL

    return normalized


# ============================================================
# TEMPORAL SIGNAL
# ============================================================

def calculate_temporal_scores(case_results):
    """
    Convert case dates into a normalized temporal signal.

    The newest retrieved case receives 1.0.

    The oldest retrieved case receives 0.0.

    If dates are unavailable, all cases receive 0.5.
    """

    dates = {}

    for case in case_results:

        date = extract_case_date(
            case.get("case_id")
        )

        dates[
            case["case_id"]
        ] = date

    valid_dates = [
        date
        for date in dates.values()
        if date is not None
    ]

    # --------------------------------------------------------
    # No usable dates
    # --------------------------------------------------------

    if not valid_dates:

        return {
            case["case_id"]: 0.5
            for case in case_results
        }

    # --------------------------------------------------------
    # Date range
    # --------------------------------------------------------

    oldest = min(
        valid_dates
    )

    newest = max(
        valid_dates
    )

    total_days = (
        newest - oldest
    ).days

    temporal_scores = {}

    for case in case_results:

        case_id = case[
            "case_id"
        ]

        date = dates[
            case_id
        ]

        if date is None:

            temporal_scores[
                case_id
            ] = 0.5

            continue

        # ----------------------------------------------------
        # All dates are identical
        # ----------------------------------------------------

        if total_days == 0:

            temporal_scores[
                case_id
            ] = 1.0

            continue

        # ----------------------------------------------------
        # Normalize date between 0 and 1
        # ----------------------------------------------------

        days_from_oldest = (
            date - oldest
        ).days

        score = (
            days_from_oldest
            / total_days
        )

        temporal_scores[
            case_id
        ] = score

    return temporal_scores


# ============================================================
# RELEVANCE-FIRST FILTER
# ============================================================

def relevance_first_filter(case_results):
    """
    Remove candidates that are substantially weaker than
    the strongest retrieved case.

    Threshold:

        best_case_score * RELEVANCE_FLOOR_RATIO
    """

    if not case_results:

        return []

    # --------------------------------------------------------
    # Find strongest relevance score
    # --------------------------------------------------------

    best_score = max(
        case["case_score"]
        for case in case_results
    )

    relevance_floor = (
        best_score
        * RELEVANCE_FLOOR_RATIO
    )

    # --------------------------------------------------------
    # Keep only sufficiently relevant cases
    # --------------------------------------------------------

    filtered = [

        case

        for case in case_results

        if case["case_score"]
        >= relevance_floor
    ]

    # --------------------------------------------------------
    # Safety fallback
    # --------------------------------------------------------

    if not filtered:

        filtered = [
            case_results[0]
        ]

    return filtered


# ============================================================
# AUTHORITY-AWARE RERANKING
# ============================================================

def authority_aware_rerank(case_results):
    """
    Apply relevance, authority, citation and temporal
    signals to already relevant cases.

    Final score:

        0.70 * relevance
        + 0.15 * authority
        + 0.10 * citation
        + 0.05 * temporal
    """

    if not case_results:

        return []

    # --------------------------------------------------------
    # Normalize relevance
    # --------------------------------------------------------

    maximum_relevance = max(
        case["case_score"]
        for case in case_results
    )

    if maximum_relevance <= 0:

        maximum_relevance = 1.0

    # --------------------------------------------------------
    # Temporal scores
    # --------------------------------------------------------

    temporal_scores = (
        calculate_temporal_scores(
            case_results
        )
    )

    reranked = []

    # --------------------------------------------------------
    # Calculate final score
    # --------------------------------------------------------

    for case in case_results:

        case_id = case[
            "case_id"
        ]

        # ----------------------------------------------------
        # Relevance
        # ----------------------------------------------------

        relevance = (
            case["case_score"]
            / maximum_relevance
        )

        relevance = min(
            max(relevance, 0.0),
            1.0
        )

        # ----------------------------------------------------
        # Authority
        # ----------------------------------------------------

        authority = get_authority_score(
            case.get("court")
        )

        # ----------------------------------------------------
        # Citation/reference signal
        # ----------------------------------------------------

        citation = citation_signal(
            case
        )

        # ----------------------------------------------------
        # Temporal signal
        # ----------------------------------------------------

        temporal = temporal_scores.get(
            case_id,
            0.5
        )

        # ----------------------------------------------------
        # Final score
        # ----------------------------------------------------

        final_score = (

            RELEVANCE_WEIGHT
            * relevance

            +

            AUTHORITY_WEIGHT
            * authority

            +

            CITATION_WEIGHT
            * citation

            +

            TEMPORAL_WEIGHT
            * temporal
        )

        # ----------------------------------------------------
        # Store individual signals
        # ----------------------------------------------------

        case_copy = dict(
            case
        )

        case_copy[
            "normalized_relevance"
        ] = relevance

        case_copy[
            "authority_score"
        ] = authority

        case_copy[
            "citation_score"
        ] = citation

        case_copy[
            "temporal_score"
        ] = temporal

        case_copy[
            "final_score"
        ] = final_score

        reranked.append(
            case_copy
        )

    # --------------------------------------------------------
    # Sort by final score
    # --------------------------------------------------------

    reranked.sort(
        key=lambda x: x[
            "final_score"
        ],
        reverse=True
    )

    return reranked


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(
    query,
    original_cases,
    filtered_cases,
    reranked_cases
):
    """
    Display relevance filtering and final authority-aware
    ranking results.
    """

    print("\n")

    print(
        "=" * 100
    )

    print(
        "AUTHORITY-AWARE CASE RETRIEVAL"
    )

    print(
        "=" * 100
    )

    print(
        f"\nQuery: {query}"
    )

    print(
        f"\nCases before relevance filtering: "
        f"{len(original_cases)}"
    )

    print(
        f"Cases after relevance filtering:  "
        f"{len(filtered_cases)}"
    )

    print(
        f"Final reranked cases: "
        f"{len(reranked_cases)}"
    )

    # --------------------------------------------------------
    # Display final cases
    # --------------------------------------------------------

    for rank, case in enumerate(
        reranked_cases[:FINAL_CASES],
        start=1
    ):

        print(
            "\n"
            + "-" * 100
        )

        print(
            f"FINAL RANK       : {rank}"
        )

        print(
            f"CASE ID          : "
            f"{case['case_id']}"
        )

        print(
            f"COURT            : "
            f"{case['court']}"
        )

        print(
            f"FINAL SCORE      : "
            f"{case['final_score']:.6f}"
        )

        print(
            f"RELEVANCE        : "
            f"{case['normalized_relevance']:.6f}"
        )

        print(
            f"AUTHORITY        : "
            f"{case['authority_score']:.6f}"
        )

        print(
            f"CITATION SIGNAL  : "
            f"{case['citation_score']:.6f}"
        )

        print(
            f"TEMPORAL SIGNAL  : "
            f"{case['temporal_score']:.6f}"
        )

        print(
            f"CASE RRF SCORE   : "
            f"{case['case_score']:.6f}"
        )

        print(
            f"CANDIDATE CHUNKS : "
            f"{case['candidate_chunk_count']}"
        )

        print(
            f"SOURCE FILE      : "
            f"{case['source_file']}"
        )

        print(
            "\nTOP SUPPORTING PASSAGES:"
        )

        # ----------------------------------------------------
        # Display supporting passages
        # ----------------------------------------------------

        for passage_rank, chunk in enumerate(
            case["supporting_chunks"],
            start=1
        ):

            print(
                "\n"
                + " " * 2
                + f"Passage {passage_rank}"
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
                        f"{start_page}-"
                        f"{end_page}"
                    )

            else:

                page_text = str(
                    chunk.get(
                        "pages",
                        []
                    )
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
            # Retrieval ranks
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

            text = text.replace(
                "\n",
                " "
            )

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
        + "=" * 100
    )


# ============================================================
# COMPLETE SEARCH PIPELINE
# ============================================================

def search(
    query,
    bm25,
    bm25_metadata,
    faiss_index,
    faiss_metadata,
    embedding_model,
):
    """
    Complete Step 10 pipeline:

        Query
          ↓
        BM25
          ↓
        FAISS
          ↓
        RRF
          ↓
        Case aggregation
          ↓
        Relevance-first filtering
          ↓
        Authority-aware reranking
          ↓
        Final cases
    """

    # --------------------------------------------------------
    # STEP 1: BM25
    # --------------------------------------------------------

    bm25_results = bm25_search(
        query,
        bm25,
        bm25_metadata,
        BM25_TOP_K
    )

    # --------------------------------------------------------
    # STEP 2: FAISS
    # --------------------------------------------------------

    faiss_results = faiss_search(
        query,
        faiss_index,
        faiss_metadata,
        embedding_model,
        FAISS_TOP_K
    )

    # --------------------------------------------------------
    # STEP 3: RRF
    # --------------------------------------------------------

    fused_results = reciprocal_rank_fusion(
        bm25_results,
        faiss_results,
        RRF_K
    )

    # --------------------------------------------------------
    # STEP 4: Case aggregation
    # --------------------------------------------------------

    case_results = aggregate_by_case(
        fused_results
    )

    # --------------------------------------------------------
    # STEP 5: Relevance-first filtering
    # --------------------------------------------------------

    filtered_cases = (
        relevance_first_filter(
            case_results
        )
    )

    # --------------------------------------------------------
    # STEP 6: Authority-aware reranking
    # --------------------------------------------------------

    reranked_cases = (
        authority_aware_rerank(
            filtered_cases
        )
    )

    return (
        case_results,
        filtered_cases,
        reranked_cases
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 100
    )

    print(
        "STEP 10 - RELEVANCE-FIRST "
        "+ AUTHORITY-AWARE LEGAL RETRIEVAL"
    )

    print(
        "=" * 100
    )

    # ========================================================
    # LOAD BM25
    # ========================================================

    bm25, bm25_metadata = load_bm25()

    print(
        f"BM25 documents: "
        f"{len(bm25_metadata)}"
    )

    # ========================================================
    # LOAD FAISS
    # ========================================================

    faiss_index, faiss_metadata = load_faiss()

    print(
        f"FAISS vectors: "
        f"{faiss_index.ntotal}"
    )

    # ========================================================
    # LOAD EMBEDDING MODEL
    # ========================================================

    embedding_model = (
        load_embedding_model()
    )

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

        return

    print(
        "\nIndex validation passed."
    )

    print(
        "\nAuthority-aware retrieval system ready."
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
                "\nExiting authority-aware search."
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

        # ----------------------------------------------------
        # SEARCH
        # ----------------------------------------------------

        print(
            "\nSearching..."
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
                "\nERROR DURING SEARCH:"
            )

            print(
                type(e).__name__,
                ":",
                e
            )

            continue

        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        display_results(
            query,
            original_cases,
            filtered_cases,
            reranked_cases
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()