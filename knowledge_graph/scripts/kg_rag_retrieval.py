import json
import os
import sys
from pathlib import Path


# ============================================================
# PROJECT PATH
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = SCRIPT_DIR.parent.parent

EXISTING_SCRIPTS_DIR = PROJECT_ROOT / "scripts"

KG_PATH = (
    PROJECT_ROOT
    / "knowledge_graph"
    / "data"
    / "knowledge_graph.json"
)


# ============================================================
# MAKE EXISTING SCRIPTS IMPORTABLE
# ============================================================

if str(EXISTING_SCRIPTS_DIR) not in sys.path:

    sys.path.insert(
        0,
        str(EXISTING_SCRIPTS_DIR)
    )


# ============================================================
# IMPORT EXISTING RAG
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

from authority_rerank import (
    relevance_first_filter,
    authority_aware_rerank,
)


# ============================================================
# CONFIGURATION
# ============================================================

BM25_TOP_K = 20

FAISS_TOP_K = 20

RRF_K = 60

FINAL_CASES = 5


# ============================================================
# LOAD KNOWLEDGE GRAPH
# ============================================================

def load_knowledge_graph():

    print(
        "\nLoading Knowledge Graph..."
    )

    if not KG_PATH.exists():

        raise FileNotFoundError(
            f"Knowledge Graph not found:\n{KG_PATH}"
        )

    with open(
        KG_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        graph = json.load(f)

    nodes = graph.get(
        "nodes",
        []
    )

    relationships = graph.get(
        "relationships",
        []
    )

    print(
        f"KG nodes         : {len(nodes)}"
    )

    print(
        f"KG relationships  : {len(relationships)}"
    )

    return graph


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text):

    if not text:

        return ""

    return (
        str(text)
        .lower()
        .strip()
    )


# ============================================================
# EXTRACT EXPLICIT SECTION / ARTICLE REFERENCES
# ============================================================

def extract_structured_entities(query):

    query_lower = normalize_text(
        query
    )

    entities = []

    # --------------------------------------------------------
    # Section
    # --------------------------------------------------------

    import re

    section_matches = re.findall(
        r"\bsection\s+(\d+[a-z]?)\b",
        query_lower,
        re.IGNORECASE
    )

    for number in section_matches:

        entities.append(
            f"section_{number.lower()}"
        )

    # --------------------------------------------------------
    # Article
    # --------------------------------------------------------

    article_matches = re.findall(
        r"\barticle\s+(\d+[a-z]?)\b",
        query_lower,
        re.IGNORECASE
    )

    for number in article_matches:

        entities.append(
            f"article_{number.lower()}"
        )

    return list(
        dict.fromkeys(
            entities
        )
    )


# ============================================================
# KG ENTITY MATCHING
# ============================================================

def match_kg_entities(
    query,
    graph
):

    nodes = graph.get(
        "nodes",
        []
    )

    entity_tokens = (
        extract_structured_entities(
            query
        )
    )

    matched = []

    for node in nodes:

        node_id = normalize_text(
            node.get(
                "node_id",
                ""
            )
        )

        properties = node.get(
            "properties",
            {}
        )

        section_name = normalize_text(
            properties.get(
                "section_name",
                ""
            )
        )

        for token in entity_tokens:

            # ------------------------------------------------
            # Section entity
            # ------------------------------------------------

            if (
                node_id
                == f"section_{token}"
            ):

                matched.append(
                    node
                )

                continue

            # ------------------------------------------------
            # Article entity
            # ------------------------------------------------

            if (
                node_id
                == f"section_{token}"
            ):

                matched.append(
                    node
                )

                continue

            # ------------------------------------------------
            # Property fallback
            # ------------------------------------------------

            if (
                section_name
                == token.replace(
                    "_",
                    " "
                )
            ):

                matched.append(
                    node
                )

    return matched


# ============================================================
# GET CASE NODES
# ============================================================

def get_case_nodes(
    graph
):

    cases = {}

    for node in graph.get(
        "nodes",
        []
    ):

        if node.get(
            "node_type"
        ) != "Case":

            continue

        node_id = node.get(
            "node_id"
        )

        properties = node.get(
            "properties",
            {}
        )

        # ----------------------------------------------------
        # Only use original corpus judgments
        # ----------------------------------------------------

        if properties.get(
            "reference_only",
            False
        ):

            continue

        cases[node_id] = node

    return cases


# ============================================================
# KG CASE RETRIEVAL
# ============================================================

def kg_case_search(
    query,
    graph
):

    matched_entities = (
        match_kg_entities(
            query,
            graph
        )
    )

    if not matched_entities:

        return (
            matched_entities,
            []
        )

    entity_ids = {
        node.get(
            "node_id"
        )
        for node in matched_entities
    }

    relationships = graph.get(
        "relationships",
        []
    )

    corpus_cases = (
        get_case_nodes(
            graph
        )
    )

    case_results = {}

    # --------------------------------------------------------
    # Traverse relationships
    # --------------------------------------------------------

    for relationship in relationships:

        relationship_type = (
            relationship.get(
                "relationship_type"
            )
        )

        if (
            relationship_type
            != "INVOLVES_SECTION"
        ):

            continue

        source = relationship.get(
            "source"
        )

        target = relationship.get(
            "target"
        )

        # ----------------------------------------------------
        # Case -> Section
        # ----------------------------------------------------

        if (
            source in corpus_cases
            and target in entity_ids
        ):

            case_id = source

        # ----------------------------------------------------
        # Section -> Case
        # ----------------------------------------------------

        elif (
            target in corpus_cases
            and source in entity_ids
        ):

            case_id = target

        else:

            continue

        # ----------------------------------------------------
        # Create result
        # ----------------------------------------------------

        if case_id not in case_results:

            case_results[case_id] = {

                "case_id": case_id,

                "kg_score": 0,

                "kg_reason":
                    "Connected through "
                    "INVOLVES_SECTION",

            }

        case_results[
            case_id
        ][
            "kg_score"
        ] += 1

    return (
        matched_entities,
        list(
            case_results.values()
        )
    )


# ============================================================
# MERGE KG WITH EXISTING RAG
# ============================================================

def merge_results(
    rag_cases,
    kg_cases
):

    kg_lookup = {
        case["case_id"]: case
        for case in kg_cases
    }

    merged = []

    for case in rag_cases:

        case_copy = dict(
            case
        )

        case_id = case.get(
            "case_id"
        )

        if case_id in kg_lookup:

            case_copy[
                "kg_match"
            ] = True

            case_copy[
                "kg_score"
            ] = kg_lookup[
                case_id
            ][
                "kg_score"
            ]

            case_copy[
                "kg_reason"
            ] = kg_lookup[
                case_id
            ][
                "kg_reason"
            ]

        else:

            case_copy[
                "kg_match"
            ] = False

            case_copy[
                "kg_score"
            ] = 0

            case_copy[
                "kg_reason"
            ] = None

        merged.append(
            case_copy
        )

    return merged


# ============================================================
# COMPLETE KG + RAG SEARCH
# ============================================================

def search(
    query,
    bm25,
    bm25_metadata,
    faiss_index,
    faiss_metadata,
    embedding_model,
    graph
):

    # ========================================================
    # STEP 1
    # EXISTING BM25
    # ========================================================

    bm25_results = bm25_search(
        query,
        bm25,
        bm25_metadata,
        BM25_TOP_K
    )

    # ========================================================
    # STEP 2
    # EXISTING FAISS
    # ========================================================

    faiss_results = faiss_search(
        query,
        faiss_index,
        faiss_metadata,
        embedding_model,
        FAISS_TOP_K
    )

    # ========================================================
    # STEP 3
    # EXISTING RRF
    # ========================================================

    fused_results = (
        reciprocal_rank_fusion(
            bm25_results,
            faiss_results,
            RRF_K
        )
    )

    # ========================================================
    # STEP 4
    # EXISTING CASE AGGREGATION
    # ========================================================

    case_results = (
        aggregate_by_case(
            fused_results
        )
    )

    # ========================================================
    # STEP 5
    # EXISTING RELEVANCE FILTER
    # ========================================================

    filtered_cases = (
        relevance_first_filter(
            case_results
        )
    )

    # ========================================================
    # STEP 6
    # EXISTING AUTHORITY RANKING
    # ========================================================

    reranked_cases = (
        authority_aware_rerank(
            filtered_cases
        )
    )

    # ========================================================
    # STEP 7
    # KG SEARCH
    # ========================================================

    (
        matched_entities,
        kg_cases
    ) = kg_case_search(
        query,
        graph
    )

    # ========================================================
    # STEP 8
    # MERGE KG SIGNAL
    # ========================================================

    final_cases = merge_results(
        reranked_cases,
        kg_cases
    )

    # ========================================================
    # STEP 9
    # PRESERVE EXISTING RANKING
    # ========================================================

    # We deliberately do NOT change the existing ranking
    # score in this first integration.
    #
    # KG is attached as an additional structured signal.
    #
    # This allows us to verify:
    #
    #     RAG result
    #     +
    #     KG relationship
    #
    # before introducing a new learned/weighted ranking.

    return (
        case_results,
        filtered_cases,
        reranked_cases,
        matched_entities,
        kg_cases,
        final_cases,
    )


# ============================================================
# DISPLAY
# ============================================================

def display_results(
    query,
    matched_entities,
    kg_cases,
    final_cases
):

    print(
        "\n"
        + "=" * 100
    )

    print(
        "KG + RAG LEGAL RETRIEVAL"
    )

    print(
        "=" * 100
    )

    print(
        f"\nQuery: {query}"
    )

    # --------------------------------------------------------
    # KG entities
    # --------------------------------------------------------

    print(
        "\n"
        + "-" * 100
    )

    print(
        "KG STRUCTURED ENTITIES"
    )

    print(
        "-" * 100
    )

    if not matched_entities:

        print(
            "No explicit Section/Article "
            "entity detected."
        )

    else:

        for node in matched_entities:

            print(
                f"\nEntity     : "
                f"{node.get('node_type')}"
            )

            print(
                f"Node ID    : "
                f"{node.get('node_id')}"
            )

            properties = node.get(
                "properties",
                {}
            )

            print(
                f"Name       : "
                f"{properties.get('section_name')}"
            )

    # --------------------------------------------------------
    # KG candidates
    # --------------------------------------------------------

    print(
        "\n"
        + "-" * 100
    )

    print(
        "KG CANDIDATE CASES"
    )

    print(
        "-" * 100
    )

    if not kg_cases:

        print(
            "No KG candidate cases."
        )

    else:

        for rank, case in enumerate(
            kg_cases[:10],
            start=1
        ):

            print(
                f"\nKG Rank    : {rank}"
            )

            print(
                f"Case ID    : "
                f"{case['case_id']}"
            )

            print(
                f"KG Score   : "
                f"{case['kg_score']}"
            )

            print(
                f"Reason     : "
                f"{case['kg_reason']}"
            )

    # --------------------------------------------------------
    # Final RAG results + KG signal
    # --------------------------------------------------------

    print(
        "\n"
        + "-" * 100
    )

    print(
        "FINAL RAG CASES + KG SIGNAL"
    )

    print(
        "-" * 100
    )

    if not final_cases:

        print(
            "No RAG cases returned."
        )

        return

    for rank, case in enumerate(
        final_cases[:FINAL_CASES],
        start=1
    ):

        print(
            f"\nCase Rank        : {rank}"
        )

        print(
            f"Case ID          : "
            f"{case.get('case_id')}"
        )

        print(
            f"Court            : "
            f"{case.get('court')}"
        )

        print(
            f"RAG Final Score   : "
            f"{case.get('final_score')}"
        )

        print(
            f"KG Match         : "
            f"{case.get('kg_match')}"
        )

        print(
            f"KG Score         : "
            f"{case.get('kg_score')}"
        )

        if case.get(
            "kg_reason"
        ):

            print(
                f"KG Reason        : "
                f"{case.get('kg_reason')}"
            )

        print(
            f"Supporting chunks: "
            f"{len(case.get('supporting_chunks', []))}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 100
    )

    print(
        "KG + EXISTING RAG INTEGRATION"
    )

    print(
        "BM25 + FAISS + RRF + Case Ranking + Knowledge Graph"
    )

    print(
        "=" * 100
    )

    # --------------------------------------------------------
    # Load existing RAG components
    # --------------------------------------------------------

    print(
        "\nLoading existing RAG..."
    )

    bm25, bm25_metadata = (
        load_bm25()
    )

    faiss_index, faiss_metadata = (
        load_faiss()
    )

    embedding_model = (
        load_embedding_model()
    )

    # --------------------------------------------------------
    # Validate indexes
    # --------------------------------------------------------

    if (
        len(bm25_metadata)
        != bm25.corpus_size
    ):

        raise RuntimeError(
            "BM25 metadata/index mismatch."
        )

    if (
        len(faiss_metadata)
        != faiss_index.ntotal
    ):

        raise RuntimeError(
            "FAISS metadata/index mismatch."
        )

    print(
        "\nExisting RAG index validation: PASS"
    )

    # --------------------------------------------------------
    # Load KG
    # --------------------------------------------------------

    graph = (
        load_knowledge_graph()
    )

    # --------------------------------------------------------
    # Interactive search
    # --------------------------------------------------------

    print(
        "\nKG + RAG system ready."
    )

    print(
        "Type 'exit' to stop."
    )

    while True:

        query = input(
            "\nEnter legal query: "
        ).strip()

        if query.lower() == "exit":

            print(
                "\nExiting KG + RAG."
            )

            break

        if not query:

            print(
                "Please enter a legal query."
            )

            continue

        try:

            (
                case_results,
                filtered_cases,
                reranked_cases,
                matched_entities,
                kg_cases,
                final_cases,
            ) = search(
                query,
                bm25,
                bm25_metadata,
                faiss_index,
                faiss_metadata,
                embedding_model,
                graph,
            )

            display_results(
                query,
                matched_entities,
                kg_cases,
                final_cases,
            )

        except Exception as error:

            print(
                "\nERROR:"
            )

            print(
                str(error)
            )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()