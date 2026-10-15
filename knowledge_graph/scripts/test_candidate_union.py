import json
import os
import sys
import re


# ============================================================
# PROJECT PATHS
# ============================================================

# Current file:
# C:\Legal Query RAG\knowledge_graph\scripts\test_candidate_union.py

KG_SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# C:\Legal Query RAG
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(KG_SCRIPT_DIR)
)

# C:\Legal Query RAG\scripts
CORE_SCRIPT_DIR = os.path.join(
    PROJECT_ROOT,
    "scripts"
)


# ============================================================
# ADD REQUIRED DIRECTORIES TO PYTHON PATH
# ============================================================

if KG_SCRIPT_DIR not in sys.path:

    sys.path.insert(
        0,
        KG_SCRIPT_DIR
    )


if CORE_SCRIPT_DIR not in sys.path:

    sys.path.insert(
        0,
        CORE_SCRIPT_DIR
    )


# ============================================================
# EXISTING RAG IMPORTS
# ============================================================

from authority_rerank import (
    load_bm25,
    load_faiss,
    load_embedding_model,
    search,
)


# ============================================================
# KNOWLEDGE GRAPH FILE
# ============================================================

KG_FILE = os.path.join(
    PROJECT_ROOT,
    "knowledge_graph",
    "data",
    "nodes",
    "all_nodes.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

QUERY = (
    "What cases involve Section 438?"
)

RAG_TOP_CASES = 5


# ============================================================
# LOAD KNOWLEDGE GRAPH
# ============================================================

def load_knowledge_graph():

    if not os.path.exists(KG_FILE):

        raise FileNotFoundError(
            "\nKnowledge graph file not found:\n"
            f"{KG_FILE}"
        )

    with open(
        KG_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    # --------------------------------------------------------
    # Dictionary format
    # --------------------------------------------------------

    if isinstance(data, dict):

        nodes = data.get(
            "nodes",
            []
        )

        relationships = data.get(
            "relationships",
            []
        )

    # --------------------------------------------------------
    # List format
    # --------------------------------------------------------

    elif isinstance(data, list):

        nodes = data
        relationships = []

    else:

        raise ValueError(
            "Unsupported knowledge graph format."
        )

    return nodes, relationships


# ============================================================
# NORMALIZE TEXT
# ============================================================

def normalize_text(text):

    if not text:

        return ""

    text = str(text).lower()

    text = re.sub(
        r"[^a-z0-9]+",
        "_",
        text
    )

    return text.strip("_")


# ============================================================
# EXTRACT STRUCTURED ENTITIES FROM QUERY
# ============================================================

def extract_structured_entities(query):

    entities = []

    # --------------------------------------------------------
    # SECTION
    # --------------------------------------------------------

    section_matches = re.findall(
        r"\bsection\s+(\d+[a-z]?)\b",
        query,
        re.IGNORECASE
    )

    for number in section_matches:

        section_name = (
            f"Section {number}"
        )

        entities.append({

            "node_type": "Section",

            "entity_name": section_name,

            "normalized_name":
                normalize_text(
                    section_name
                )
        })

    # --------------------------------------------------------
    # ARTICLE
    # --------------------------------------------------------

    article_matches = re.findall(
        r"\barticle\s+(\d+[a-z]?)\b",
        query,
        re.IGNORECASE
    )

    for number in article_matches:

        article_name = (
            f"Article {number}"
        )

        entities.append({

            "node_type": "Section",

            "entity_name": article_name,

            "normalized_name":
                normalize_text(
                    article_name
                )
        })

    return entities


# ============================================================
# MATCH QUERY ENTITY WITH KG NODE
# ============================================================

def match_kg_entities(
    query_entities,
    nodes
):

    matched_nodes = []

    for query_entity in query_entities:

        target_name = query_entity[
            "normalized_name"
        ]

        for node in nodes:

            # ------------------------------------------------
            # We only search Section nodes here
            # ------------------------------------------------

            if node.get(
                "node_type"
            ) != "Section":

                continue

            properties = node.get(
                "properties",
                {}
            )

            section_name = properties.get(
                "section_name",
                ""
            )

            normalized_section_name = (
                normalize_text(
                    section_name
                )
            )

            normalized_node_id = (
                normalize_text(
                    node.get(
                        "node_id",
                        ""
                    )
                )
            )

            # ------------------------------------------------
            # Match using section name
            # ------------------------------------------------

            if (
                normalized_section_name
                == target_name
            ):

                matched_nodes.append(
                    node
                )

                continue

            # ------------------------------------------------
            # Match using node ID
            #
            # Example:
            #
            # Query:
            # Section 438
            #
            # Normalized:
            # section_438
            #
            # Actual KG ID:
            # section_section_438
            # ------------------------------------------------

            expected_node_id = (
                f"section_{target_name}"
            )

            if (
                normalized_node_id
                == expected_node_id
            ):

                matched_nodes.append(
                    node
                )

    # --------------------------------------------------------
    # Remove duplicate nodes
    # --------------------------------------------------------

    unique_nodes = {}

    for node in matched_nodes:

        key = (
            node.get("node_type"),
            node.get("node_id")
        )

        unique_nodes[key] = node

    return list(
        unique_nodes.values()
    )


# ============================================================
# FIND ACTUAL CORPUS CASES
# ============================================================

def find_connected_cases(
    matched_nodes,
    relationships,
    nodes
):

    matched_node_ids = {

        node.get("node_id")

        for node in matched_nodes

    }

    # --------------------------------------------------------
    # Build node lookup
    # --------------------------------------------------------

    node_lookup = {}

    for node in nodes:

        key = (
            node.get("node_type"),
            node.get("node_id")
        )

        node_lookup[key] = node

    # --------------------------------------------------------
    # Traverse:
    #
    # Case
    #   |
    #   | INVOLVES_SECTION
    #   ↓
    # Section
    # --------------------------------------------------------

    case_ids = set()

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
        # Target must be our matched section
        # ----------------------------------------------------

        if target not in matched_node_ids:

            continue

        # ----------------------------------------------------
        # Source must be a Case node
        # ----------------------------------------------------

        source_node = node_lookup.get(
            (
                "Case",
                source
            )
        )

        if source_node is None:

            continue

        properties = source_node.get(
            "properties",
            {}
        )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Ignore reference-only cases.
        #
        # These are cases mentioned by another judgment,
        # but are not part of our 205-case corpus.
        # ----------------------------------------------------

        if properties.get(
            "reference_only",
            False
        ):

            continue

        case_ids.add(
            source
        )

    return sorted(
        case_ids
    )


# ============================================================
# KG SEARCH
# ============================================================

def kg_search(
    query,
    nodes,
    relationships
):

    print(
        "\n"
        + "=" * 90
    )

    print(
        "KNOWLEDGE GRAPH SEARCH"
    )

    print(
        "=" * 90
    )

    # --------------------------------------------------------
    # Extract entities
    # --------------------------------------------------------

    query_entities = (
        extract_structured_entities(
            query
        )
    )

    print(
        "\nStructured query entities:"
    )

    if not query_entities:

        print(
            "  None detected."
        )

    else:

        for entity in query_entities:

            print(
                f"  {entity['node_type']} "
                f"-> "
                f"{entity['entity_name']}"
            )

    # --------------------------------------------------------
    # Match KG nodes
    # --------------------------------------------------------

    matched_nodes = (
        match_kg_entities(
            query_entities,
            nodes
        )
    )

    print(
        "\nMatched KG nodes:"
    )

    if not matched_nodes:

        print(
            "  None"
        )

    else:

        for node in matched_nodes:

            properties = node.get(
                "properties",
                {}
            )

            print(
                f"  Type   : "
                f"{node.get('node_type')}"
            )

            print(
                f"  ID     : "
                f"{node.get('node_id')}"
            )

            print(
                f"  Name   : "
                f"{properties.get('section_name', '')}"
            )

    # --------------------------------------------------------
    # Find actual case IDs
    # --------------------------------------------------------

    case_ids = (
        find_connected_cases(
            matched_nodes,
            relationships,
            nodes
        )
    )

    print(
        "\nKG candidate CASE IDs:"
    )

    if not case_ids:

        print(
            "  None"
        )

    else:

        for case_id in case_ids:

            print(
                f"  {case_id}"
            )

    return {

        "query_entities":
            query_entities,

        "matched_nodes":
            matched_nodes,

        "case_ids":
            case_ids
    }


# ============================================================
# GET CASE IDs FROM RAG
# ============================================================

def get_rag_case_ids(
    rag_cases
):

    case_ids = set()

    for case in rag_cases:

        case_id = case.get(
            "case_id"
        )

        if case_id:

            case_ids.add(
                case_id
            )

    return case_ids


# ============================================================
# BUILD CANDIDATE UNION
# ============================================================

def build_candidate_union(
    rag_cases,
    kg_case_ids
):

    rag_case_ids = (
        get_rag_case_ids(
            rag_cases
        )
    )

    kg_case_ids = set(
        kg_case_ids
    )

    # --------------------------------------------------------
    # Union
    # --------------------------------------------------------

    union_case_ids = (
        rag_case_ids
        | kg_case_ids
    )

    # --------------------------------------------------------
    # Overlap
    # --------------------------------------------------------

    overlap_case_ids = (
        rag_case_ids
        & kg_case_ids
    )

    # --------------------------------------------------------
    # KG-only
    # --------------------------------------------------------

    kg_only_case_ids = (
        kg_case_ids
        - rag_case_ids
    )

    return {

        "rag_case_ids":
            rag_case_ids,

        "kg_case_ids":
            kg_case_ids,

        "union_case_ids":
            union_case_ids,

        "overlap_case_ids":
            overlap_case_ids,

        "kg_only_case_ids":
            kg_only_case_ids
    }


# ============================================================
# DISPLAY RAG RESULTS
# ============================================================

def display_rag_results(
    rag_cases
):

    print(
        "\n"
        + "=" * 90
    )

    print(
        "EXISTING RAG FINAL CASES"
    )

    print(
        "=" * 90
    )

    if not rag_cases:

        print(
            "No RAG cases returned."
        )

        return

    for rank, case in enumerate(
        rag_cases,
        start=1
    ):

        case_id = case.get(
            "case_id",
            "UNKNOWN"
        )

        court = case.get(
            "court",
            "UNKNOWN"
        )

        score = case.get(
            "final_score"
        )

        if score is None:

            score = case.get(
                "case_score",
                0.0
            )

        print(
            f"{rank}. {case_id}"
        )

        print(
            f"   Court : {court}"
        )

        print(
            f"   Score : {float(score):.6f}"
        )


# ============================================================
# DISPLAY KG RESULTS
# ============================================================

def display_kg_results(
    kg_case_ids
):

    print(
        "\n"
        + "=" * 90
    )

    print(
        "KG CANDIDATE CASES"
    )

    print(
        "=" * 90
    )

    if not kg_case_ids:

        print(
            "No KG candidate cases found."
        )

        return

    for rank, case_id in enumerate(
        sorted(kg_case_ids),
        start=1
    ):

        print(
            f"{rank}. {case_id}"
        )


# ============================================================
# DISPLAY UNION
# ============================================================

def display_union(
    union
):

    print(
        "\n"
        + "=" * 90
    )

    print(
        "CANDIDATE UNION VALIDATION"
    )

    print(
        "=" * 90
    )

    print(
        f"RAG candidates      : "
        f"{len(union['rag_case_ids'])}"
    )

    print(
        f"KG candidates       : "
        f"{len(union['kg_case_ids'])}"
    )

    print(
        f"Overlap             : "
        f"{len(union['overlap_case_ids'])}"
    )

    print(
        f"KG-only candidates  : "
        f"{len(union['kg_only_case_ids'])}"
    )

    print(
        f"Union candidates    : "
        f"{len(union['union_case_ids'])}"
    )

    # --------------------------------------------------------
    # Validate that no Section/Article IDs entered union
    # --------------------------------------------------------

    invalid_ids = []

    for case_id in union[
        "union_case_ids"
    ]:

        case_id_text = str(
            case_id
        ).lower()

        if (
            case_id_text.startswith(
                "section_"
            )
            or
            case_id_text.startswith(
                "article_"
            )
        ):

            invalid_ids.append(
                case_id
            )

    print(
        "\n"
        + "-" * 90
    )

    print(
        "VALIDATION"
    )

    print(
        "-" * 90
    )

    if invalid_ids:

        print(
            "[FAIL] Invalid entity IDs "
            "found in candidate union:"
        )

        for item in invalid_ids:

            print(
                f"  {item}"
            )

    else:

        print(
            "[PASS] Candidate union contains "
            "only case IDs."
        )

    # --------------------------------------------------------
    # KG-only cases
    # --------------------------------------------------------

    if union[
        "kg_only_case_ids"
    ]:

        print(
            "\nKG-only case candidates:"
        )

        for case_id in sorted(
            union[
                "kg_only_case_ids"
            ]
        ):

            print(
                f"  {case_id}"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 90
    )

    print(
        "KG + RAG CANDIDATE UNION VALIDATION"
    )

    print(
        "=" * 90
    )

    print(
        f"\nQuery:\n{QUERY}"
    )

    # ========================================================
    # CHECK PATHS
    # ========================================================

    print(
        "\n"
        + "=" * 90
    )

    print(
        "PATH VALIDATION"
    )

    print(
        "=" * 90
    )

    print(
        f"Project root : "
        f"{PROJECT_ROOT}"
    )

    print(
        f"Core scripts : "
        f"{CORE_SCRIPT_DIR}"
    )

    print(
        f"KG file      : "
        f"{KG_FILE}"
    )

    if not os.path.exists(
        os.path.join(
            CORE_SCRIPT_DIR,
            "authority_rerank.py"
        )
    ):

        raise FileNotFoundError(
            "\nCould not find authority_rerank.py at:\n"
            f"{CORE_SCRIPT_DIR}"
        )

    if not os.path.exists(
        KG_FILE
    ):

        raise FileNotFoundError(
            "\nCould not find all_nodes.json at:\n"
            f"{KG_FILE}"
        )

    print(
        "\n[PASS] Required paths found."
    )

    # ========================================================
    # LOAD KNOWLEDGE GRAPH
    # ========================================================

    print(
        "\nLoading Knowledge Graph..."
    )

    nodes, relationships = (
        load_knowledge_graph()
    )

    print(
        f"KG nodes         : "
        f"{len(nodes)}"
    )

    print(
        f"KG relationships : "
        f"{len(relationships)}"
    )

    # ========================================================
    # KG SEARCH
    # ========================================================

    kg_result = kg_search(
        QUERY,
        nodes,
        relationships
    )

    kg_case_ids = kg_result[
        "case_ids"
    ]

    # ========================================================
    # LOAD EXISTING RAG
    # ========================================================

    print(
        "\n"
        + "=" * 90
    )

    print(
        "LOADING EXISTING RAG"
    )

    print(
        "=" * 90
    )

    bm25, bm25_metadata = (
        load_bm25()
    )

    print(
        f"BM25 documents : "
        f"{len(bm25_metadata)}"
    )

    faiss_index, faiss_metadata = (
        load_faiss()
    )

    print(
        f"FAISS vectors  : "
        f"{faiss_index.ntotal}"
    )

    embedding_model = (
        load_embedding_model()
    )

    print(
        "Embedding model: loaded"
    )

    # ========================================================
    # RUN EXISTING RAG
    # ========================================================

    print(
        "\nRunning existing RAG..."
    )

    (
        original_cases,
        filtered_cases,
        reranked_cases
    ) = search(
        QUERY,
        bm25,
        bm25_metadata,
        faiss_index,
        faiss_metadata,
        embedding_model
    )

    # --------------------------------------------------------
    # Take final RAG cases
    # --------------------------------------------------------

    rag_cases = (
        reranked_cases[
            :RAG_TOP_CASES
        ]
    )

    display_rag_results(
        rag_cases
    )

    # ========================================================
    # DISPLAY KG CASES
    # ========================================================

    display_kg_results(
        kg_case_ids
    )

    # ========================================================
    # BUILD UNION
    # ========================================================

    union = (
        build_candidate_union(
            rag_cases,
            kg_case_ids
        )
    )

    # ========================================================
    # DISPLAY UNION
    # ========================================================

    display_union(
        union
    )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print(
        "\n"
        + "=" * 90
    )

    print(
        "FINAL CANDIDATE UNION"
    )

    print(
        "=" * 90
    )

    for case_id in sorted(
        union[
            "union_case_ids"
        ]
    ):

        sources = []

        if case_id in union[
            "rag_case_ids"
        ]:

            sources.append(
                "RAG"
            )

        if case_id in union[
            "kg_case_ids"
        ]:

            sources.append(
                "KG"
            )

        print(
            f"{case_id}"
            f"  ->  "
            f"{' + '.join(sources)}"
        )

    # ========================================================
    # COMPLETION
    # ========================================================

    print(
        "\n"
        + "=" * 90
    )

    print(
        "KG + RAG CANDIDATE UNION TEST COMPLETE"
    )

    print(
        "=" * 90
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()