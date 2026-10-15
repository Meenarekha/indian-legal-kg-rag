import os
import sys
import json
import re
import numpy as np


# ============================================================
# PATH CONFIGURATION
# ============================================================

KG_SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(KG_SCRIPT_DIR)
)

CORE_SCRIPT_DIR = os.path.join(
    PROJECT_ROOT,
    "scripts"
)

KG_NODES_FILE = os.path.join(
    PROJECT_ROOT,
    "knowledge_graph",
    "data",
    "nodes",
    "all_nodes.json"
)


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

from hybrid_search import (
    load_bm25,
    load_faiss,
    load_embedding_model,
    tokenize,
)

from authority_rerank import (
    citation_signal,
    calculate_temporal_scores,
    get_authority_score,
)

from case_level_search import (
    aggregate_by_case,
)


# ============================================================
# CONFIGURATION
# ============================================================

RRF_K = 60

FINAL_CASES = 5

SUPPORTING_CHUNKS = 3

# ------------------------------------------------------------
# KG-AWARE FINAL RANKING WEIGHTS
#
# Relevance remains dominant.
# KG provides a bounded structured relevance signal.
# ------------------------------------------------------------

RELEVANCE_WEIGHT = 0.80

AUTHORITY_WEIGHT = 0.10

KG_WEIGHT = 0.07

CITATION_WEIGHT = 0.02

TEMPORAL_WEIGHT = 0.01


# ============================================================
# PATH VALIDATION
# ============================================================

def validate_paths():

    print("=" * 90)
    print("PATH VALIDATION")
    print("=" * 90)

    print(
        f"Project root : {PROJECT_ROOT}"
    )

    print(
        f"Core scripts : {CORE_SCRIPT_DIR}"
    )

    print(
        f"KG file      : {KG_NODES_FILE}"
    )

    required_files = [

        KG_NODES_FILE,

        os.path.join(
            CORE_SCRIPT_DIR,
            "authority_rerank.py"
        ),

        os.path.join(
            CORE_SCRIPT_DIR,
            "hybrid_search.py"
        ),

        os.path.join(
            CORE_SCRIPT_DIR,
            "case_level_search.py"
        ),
    ]

    for path in required_files:

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )

    print(
        "\n[PASS] Required paths found."
    )


# ============================================================
# LOAD KNOWLEDGE GRAPH
# ============================================================

def load_knowledge_graph():

    print(
        "\nLoading Knowledge Graph..."
    )

    with open(
        KG_NODES_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    nodes = data.get(
        "nodes",
        []
    )

    relationships = data.get(
        "relationships",
        []
    )

    print(
        f"KG nodes         : {len(nodes)}"
    )

    print(
        f"KG relationships : {len(relationships)}"
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
# EXTRACT STRUCTURED LEGAL ENTITIES
# ============================================================

def extract_structured_entities(query):

    entities = []

    # --------------------------------------------------------
    # STATUTORY SECTION
    # --------------------------------------------------------

    section_matches = re.findall(
        r"\bsection\s+(\d+[a-z]?)\b",
        query,
        flags=re.IGNORECASE
    )

    for number in section_matches:

        entities.append(
            {
                "type": "Section",
                "name": f"Section {number}"
            }
        )

    # --------------------------------------------------------
    # CONSTITUTIONAL ARTICLE
    # --------------------------------------------------------

    article_matches = re.findall(
        r"\barticle\s+(\d+[a-z]?)\b",
        query,
        flags=re.IGNORECASE
    )

    for number in article_matches:

        entities.append(
            {
                "type": "Section",
                "name": f"Article {number}"
            }
        )

    return entities


# ============================================================
# MATCH KG ENTITIES
# ============================================================

def match_kg_entities(
    entities,
    node_list
):

    matches = []

    for entity in entities:

        target_name = normalize_text(
            entity["name"]
        )

        for node in node_list:

            if node.get(
                "node_type"
            ) != entity["type"]:

                continue

            properties = node.get(
                "properties",
                {}
            )

            node_name = properties.get(
                "section_name"
            )

            if not node_name:

                node_name = properties.get(
                    "name"
                )

            if not node_name:
                continue

            if normalize_text(
                node_name
            ) == target_name:

                matches.append(
                    node
                )

    return matches


# ============================================================
# FIND CORPUS CASE NODES
# ============================================================

def get_corpus_case_nodes(
    node_list
):

    case_nodes = {}

    for node in node_list:

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
        # IMPORTANT:
        # Reference-only cited cases are not retrieval cases.
        # ----------------------------------------------------

        if properties.get(
            "reference_only",
            False
        ):

            continue

        if node_id:

            case_nodes[
                node_id
            ] = node

    return case_nodes


# ============================================================
# KG ENTITY -> CASE TRAVERSAL
# ============================================================

def get_kg_case_candidates(
    matched_nodes,
    node_list,
    relationships
):

    case_nodes = get_corpus_case_nodes(
        node_list
    )

    matched_node_ids = {

        node.get(
            "node_id"
        )

        for node in matched_nodes

    }

    case_ids = set()

    for relationship in relationships:

        relationship_type = (
            relationship.get(
                "relationship_type"
            )
        )

        if relationship_type != (
            "INVOLVES_SECTION"
        ):

            continue

        source = relationship.get(
            "source"
        )

        target = relationship.get(
            "target"
        )

        # ----------------------------------------------------
        # Normal case -> section direction
        # ----------------------------------------------------

        if (

            source in case_nodes

            and

            target in matched_node_ids

        ):

            case_ids.add(
                source
            )

        # ----------------------------------------------------
        # Safety support for reverse direction
        # ----------------------------------------------------

        elif (

            target in case_nodes

            and

            source in matched_node_ids

        ):

            case_ids.add(
                target
            )

    return case_ids


# ============================================================
# BUILD CASE -> CHUNK MAP
# ============================================================

def build_case_chunk_map(
    metadata
):

    case_map = {}

    for index, item in enumerate(
        metadata
    ):

        case_id = item.get(
            "case_id"
        )

        if not case_id:
            continue

        if case_id not in case_map:

            case_map[
                case_id
            ] = []

        case_map[
            case_id
        ].append(
            index
        )

    return case_map


# ============================================================
# CREATE CHUNK RESULT
# ============================================================

def create_chunk_result(
    metadata,
    index
):

    item = metadata[
        index
    ]

    pages = item.get(
        "pages",
        []
    )

    return {

        "rank": None,

        "score": 0.0,

        "chunk_id":
            item.get(
                "chunk_id"
            ),

        "case_id":
            item.get(
                "case_id"
            ),

        "court":
            item.get(
                "court"
            ),

        "source_file":
            item.get(
                "source_file"
            ),

        "start_page":
            item.get(
                "start_page",
                pages[0]
                if pages
                else None
            ),

        "end_page":
            item.get(
                "end_page",
                pages[-1]
                if pages
                else None
            ),

        "pages":
            pages,

        "paragraph_start":
            item.get(
                "paragraph_start"
            ),

        "paragraph_end":
            item.get(
                "paragraph_end"
            ),

        "text":
            item.get(
                "text",
                ""
            ),

        "rrf_score":
            0.0,

        "bm25_rank":
            None,

        "bm25_score":
            None,

        "faiss_rank":
            None,

        "faiss_score":
            None,
    }


# ============================================================
# CANDIDATE-SPECIFIC RETRIEVAL
# ============================================================

def retrieve_candidate_chunks(
    query,
    candidate_case_ids,
    bm25,
    bm25_metadata,
    faiss_index,
    faiss_metadata,
    embedding_model
):

    print(
        "\nRunning candidate-specific "
        "BM25 + Dense retrieval..."
    )

    bm25_case_map = (
        build_case_chunk_map(
            bm25_metadata
        )
    )

    faiss_case_map = (
        build_case_chunk_map(
            faiss_metadata
        )
    )

    # --------------------------------------------------------
    # Find candidate chunks
    # --------------------------------------------------------

    candidate_bm25_indices = []

    for case_id in candidate_case_ids:

        candidate_bm25_indices.extend(
            bm25_case_map.get(
                case_id,
                []
            )
        )

    candidate_faiss_indices = []

    for case_id in candidate_case_ids:

        candidate_faiss_indices.extend(
            faiss_case_map.get(
                case_id,
                []
            )
        )

    candidate_bm25_indices = sorted(
        set(
            candidate_bm25_indices
        )
    )

    candidate_faiss_indices = sorted(
        set(
            candidate_faiss_indices
        )
    )

    print(
        f"Candidate BM25 chunks  : "
        f"{len(candidate_bm25_indices)}"
    )

    print(
        f"Candidate FAISS chunks : "
        f"{len(candidate_faiss_indices)}"
    )

    if not candidate_bm25_indices:

        return []

    if not candidate_faiss_indices:

        return []

    # ========================================================
    # BM25
    # ========================================================

    query_tokens = tokenize(
        query
    )

    bm25_scores = bm25.get_scores(
        query_tokens
    )

    bm25_order = np.argsort(
        bm25_scores
    )[::-1]

    bm25_rank_lookup = {}

    for rank, index in enumerate(
        bm25_order,
        start=1
    ):

        bm25_rank_lookup[
            int(index)
        ] = rank

    # ========================================================
    # DENSE
    # ========================================================

    query_embedding = (
        embedding_model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True
        )
    )

    query_embedding = (
        query_embedding.astype(
            "float32"
        )
    )

    dense_scores, dense_indices = (
        faiss_index.search(
            query_embedding,
            faiss_index.ntotal
        )
    )

    dense_rank_lookup = {}

    dense_score_lookup = {}

    for rank, (
        score,
        index
    ) in enumerate(
        zip(
            dense_scores[0],
            dense_indices[0]
        ),
        start=1
    ):

        index = int(index)

        if index < 0:
            continue

        dense_rank_lookup[
            index
        ] = rank

        dense_score_lookup[
            index
        ] = float(
            score
        )

    # ========================================================
    # BUILD CHUNK RESULT MAP
    # ========================================================

    chunk_results = {}

    # --------------------------------------------------------
    # BM25 candidate chunks
    # --------------------------------------------------------

    for index in candidate_bm25_indices:

        item = bm25_metadata[
            index
        ]

        chunk_id = item.get(
            "chunk_id"
        )

        if not chunk_id:
            continue

        result = create_chunk_result(
            bm25_metadata,
            index
        )

        result[
            "bm25_rank"
        ] = bm25_rank_lookup.get(
            index
        )

        result[
            "bm25_score"
        ] = float(
            bm25_scores[index]
        )

        chunk_results[
            chunk_id
        ] = result

    # --------------------------------------------------------
    # FAISS candidate chunks
    # --------------------------------------------------------

    for index in candidate_faiss_indices:

        item = faiss_metadata[
            index
        ]

        chunk_id = item.get(
            "chunk_id"
        )

        if not chunk_id:
            continue

        if chunk_id in chunk_results:

            result = chunk_results[
                chunk_id
            ]

        else:

            result = create_chunk_result(
                faiss_metadata,
                index
            )

            chunk_results[
                chunk_id
            ] = result

        result[
            "faiss_rank"
        ] = dense_rank_lookup.get(
            index
        )

        result[
            "faiss_score"
        ] = dense_score_lookup.get(
            index
        )

    # ========================================================
    # RRF
    # ========================================================

    for result in chunk_results.values():

        rrf_score = 0.0

        bm25_rank = result.get(
            "bm25_rank"
        )

        faiss_rank = result.get(
            "faiss_rank"
        )

        if bm25_rank is not None:

            rrf_score += (
                1.0
                /
                (
                    RRF_K
                    + bm25_rank
                )
            )

        if faiss_rank is not None:

            rrf_score += (
                1.0
                /
                (
                    RRF_K
                    + faiss_rank
                )
            )

        result[
            "rrf_score"
        ] = rrf_score

    ranked_chunks = sorted(
        chunk_results.values(),
        key=lambda x:
            x["rrf_score"],
        reverse=True
    )

    return ranked_chunks


# ============================================================
# MARK KG MATCHES
# ============================================================

def mark_kg_candidates(
    case_results,
    kg_case_ids
):

    for case in case_results:

        case_id = case.get(
            "case_id"
        )

        if case_id in kg_case_ids:

            case[
                "kg_match"
            ] = True

            case[
                "kg_relevance"
            ] = 1.0

            case[
                "kg_reason"
            ] = (
                "Explicit legal entity "
                "matched through the "
                "Knowledge Graph."
            )

        else:

            case[
                "kg_match"
            ] = False

            case[
                "kg_relevance"
            ] = 0.0

            case[
                "kg_reason"
            ] = None

    return case_results


# ============================================================
# KG-AWARE RELEVANCE FILTER
# ============================================================

def kg_aware_relevance_filter(
    case_results,
    has_structured_query
):

    if not case_results:

        return []

    # --------------------------------------------------------
    # Find strongest retrieval relevance
    # --------------------------------------------------------

    best_score = max(
        case.get(
            "case_score",
            0.0
        )
        for case in case_results
    )

    relevance_floor = (
        best_score
        * 0.45
    )

    # --------------------------------------------------------
    # Normal query
    #
    # Behaves like the existing RAG filter.
    # --------------------------------------------------------

    if not has_structured_query:

        filtered = [

            case

            for case in case_results

            if case.get(
                "case_score",
                0.0
            ) >= relevance_floor

        ]

        if not filtered:

            return [
                case_results[0]
            ]

        return filtered

    # --------------------------------------------------------
    # Structured legal query
    #
    # IMPORTANT:
    #
    # KG-confirmed cases are retained even if their
    # lexical/semantic score is below the generic floor.
    #
    # This prevents the KG contribution from being discarded
    # before final ranking.
    # --------------------------------------------------------

    filtered = []

    for case in case_results:

        retrieval_relevant = (
            case.get(
                "case_score",
                0.0
            )
            >= relevance_floor
        )

        kg_relevant = (
            case.get(
                "kg_match",
                False
            )
        )

        if (
            retrieval_relevant
            or
            kg_relevant
        ):

            filtered.append(
                case
            )

    if not filtered:

        return [
            case_results[0]
        ]

    return filtered


# ============================================================
# KG-AWARE FINAL RANKING
# ============================================================

def kg_aware_rerank(
    case_results
):

    if not case_results:

        return []

    # --------------------------------------------------------
    # Normalize retrieval relevance
    # --------------------------------------------------------

    maximum_relevance = max(

        case.get(
            "case_score",
            0.0
        )

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

        case_id = case.get(
            "case_id"
        )

        # ----------------------------------------------------
        # Retrieval relevance
        # ----------------------------------------------------

        relevance = (
            case.get(
                "case_score",
                0.0
            )
            /
            maximum_relevance
        )

        relevance = min(
            max(
                relevance,
                0.0
            ),
            1.0
        )

        # ----------------------------------------------------
        # Court authority
        # ----------------------------------------------------

        authority = get_authority_score(
            case.get(
                "court"
            )
        )

        # ----------------------------------------------------
        # KG structured relevance
        # ----------------------------------------------------

        kg_relevance = case.get(
            "kg_relevance",
            0.0
        )

        kg_relevance = min(
            max(
                kg_relevance,
                0.0
            ),
            1.0
        )

        # ----------------------------------------------------
        # Citation signal
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
        # FINAL SCORE
        #
        # Retrieval remains dominant.
        # KG contributes a bounded structured signal.
        # ----------------------------------------------------

        final_score = (

            RELEVANCE_WEIGHT
            * relevance

            +

            AUTHORITY_WEIGHT
            * authority

            +

            KG_WEIGHT
            * kg_relevance

            +

            CITATION_WEIGHT
            * citation

            +

            TEMPORAL_WEIGHT
            * temporal

        )

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
            "kg_relevance"
        ] = kg_relevance

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
    # Sort
    # --------------------------------------------------------

    reranked.sort(
        key=lambda x:
            x["final_score"],
        reverse=True
    )

    return reranked


# ============================================================
# MAIN SEARCH PIPELINE
# ============================================================

def search(
    query,
    node_list,
    relationships,
    bm25,
    bm25_metadata,
    faiss_index,
    faiss_metadata,
    embedding_model
):

    # ========================================================
    # 1. EXTRACT STRUCTURED ENTITIES
    # ========================================================

    entities = (
        extract_structured_entities(
            query
        )
    )

    print(
        "\nStructured query entities:"
    )

    if entities:

        for entity in entities:

            print(
                f"  {entity['type']} "
                f"-> {entity['name']}"
            )

    else:

        print(
            "  No explicit Section/Article "
            "entity detected."
        )

    has_structured_query = bool(
        entities
    )

    # ========================================================
    # 2. MATCH KG
    # ========================================================

    matched_nodes = (
        match_kg_entities(
            entities,
            node_list
        )
    )

    print(
        "\nMatched KG nodes:"
    )

    if matched_nodes:

        for node in matched_nodes:

            properties = node.get(
                "properties",
                {}
            )

            print(
                f"  Type : "
                f"{node.get('node_type')}"
            )

            print(
                f"  ID   : "
                f"{node.get('node_id')}"
            )

            print(
                f"  Name : "
                f"{properties.get('section_name', properties.get('name'))}"
            )

    else:

        print(
            "  No KG nodes matched."
        )

    # ========================================================
    # 3. KG CASE CANDIDATES
    # ========================================================

    kg_case_ids = (
        get_kg_case_candidates(
            matched_nodes,
            node_list,
            relationships
        )
    )

    print(
        "\nKG candidate CASE IDs:"
    )

    if kg_case_ids:

        for case_id in sorted(
            kg_case_ids
        ):

            print(
                f"  {case_id}"
            )

    else:

        print(
            "  No KG case candidates."
        )

    # ========================================================
    # 4. EXISTING RAG
    # ========================================================

    print(
        "\nRunning existing RAG..."
    )

    from authority_rerank import (
        search as rag_search
    )

    (
        original_cases,
        original_filtered,
        original_reranked
    ) = rag_search(
        query,
        bm25,
        bm25_metadata,
        faiss_index,
        faiss_metadata,
        embedding_model
    )

    existing_rag_case_ids = {

        case.get(
            "case_id"
        )

        for case in original_reranked

    }

    # ========================================================
    # 5. UNION
    # ========================================================

    union_case_ids = (
        existing_rag_case_ids
        |
        kg_case_ids
    )

    overlap = (
        existing_rag_case_ids
        &
        kg_case_ids
    )

    print(
        "\nCandidate union:"
    )

    print(
        f"Existing RAG cases : "
        f"{len(existing_rag_case_ids)}"
    )

    print(
        f"KG cases           : "
        f"{len(kg_case_ids)}"
    )

    print(
        f"Overlap             : "
        f"{len(overlap)}"
    )

    print(
        f"Union cases        : "
        f"{len(union_case_ids)}"
    )

    # ========================================================
    # 6. CANDIDATE-SPECIFIC RETRIEVAL
    # ========================================================

    ranked_chunks = (
        retrieve_candidate_chunks(
            query,
            union_case_ids,
            bm25,
            bm25_metadata,
            faiss_index,
            faiss_metadata,
            embedding_model
        )
    )

    print(
        f"\nCandidate evidence chunks: "
        f"{len(ranked_chunks)}"
    )

    # ========================================================
    # 7. CASE AGGREGATION
    # ========================================================

    case_results = (
        aggregate_by_case(
            ranked_chunks
        )
    )

    # ========================================================
    # 8. MARK KG CASES
    # ========================================================

    case_results = (
        mark_kg_candidates(
            case_results,
            kg_case_ids
        )
    )

    # ========================================================
    # 9. KG-AWARE RELEVANCE FILTER
    # ========================================================

    filtered_cases = (
        kg_aware_relevance_filter(
            case_results,
            has_structured_query
        )
    )

    # ========================================================
    # 10. KG-AWARE FINAL RANKING
    # ========================================================

    reranked_cases = (
        kg_aware_rerank(
            filtered_cases
        )
    )

    # ========================================================
    # 11. FINAL TOP CASES
    # ========================================================

    final_cases = (
        reranked_cases[
            :FINAL_CASES
        ]
    )

    return (
        final_cases,
        case_results,
        filtered_cases,
        ranked_chunks,
        kg_case_ids,
        union_case_ids
    )


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(
    query,
    final_cases,
    kg_case_ids,
    union_case_ids
):

    print(
        "\n"
        + "=" * 90
    )

    print(
        "KG + RAG FINAL RANKED RETRIEVAL"
    )

    print(
        "=" * 90
    )

    print(
        f"\nQuery:\n{query}"
    )

    print(
        f"\nKG candidates : "
        f"{len(kg_case_ids)}"
    )

    print(
        f"Union cases   : "
        f"{len(union_case_ids)}"
    )

    print(
        f"Final cases   : "
        f"{len(final_cases)}"
    )

    for rank, case in enumerate(
        final_cases,
        start=1
    ):

        print(
            "\n"
            + "-" * 90
        )

        print(
            f"RANK           : {rank}"
        )

        print(
            f"CASE ID        : "
            f"{case.get('case_id')}"
        )

        print(
            f"COURT          : "
            f"{case.get('court')}"
        )

        print(
            f"FINAL SCORE    : "
            f"{case.get('final_score', 0):.6f}"
        )

        print(
            f"RELEVANCE      : "
            f"{case.get('normalized_relevance', 0):.6f}"
        )

        print(
            f"AUTHORITY      : "
            f"{case.get('authority_score', 0):.6f}"
        )

        print(
            f"KG RELEVANCE   : "
            f"{case.get('kg_relevance', 0):.6f}"
        )

        print(
            f"CITATION       : "
            f"{case.get('citation_score', 0):.6f}"
        )

        print(
            f"TEMPORAL       : "
            f"{case.get('temporal_score', 0):.6f}"
        )

        print(
            f"KG MATCH        : "
            f"{case.get('kg_match')}"
        )

        if case.get(
            "kg_match"
        ):

            print(
                f"KG REASON       : "
                f"{case.get('kg_reason')}"
            )

        print(
            f"SUPPORTING CHUNKS : "
            f"{len(case.get('supporting_chunks', []))}"
        )

        print(
            "\nTop supporting evidence:"
        )

        for passage_rank, chunk in enumerate(
            case.get(
                "supporting_chunks",
                []
            )[:SUPPORTING_CHUNKS],
            start=1
        ):

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
                f"\n  Evidence {passage_rank}"
            )

            print(
                f"  RRF score : "
                f"{chunk.get('rrf_score', 0):.6f}"
            )

            print(
                f"  Pages     : "
                f"{chunk.get('pages', [])}"
            )

            print(
                f"  Text      : "
                f"{text}"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 90
    )

    print(
        "KG + RAG KG-AWARE RANKING"
    )

    print(
        "=" * 90
    )

    query = input(
        "\nEnter legal query: "
    ).strip()

    if not query:

        print(
            "ERROR: Query cannot be empty."
        )

        return

    try:

        # ----------------------------------------------------
        # PATHS
        # ----------------------------------------------------

        validate_paths()

        # ----------------------------------------------------
        # KG
        # ----------------------------------------------------

        (
            node_list,
            relationships
        ) = load_knowledge_graph()

        # ----------------------------------------------------
        # BM25
        # ----------------------------------------------------

        print(
            "\nLoading BM25..."
        )

        (
            bm25,
            bm25_metadata
        ) = load_bm25()

        print(
            f"BM25 documents: "
            f"{len(bm25_metadata)}"
        )

        # ----------------------------------------------------
        # FAISS
        # ----------------------------------------------------

        print(
            "\nLoading FAISS..."
        )

        (
            faiss_index,
            faiss_metadata
        ) = load_faiss()

        print(
            f"FAISS vectors: "
            f"{faiss_index.ntotal}"
        )

        # ----------------------------------------------------
        # EMBEDDING MODEL
        # ----------------------------------------------------

        print(
            "\nLoading embedding model..."
        )

        embedding_model = (
            load_embedding_model()
        )

        print(
            "Embedding model loaded."
        )

        # ----------------------------------------------------
        # SEARCH
        # ----------------------------------------------------

        (
            final_cases,
            case_results,
            filtered_cases,
            ranked_chunks,
            kg_case_ids,
            union_case_ids
        ) = search(
            query,
            node_list,
            relationships,
            bm25,
            bm25_metadata,
            faiss_index,
            faiss_metadata,
            embedding_model
        )

        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        display_results(
            query,
            final_cases,
            kg_case_ids,
            union_case_ids
        )

        print(
            "\n"
            + "=" * 90
        )

        print(
            "KG + RAG KG-AWARE RANKING TEST COMPLETE"
        )

        print(
            "=" * 90
        )

    except Exception as error:

        print(
            "\n"
            + "=" * 90
        )

        print(
            "ERROR"
        )

        print(
            "=" * 90
        )

        print(
            f"{type(error).__name__}: "
            f"{error}"
        )

        raise


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()