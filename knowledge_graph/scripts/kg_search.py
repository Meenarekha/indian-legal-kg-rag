import json
import re
from pathlib import Path
from collections import defaultdict


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

KG_PATH = (
    PROJECT_ROOT
    / "knowledge_graph"
    / "data"
    / "knowledge_graph.json"
)


# ============================================================
# LOAD KNOWLEDGE GRAPH
# ============================================================

def load_knowledge_graph():
    """
    Load the final knowledge graph JSON file.
    """

    if not KG_PATH.exists():
        raise FileNotFoundError(
            f"Knowledge graph file not found:\n{KG_PATH}"
        )

    with open(KG_PATH, "r", encoding="utf-8") as file:
        graph = json.load(file)

    nodes = graph.get("nodes", [])
    relationships = graph.get("relationships", [])

    print("\nKnowledge Graph loaded successfully.")
    print(f"Nodes         : {len(nodes)}")
    print(f"Relationships : {len(relationships)}")

    return nodes, relationships


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text):
    """
    Normalize text for simple matching.

    Example:
        "Section 438 CrPC"
            ->
        "section 438 crpc"
    """

    if not text:
        return ""

    text = text.lower()

    text = re.sub(r"[^a-z0-9]+", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# BUILD LOOKUP TABLES
# ============================================================

def build_lookup_tables(nodes, relationships):
    """
    Build fast lookup structures.

    node_by_id:
        node_id -> node

    outgoing:
        source_node_id -> relationships

    incoming:
        target_node_id -> relationships
    """

    node_by_id = {}

    for node in nodes:
        node_id = node.get("node_id")

        if node_id:
            node_by_id[node_id] = node

    outgoing = defaultdict(list)
    incoming = defaultdict(list)

    for relationship in relationships:

        source = relationship.get("source")
        target = relationship.get("target")

        if source:
            outgoing[source].append(relationship)

        if target:
            incoming[target].append(relationship)

    return node_by_id, outgoing, incoming


# ============================================================
# GET SEARCHABLE NODE TEXT
# ============================================================

def get_node_text(node):
    """
    Convert useful node properties into searchable text.
    """

    properties = node.get("properties", {})

    values = []

    # Node ID may itself contain useful information
    node_id = node.get("node_id", "")

    if node_id:
        values.append(node_id)

    # Common properties used in our KG
    possible_fields = [
        "case_id",
        "case_name",
        "court",
        "date",
        "judge_name",
        "section",
        "section_name",
        "name",
        "title"
    ]

    for field in possible_fields:

        value = properties.get(field)

        if value:
            values.append(str(value))

    return normalize_text(" ".join(values))


# ============================================================
# EXTRACT SECTION NUMBERS FROM QUERY
# ============================================================

def extract_section_numbers(query):
    """
    Extract explicit Section references.

    Examples:

        Section 438
        section 302
        Sec 144
        Sec. 420
    """

    pattern = r"\b(?:section|sec)\.?\s*(\d+[a-zA-Z]?)\b"

    matches = re.findall(
        pattern,
        query,
        flags=re.IGNORECASE
    )

    return list(dict.fromkeys(match.lower() for match in matches))


# ============================================================
# FIND DIRECT NODE MATCHES
# ============================================================

def find_matching_nodes(query, nodes):
    """
    Find nodes whose text directly matches useful parts
    of the user's query.
    """

    normalized_query = normalize_text(query)

    section_numbers = extract_section_numbers(query)

    matches = []

    for node in nodes:

        node_type = node.get("node_type", "")

        node_text = get_node_text(node)

        score = 0
        reasons = []

        # ----------------------------------------------------
        # SECTION MATCH
        # ----------------------------------------------------

        if node_type == "Section":

            for section_number in section_numbers:

                section_pattern = rf"\b{re.escape(section_number)}\b"

                if re.search(section_pattern, node_text):

                    score += 5

                    reasons.append(
                        f"Section {section_number} match"
                    )

        # ----------------------------------------------------
        # CASE MATCH
        # ----------------------------------------------------

        elif node_type == "Case":

            properties = node.get("properties", {})

            case_name = normalize_text(
                properties.get("case_name", "")
            )

            if case_name:

                # Exact case name contained in query
                if case_name in normalized_query:

                    score += 8
                    reasons.append("Exact case-name match")

                else:
                    # Token overlap for partial case-name search
                    case_tokens = set(case_name.split())
                    query_tokens = set(normalized_query.split())

                    if case_tokens:

                        overlap = (
                            len(case_tokens & query_tokens)
                            / len(case_tokens)
                        )

                        if overlap >= 0.60:
                            score += 4
                            reasons.append(
                                "Partial case-name match"
                            )

        # ----------------------------------------------------
        # COURT MATCH
        # ----------------------------------------------------

        elif node_type == "Court":

            properties = node.get("properties", {})

            court_name = normalize_text(
                properties.get(
                    "name",
                    properties.get("court", "")
                )
            )

            if court_name and court_name in normalized_query:

                score += 3
                reasons.append("Court match")

        if score > 0:

            matches.append(
                {
                    "node": node,
                    "score": score,
                    "reasons": reasons
                }
            )

    matches.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return matches


# ============================================================
# GET CASES CONNECTED TO MATCHED NODE
# ============================================================

def find_connected_cases(
    matched_node,
    node_by_id,
    outgoing,
    incoming
):
    """
    Find Case nodes connected to a matched KG node.

    Example:

        Section 438
            <- INVOLVES_SECTION -
        Case A

    or

        Case A
            -> CITES
        Case B
    """

    node_id = matched_node.get("node_id")

    connected_cases = []

    # --------------------------------------------------------
    # OUTGOING RELATIONSHIPS
    # --------------------------------------------------------

    for relationship in outgoing.get(node_id, []):

        target_id = relationship.get("target")

        target_node = node_by_id.get(target_id)

        if not target_node:
            continue

        if target_node.get("node_type") == "Case":

            connected_cases.append(
                {
                    "case_node": target_node,
                    "relationship": relationship.get(
                        "relationship_type"
                    ),
                    "direction": "outgoing"
                }
            )

    # --------------------------------------------------------
    # INCOMING RELATIONSHIPS
    # --------------------------------------------------------

    for relationship in incoming.get(node_id, []):

        source_id = relationship.get("source")

        source_node = node_by_id.get(source_id)

        if not source_node:
            continue

        if source_node.get("node_type") == "Case":

            connected_cases.append(
                {
                    "case_node": source_node,
                    "relationship": relationship.get(
                        "relationship_type"
                    ),
                    "direction": "incoming"
                }
            )

    return connected_cases


# ============================================================
# SEARCH KNOWLEDGE GRAPH
# ============================================================

def search_knowledge_graph(
    query,
    nodes,
    relationships,
    top_k=10
):
    """
    Search KG and return candidate corpus cases.
    """

    node_by_id, outgoing, incoming = build_lookup_tables(
        nodes,
        relationships
    )

    matches = find_matching_nodes(
        query,
        nodes
    )

    candidate_cases = {}

    # ========================================================
    # PROCESS MATCHED NODES
    # ========================================================

    for match in matches:

        matched_node = match["node"]

        node_type = matched_node.get("node_type")

        match_score = match["score"]

        reasons = match["reasons"]

        # ----------------------------------------------------
        # CASE NODE ITSELF
        # ----------------------------------------------------

        if node_type == "Case":

            properties = matched_node.get(
                "properties",
                {}
            )

            # Only actual corpus cases should become
            # retrieval candidates.
            if not properties.get(
                "reference_only",
                False
            ):

                case_id = properties.get(
                    "case_id",
                    matched_node.get("node_id")
                )

                candidate_cases[case_id] = {
                    "case_id": case_id,
                    "case_name": properties.get(
                        "case_name",
                        case_id
                    ),
                    "court": properties.get(
                        "court",
                        "Unknown"
                    ),
                    "kg_score": match_score,
                    "reasons": list(reasons)
                }

        # ----------------------------------------------------
        # CASES CONNECTED TO MATCHED NODE
        # ----------------------------------------------------

        connected_cases = find_connected_cases(
            matched_node,
            node_by_id,
            outgoing,
            incoming
        )

        for connected in connected_cases:

            case_node = connected["case_node"]

            properties = case_node.get(
                "properties",
                {}
            )

            # Skip citation-only external cases
            if properties.get(
                "reference_only",
                False
            ):
                continue

            case_id = properties.get(
                "case_id",
                case_node.get("node_id")
            )

            relationship_type = connected[
                "relationship"
            ]

            relationship_reason = (
                f"Connected through "
                f"{relationship_type}"
            )

            # Give connected cases slightly lower score
            connected_score = max(
                match_score - 1,
                1
            )

            if case_id not in candidate_cases:

                candidate_cases[case_id] = {
                    "case_id": case_id,
                    "case_name": properties.get(
                        "case_name",
                        case_id
                    ),
                    "court": properties.get(
                        "court",
                        "Unknown"
                    ),
                    "kg_score": connected_score,
                    "reasons": [
                        relationship_reason
                    ]
                }

            else:

                candidate_cases[case_id][
                    "kg_score"
                ] += connected_score

                candidate_cases[case_id][
                    "reasons"
                ].append(
                    relationship_reason
                )

    # ========================================================
    # SORT RESULTS
    # ========================================================

    results = list(candidate_cases.values())

    results.sort(
        key=lambda item: item["kg_score"],
        reverse=True
    )

    return results[:top_k], matches


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(query, results, matches):
    """
    Pretty-print KG search results.
    """

    print("\n" + "=" * 75)

    print("KNOWLEDGE GRAPH SEARCH")

    print("=" * 75)

    print(f"\nQuery:\n{query}")

    print("\nMatched KG entities:")

    if not matches:

        print("  No direct KG entity matches.")

    else:

        for index, match in enumerate(
            matches[:10],
            start=1
        ):

            node = match["node"]

            print(
                f"\n{index}. "
                f"{node.get('node_type')} | "
                f"{node.get('node_id')}"
            )

            print(
                f"   Score   : {match['score']}"
            )

            print(
                f"   Reasons : "
                f"{', '.join(match['reasons'])}"
            )

    print("\n" + "-" * 75)

    print("KG CANDIDATE CASES")

    print("-" * 75)

    if not results:

        print(
            "\nNo corpus candidate cases "
            "were found from the KG."
        )

        return

    for rank, result in enumerate(
        results,
        start=1
    ):

        print(f"\nRank {rank}")

        print(
            f"Case Name : "
            f"{result['case_name']}"
        )

        print(
            f"Case ID   : "
            f"{result['case_id']}"
        )

        print(
            f"Court     : "
            f"{result['court']}"
        )

        print(
            f"KG Score  : "
            f"{result['kg_score']}"
        )

        print(
            "Reason    : "
            + "; ".join(result["reasons"])
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)

    print(
        "INDIAN LEGAL KNOWLEDGE GRAPH SEARCH"
    )

    print("=" * 75)

    # --------------------------------------------------------
    # Load KG
    # --------------------------------------------------------

    nodes, relationships = load_knowledge_graph()

    # --------------------------------------------------------
    # User query
    # --------------------------------------------------------

    query = input(
        "\nEnter legal query: "
    ).strip()

    if not query:

        print("\nQuery cannot be empty.")

        return

    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    results, matches = search_knowledge_graph(
        query=query,
        nodes=nodes,
        relationships=relationships,
        top_k=10
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    display_results(
        query,
        results,
        matches
    )


if __name__ == "__main__":
    main()