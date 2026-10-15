import json
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GRAPH_FILE = (
    PROJECT_ROOT
    / "knowledge_graph"
    / "data"
    / "nodes"
    / "all_nodes.json"
)

CLEANED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cleaned"
)


# ============================================================
# LOAD GRAPH
# ============================================================

def load_graph():

    with open(
        GRAPH_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# ============================================================
# GET ORIGINAL CORPUS CASE IDs
# ============================================================

def get_corpus_case_ids():

    case_ids = set()

    for case_file in CLEANED_DIR.glob("*.json"):

        try:

            with open(
                case_file,
                "r",
                encoding="utf-8"
            ) as f:

                case_data = json.load(f)

            case_id = case_data.get("case_id")

            if case_id:
                case_ids.add(case_id)

        except Exception:

            continue

    return case_ids


# ============================================================
# VALIDATE NODE STRUCTURE
# ============================================================

def validate_nodes(nodes):

    errors = []

    node_ids = set()

    duplicate_ids = []

    for index, node in enumerate(nodes):

        if not isinstance(node, dict):

            errors.append(
                f"Node {index} is not a dictionary"
            )

            continue

        required_fields = [
            "node_type",
            "node_id",
            "properties"
        ]

        for field in required_fields:

            if field not in node:

                errors.append(
                    f"Node {index} missing field: {field}"
                )

        node_id = node.get("node_id")

        if node_id:

            if node_id in node_ids:

                duplicate_ids.append(node_id)

            node_ids.add(node_id)


    return errors, node_ids, duplicate_ids


# ============================================================
# VALIDATE RELATIONSHIPS
# ============================================================

def validate_relationships(
    relationships,
    node_ids
):

    errors = []

    duplicate_relationships = []

    relationship_keys = set()

    for index, relationship in enumerate(
        relationships
    ):

        if not isinstance(
            relationship,
            dict
        ):

            errors.append(
                f"Relationship {index} "
                f"is not a dictionary"
            )

            continue


        required_fields = [
            "relationship_type",
            "source",
            "target"
        ]


        for field in required_fields:

            if field not in relationship:

                errors.append(
                    f"Relationship {index} "
                    f"missing field: {field}"
                )


        source = relationship.get(
            "source"
        )

        target = relationship.get(
            "target"
        )

        relationship_type = relationship.get(
            "relationship_type"
        )


        # Check source node
        if source and source not in node_ids:

            errors.append(
                f"Relationship {index}: "
                f"source node does not exist: "
                f"{source}"
            )


        # Check target node
        if target and target not in node_ids:

            errors.append(
                f"Relationship {index}: "
                f"target node does not exist: "
                f"{target}"
            )


        # Check duplicates
        key = (
            relationship_type,
            source,
            target
        )


        if key in relationship_keys:

            duplicate_relationships.append(
                key
            )

        relationship_keys.add(key)


    return (
        errors,
        duplicate_relationships
    )


# ============================================================
# NODE TYPE COUNTS
# ============================================================

def count_node_types(nodes):

    counts = {}

    for node in nodes:

        node_type = node.get(
            "node_type",
            "UNKNOWN"
        )

        counts[node_type] = (
            counts.get(node_type, 0) + 1
        )

    return counts


# ============================================================
# RELATIONSHIP COUNTS
# ============================================================

def count_relationship_types(
    relationships
):

    counts = {}

    for relationship in relationships:

        relationship_type = relationship.get(
            "relationship_type",
            "UNKNOWN"
        )

        counts[relationship_type] = (
            counts.get(
                relationship_type,
                0
            ) + 1
        )

    return counts


# ============================================================
# CITATION COVERAGE
# ============================================================

def calculate_citation_coverage(
    relationships,
    corpus_case_ids
):

    citation_relationships = []

    internal_citations = []

    external_citations = []


    for relationship in relationships:

        if relationship.get(
            "relationship_type"
        ) != "CITES":

            continue


        source = relationship.get(
            "source"
        )

        target = relationship.get(
            "target"
        )


        citation_relationships.append(
            relationship
        )


        if target in corpus_case_ids:

            internal_citations.append(
                relationship
            )

        else:

            external_citations.append(
                relationship
            )


    total = len(
        citation_relationships
    )

    internal = len(
        internal_citations
    )

    external = len(
        external_citations
    )


    if total > 0:

        internal_percentage = (
            internal / total
        ) * 100

        external_percentage = (
            external / total
        ) * 100

    else:

        internal_percentage = 0
        external_percentage = 0


    return {
        "total": total,
        "internal": internal,
        "external": external,
        "internal_percentage":
            internal_percentage,
        "external_percentage":
            external_percentage
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("==========================================")
    print("KNOWLEDGE GRAPH VALIDATION")
    print("==========================================")
    print()


    # --------------------------------------------------------
    # Check graph file
    # --------------------------------------------------------

    if not GRAPH_FILE.exists():

        print("[ERROR] Graph file not found:")
        print(GRAPH_FILE)
        return


    print(
        f"Graph file : {GRAPH_FILE}"
    )

    print()


    # --------------------------------------------------------
    # Load graph
    # --------------------------------------------------------

    try:

        graph = load_graph()

    except Exception as error:

        print(
            "[ERROR] Could not load graph:"
        )

        print(error)

        return


    nodes = graph.get(
        "nodes",
        []
    )

    relationships = graph.get(
        "relationships",
        []
    )


    print(
        "[OK] JSON file loaded successfully"
    )

    print()


    # --------------------------------------------------------
    # Validate nodes
    # --------------------------------------------------------

    node_errors, node_ids, duplicate_nodes = (
        validate_nodes(nodes)
    )


    print("NODE VALIDATION")
    print("----------------")

    print(
        f"Total nodes      : {len(nodes)}"
    )

    print(
        f"Unique node IDs  : {len(node_ids)}"
    )

    print(
        f"Duplicate IDs    : {len(duplicate_nodes)}"
    )

    print(
        f"Node errors      : {len(node_errors)}"
    )


    # --------------------------------------------------------
    # Validate relationships
    # --------------------------------------------------------

    (
        relationship_errors,
        duplicate_relationships
    ) = validate_relationships(
        relationships,
        node_ids
    )


    print()
    print("RELATIONSHIP VALIDATION")
    print("----------------")

    print(
        f"Total relationships : "
        f"{len(relationships)}"
    )

    print(
        f"Duplicate relations : "
        f"{len(duplicate_relationships)}"
    )

    print(
        f"Relationship errors  : "
        f"{len(relationship_errors)}"
    )


    # --------------------------------------------------------
    # Node type counts
    # --------------------------------------------------------

    node_counts = count_node_types(
        nodes
    )


    print()
    print("NODE TYPES")
    print("----------------")

    for node_type, count in sorted(
        node_counts.items()
    ):

        print(
            f"{node_type:15s}: {count}"
        )


    # --------------------------------------------------------
    # Relationship counts
    # --------------------------------------------------------

    relationship_counts = (
        count_relationship_types(
            relationships
        )
    )


    print()
    print("RELATIONSHIP TYPES")
    print("----------------")

    for relationship_type, count in sorted(
        relationship_counts.items()
    ):

        print(
            f"{relationship_type:20s}: "
            f"{count}"
        )


    # --------------------------------------------------------
    # Original corpus
    # --------------------------------------------------------

    corpus_case_ids = (
        get_corpus_case_ids()
    )


    print()
    print("ORIGINAL CORPUS")
    print("----------------")

    print(
        f"Corpus judgments : "
        f"{len(corpus_case_ids)}"
    )


    # --------------------------------------------------------
    # Citation coverage
    # --------------------------------------------------------

    citation_stats = (
        calculate_citation_coverage(
            relationships,
            corpus_case_ids
        )
    )


    print()
    print("CITATION COVERAGE")
    print("----------------")

    print(
        f"Total CITES relationships : "
        f"{citation_stats['total']}"
    )

    print(
        f"Internal citations        : "
        f"{citation_stats['internal']}"
    )

    print(
        f"External citations        : "
        f"{citation_stats['external']}"
    )

    print(
        f"Internal coverage         : "
        f"{citation_stats['internal_percentage']:.2f}%"
    )

    print(
        f"External coverage         : "
        f"{citation_stats['external_percentage']:.2f}%"
    )


    # --------------------------------------------------------
    # Show first errors
    # --------------------------------------------------------

    all_errors = (
        node_errors
        + relationship_errors
    )


    if all_errors:

        print()
        print("FIRST VALIDATION ERRORS")
        print("----------------")

        for error in all_errors[:20]:

            print(
                f"- {error}"
            )

    else:

        print()
        print("[OK] No structural validation errors found.")


    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    print()
    print("==========================================")

    if not all_errors:

        print(
            "VALIDATION STATUS : PASS"
        )

    else:

        print(
            "VALIDATION STATUS : REVIEW REQUIRED"
        )

    print("==========================================")
    print()


if __name__ == "__main__":
    main()