import json
from pathlib import Path
from collections import Counter


PROJECT_ROOT = Path(__file__).resolve().parents[2]

NODES_FILE = (
    PROJECT_ROOT
    / "knowledge_graph"
    / "data"
    / "nodes"
    / "all_nodes.json"
)

RELATIONSHIPS_FILE = (
    PROJECT_ROOT
    / "knowledge_graph"
    / "data"
    / "relationships"
    / "validated_typed_citation_edges.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "knowledge_graph"
    / "data"
    / "knowledge_graph.json"
)


def main():

    print("=" * 70)
    print("BUILDING FINAL KNOWLEDGE GRAPH")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load nodes
    # ---------------------------------------------------------

    with open(
        NODES_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        node_data = json.load(f)

    nodes = node_data.get(
        "nodes",
        []
    )

    print()
    print(
        f"Existing nodes : {len(nodes)}"
    )

    # ---------------------------------------------------------
    # Load validated typed citation edges
    # ---------------------------------------------------------

    with open(
        RELATIONSHIPS_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        relationship_data = json.load(f)

    citation_edges = relationship_data.get(
        "edges",
        []
    )

    print(
        f"Typed citation edges : "
        f"{len(citation_edges)}"
    )

    # ---------------------------------------------------------
    # Existing relationships
    # ---------------------------------------------------------

    existing_relationships = node_data.get(
        "relationships",
        []
    )

    print(
        f"Existing relationships : "
        f"{len(existing_relationships)}"
    )

    # ---------------------------------------------------------
    # Convert typed citation edges
    # into KG relationship format
    # ---------------------------------------------------------

    final_relationships = list(
        existing_relationships
    )

    for edge in citation_edges:

        relationship = {
            "relationship_type": edge[
                "relationship"
            ],

            "source": edge[
                "source_case_id"
            ],

            "target": edge[
                "target_case_id"
            ],

            "evidence": {
                "source_case_name": edge[
                    "source_case_name"
                ],

                "target_case_name": edge[
                    "target_case_name"
                ],

                "classification": edge.get(
                    "classification"
                ),

                "strong_signal_count": edge.get(
                    "strong_signal_count",
                    0
                ),

                "weak_signal_count": edge.get(
                    "weak_signal_count",
                    0
                ),

                "context": edge.get(
                    "context",
                    ""
                )
            }
        }

        final_relationships.append(
            relationship
        )

    # ---------------------------------------------------------
    # Remove exact duplicate relationships
    # ---------------------------------------------------------

    unique_relationships = []

    seen = set()

    for relationship in final_relationships:

        key = (
            relationship.get(
                "relationship_type"
            ),

            relationship.get(
                "source"
            ),

            relationship.get(
                "target"
            )
        )

        if key not in seen:

            seen.add(key)

            unique_relationships.append(
                relationship
            )

    # ---------------------------------------------------------
    # Relationship statistics
    # ---------------------------------------------------------

    relationship_counts = Counter(
        relationship.get(
            "relationship_type"
        )
        for relationship in unique_relationships
    )

    # ---------------------------------------------------------
    # Node statistics
    # ---------------------------------------------------------

    node_counts = Counter(
        node.get(
            "node_type"
        )
        for node in nodes
    )

    # ---------------------------------------------------------
    # Final graph
    # ---------------------------------------------------------

    graph = {

        "metadata": {

            "description": (
                "Indian legal case Knowledge Graph "
                "constructed from the project corpus."
            ),

            "corpus_size": 205,

            "node_count": len(
                nodes
            ),

            "relationship_count": len(
                unique_relationships
            ),

            "citation_relationship_count": len(
                citation_edges
            ),

            "node_type_counts": dict(
                node_counts
            ),

            "relationship_type_counts": dict(
                relationship_counts
            ),

            "citation_validation": (
                "Validated typed citation edges"
            )
        },

        "nodes": nodes,

        "relationships": unique_relationships
    }

    # ---------------------------------------------------------
    # Save final graph
    # ---------------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            graph,
            f,
            indent=2,
            ensure_ascii=False
        )

    # ---------------------------------------------------------
    # Terminal output
    # ---------------------------------------------------------

    print()
    print(
        "FINAL GRAPH STATISTICS"
    )

    print("-" * 70)

    print(
        f"Nodes         : "
        f"{len(nodes)}"
    )

    print(
        f"Relationships : "
        f"{len(unique_relationships)}"
    )

    # ---------------------------------------------------------
    # Node types
    # ---------------------------------------------------------

    print()
    print("NODE TYPES")
    print("-" * 70)

    for node_type, count in sorted(
        node_counts.items(),
        key=lambda item: str(
            item[0]
        )
    ):

        node_type_label = (
            str(node_type)
            if node_type is not None
            else "UNKNOWN"
        )

        print(
            f"{node_type_label:<20} : "
            f"{count}"
        )

    # ---------------------------------------------------------
    # Relationship types
    # ---------------------------------------------------------

    print()
    print(
        "RELATIONSHIP TYPES"
    )

    print("-" * 70)

    for relationship_type, count in sorted(
        relationship_counts.items(),
        key=lambda item: str(
            item[0]
        )
    ):

        relationship_type_label = (
            str(relationship_type)
            if relationship_type is not None
            else "UNKNOWN"
        )

        print(
            f"{relationship_type_label:<20} : "
            f"{count}"
        )

    # ---------------------------------------------------------
    # Final message
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "FINAL KNOWLEDGE GRAPH CREATED"
    )
    print("=" * 70)

    print()
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()