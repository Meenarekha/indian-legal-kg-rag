import json
from pathlib import Path
from collections import Counter


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "knowledge_graph"
    / "data"
    / "relationships"
    / "typed_citation_edges.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "knowledge_graph"
    / "data"
    / "relationships"
    / "validated_typed_citation_edges.json"
)


VALID_RELATIONSHIPS = {
    "CITES",
    "COMPLIES_WITH",
    "CONSIDERS",
    "ENDORSES",
    "REFERS_TO",
    "REITERATES",
    "RELIES_ON"
}


def normalize_name(name):
    return " ".join(name.lower().split())


def main():

    print("=" * 70)
    print("VALIDATING TYPED CITATION GRAPH")
    print("=" * 70)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    edges = data.get("edges", [])

    print()
    print(f"Input edges : {len(edges)}")

    valid_edges = []
    invalid_edges = []

    seen = set()

    self_citations = 0
    duplicates = 0
    invalid_relationships = 0
    missing_context = 0
    missing_classification = 0

    for edge in edges:

        source_id = edge.get("source_case_id", "")
        target_id = edge.get("target_case_id", "")

        source_name = edge.get("source_case_name", "")
        target_name = edge.get("target_case_name", "")

        relationship = edge.get("relationship", "")
        context = edge.get("context", "")
        classification = edge.get("classification", "")

        errors = []

        # ---------------------------------------------------------
        # Self-citation check
        # ---------------------------------------------------------

        same_id = source_id == target_id

        same_name = (
            normalize_name(source_name)
            == normalize_name(target_name)
        )

        if same_id or same_name:
            self_citations += 1
            errors.append("self_citation")

        # ---------------------------------------------------------
        # Duplicate check
        # ---------------------------------------------------------

        edge_key = (
            source_id,
            target_id,
            relationship
        )

        if edge_key in seen:
            duplicates += 1
            errors.append("duplicate")

        seen.add(edge_key)

        # ---------------------------------------------------------
        # Relationship validation
        # ---------------------------------------------------------

        if relationship not in VALID_RELATIONSHIPS:
            invalid_relationships += 1
            errors.append("invalid_relationship")

        # ---------------------------------------------------------
        # Evidence validation
        # ---------------------------------------------------------

        if not context.strip():
            missing_context += 1
            errors.append("missing_context")

        if not classification.strip():
            missing_classification += 1
            errors.append("missing_classification")

        # ---------------------------------------------------------
        # Final decision
        # ---------------------------------------------------------

        if errors:
            invalid_edges.append({
                "edge": edge,
                "errors": errors
            })
        else:
            valid_edges.append(edge)

    # -------------------------------------------------------------
    # Relationship counts
    # -------------------------------------------------------------

    relationship_counts = Counter(
        edge["relationship"]
        for edge in valid_edges
    )

    # -------------------------------------------------------------
    # Output
    # -------------------------------------------------------------

    output = {
        "metadata": {
            "input_edges": len(edges),
            "valid_edges": len(valid_edges),
            "invalid_edges": len(invalid_edges),
            "self_citations": self_citations,
            "duplicates": duplicates,
            "invalid_relationships": invalid_relationships,
            "missing_context": missing_context,
            "missing_classification": missing_classification,
            "relationship_counts": dict(relationship_counts),
            "validation_status": (
                "PASS"
                if len(invalid_edges) == 0
                else "REVIEW_REQUIRED"
            )
        },
        "edges": valid_edges,
        "invalid_edges": invalid_edges
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False
        )

    # -------------------------------------------------------------
    # Terminal output
    # -------------------------------------------------------------

    print()
    print("VALIDATION RESULTS")
    print("-" * 70)

    print(f"Valid edges            : {len(valid_edges)}")
    print(f"Invalid edges          : {len(invalid_edges)}")
    print(f"Self-citations         : {self_citations}")
    print(f"Duplicates             : {duplicates}")
    print(f"Invalid relationships  : {invalid_relationships}")
    print(f"Missing context        : {missing_context}")
    print(f"Missing classification : {missing_classification}")

    print()
    print("VALID RELATIONSHIP COUNTS")
    print("-" * 70)

    for relationship, count in sorted(
        relationship_counts.items()
    ):
        print(
            f"{relationship:<20} : {count}"
        )

    print()
    print("=" * 70)

    if len(invalid_edges) == 0:
        print("VALIDATION PASS")
    else:
        print("VALIDATION REQUIRES REVIEW")

    print("=" * 70)

    print()
    print("Output:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()