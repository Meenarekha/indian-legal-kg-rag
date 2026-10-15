import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "knowledge_graph"
    / "data"
    / "relationships"
    / "final_citation_review.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "knowledge_graph"
    / "data"
    / "relationships"
    / "typed_citation_edges.json"
)


def classify_relationship(context):
    text = context.lower()

    # Strongest explicit reliance language
    if "relied upon the judgment" in text or "relied on the judgment" in text:
        return "RELIES_ON"

    if "relied upon" in text or "relied on" in text:
        return "RELIES_ON"

    # Explicit compliance with another judgment
    if "compliance of the decision" in text:
        return "COMPLIES_WITH"

    if "in compliance of" in text:
        return "COMPLIES_WITH"

    # Explicit repetition / reaffirmation
    if "reiterated" in text:
        return "REITERATES"

    if "reaffirmed" in text:
        return "REAFFIRMS"

    # Explicit endorsement
    if "endorsed in" in text:
        return "ENDORSES"

    # Explicit re-emphasis
    if "re-emphasized" in text or "reemphasized" in text:
        return "REITERATES"

    # Explicit reference to a judgment
    if "decision in" in text:
        return "REFERS_TO"

    if "judgment in" in text:
        return "REFERS_TO"

    # Explicit consideration of authorities
    if "considered hosts of authorities" in text:
        return "CONSIDERS"

    if "considered" in text and "authority" in text:
        return "CONSIDERS"

    # General discussion
    if "this court" in text and "held" in text:
        return "REFERS_TO"

    return "CITES"


def main():

    print("=" * 70)
    print("CLASSIFYING FINAL CITATION RELATIONSHIPS")
    print("=" * 70)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    edges = data.get("edges", [])

    typed_edges = []

    for edge in edges:

        context = edge.get("context", "")

        relationship = classify_relationship(context)

        typed_edge = {
            "source_case_id": edge["source_case_id"],
            "source_case_name": edge["source_case_name"],
            "target_case_id": edge["target_case_id"],
            "target_case_name": edge["target_case_name"],
            "relationship": relationship,
            "classification": edge.get("classification"),
            "strong_signal_count": edge.get("strong_signal_count", 0),
            "weak_signal_count": edge.get("weak_signal_count", 0),
            "context": context
        }

        typed_edges.append(typed_edge)

    # Count relationship types
    counts = {}

    for edge in typed_edges:
        relationship = edge["relationship"]
        counts[relationship] = counts.get(relationship, 0) + 1

    print()
    print(f"Input edges : {len(edges)}")
    print()

    print("RELATIONSHIP COUNTS")
    print("-" * 70)

    for relationship, count in sorted(counts.items()):
        print(f"{relationship:<20} : {count}")

    print()
    print("CLASSIFIED EDGES")
    print("-" * 70)

    for i, edge in enumerate(typed_edges, start=1):

        print()
        print(f"[{i}]")
        print(f"Source       : {edge['source_case_name']}")
        print(f"Target       : {edge['target_case_name']}")
        print(f"Relationship : {edge['relationship']}")
        print(f"Strong       : {edge['strong_signal_count']}")
        print(f"Weak         : {edge['weak_signal_count']}")

    output = {
        "metadata": {
            "total_edges": len(typed_edges),
            "relationship_counts": counts,
            "method": (
                "Deterministic rule-based classification using "
                "explicit citation language in the judgment context."
            ),
            "legal_validation": (
                "Automated classification only; not a substitute "
                "for expert legal validation."
            )
        },
        "edges": typed_edges
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 70)
    print("TYPED CITATION GRAPH CREATED")
    print("=" * 70)
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()