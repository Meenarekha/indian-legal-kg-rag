import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "knowledge_graph"
    / "data"
    / "relationships"
    / "final_citation_edges.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "knowledge_graph"
    / "data"
    / "relationships"
    / "final_citation_review.json"
)


def main():

    print("=" * 70)
    print("FINAL REVIEW OF HIGH-CONFIDENCE CITATION EDGES")
    print("=" * 70)

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    edges = data.get("cites_edges", [])

    print()
    print(f"High-confidence edges : {len(edges)}")

    print()
    print("FINAL CITES EDGES")
    print("-" * 70)

    for i, edge in enumerate(edges, start=1):

        print()
        print(f"[{i}]")
        print(f"Source : {edge['source_case_name']}")
        print(f"Target : {edge['target_case_name']}")

        print(
            f"Classification : "
            f"{edge.get('classification', 'N/A')}"
        )

        print(
            f"Strong signals : "
            f"{edge.get('strong_signal_count', 0)}"
        )

        print(
            f"Weak signals   : "
            f"{edge.get('weak_signal_count', 0)}"
        )

        print(
            f"Context        : "
            f"{edge.get('context', '')}"
        )

    # ------------------------------------------------------
    # Create a review report
    # ------------------------------------------------------

    review = {

        "metadata": {

            "total_high_confidence_edges":
                len(edges),

            "review_status":
                "ready_for_graph_integration",

            "note":
                (
                    "Edges were promoted after exact "
                    "corpus-name matching, self-citation "
                    "filtering, and deterministic citation "
                    "context classification."
                )
        },

        "edges": edges
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            review,
            f,
            indent=2,
            ensure_ascii=False
        )

    print()
    print("=" * 70)
    print("REVIEW REPORT CREATED")
    print("=" * 70)

    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()