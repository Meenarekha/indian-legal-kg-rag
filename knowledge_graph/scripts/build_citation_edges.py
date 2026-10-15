import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "knowledge_graph"
    / "data"
    / "relationships"
    / "classified_citation_contexts.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "knowledge_graph"
    / "data"
    / "relationships"
    / "final_citation_edges.json"
)


def main():

    print("=" * 70)
    print("BUILDING HIGH-CONFIDENCE CITATION EDGES")
    print("=" * 70)

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        report = json.load(f)

    citations = report.get(
        "internal_citations",
        []
    )

    strong_edges = []
    weak_references = []
    uncertain_references = []

    for citation in citations:

        classification = citation.get(
            "classification"
        )

        if classification == "strong_candidate":

            strong_edges.append(citation)

        elif classification == "weak_candidate":

            weak_references.append(citation)

        elif classification == "uncertain":

            uncertain_references.append(citation)

    output = {

        "metadata": {

            "input_candidates":
                len(citations),

            "high_confidence_cites_edges":
                len(strong_edges),

            "weak_references":
                len(weak_references),

            "uncertain_references":
                len(uncertain_references),

            "policy":
                (
                    "Only strong citation-context candidates "
                    "are promoted to CASE -> CITES -> CASE "
                    "relationships. Weak and uncertain "
                    "references are retained separately and "
                    "are not treated as graph citation edges."
                )
        },

        "cites_edges":
            strong_edges,

        "weak_references":
            weak_references,

        "uncertain_references":
            uncertain_references
    }

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False
        )

    print()
    print("RESULTS")
    print("-" * 70)

    print(
        f"Input candidates       : "
        f"{len(citations)}"
    )

    print(
        f"High-confidence edges  : "
        f"{len(strong_edges)}"
    )

    print(
        f"Weak references        : "
        f"{len(weak_references)}"
    )

    print(
        f"Uncertain references   : "
        f"{len(uncertain_references)}"
    )

    print()
    print("HIGH-CONFIDENCE CITES EDGES")
    print("-" * 70)

    for edge in strong_edges:

        print(
            f"{edge['source_case_name']}"
        )

        print(
            f"  -> {edge['target_case_name']}"
        )

    print()
    print("OUTPUT")
    print("-" * 70)

    print(OUTPUT_FILE)

    print()
    print("=" * 70)
    print("CITATION EDGE BUILD COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()