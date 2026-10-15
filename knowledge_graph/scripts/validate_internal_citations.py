import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "knowledge_graph"
    / "data"
    / "relationships"
    / "internal_citation_text_report.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "knowledge_graph"
    / "data"
    / "relationships"
    / "validated_internal_citation_report.json"
)


def normalize_case_name(name):
    """
    Normalize a case name for reliable comparison.
    """

    text = name.lower()

    # Normalize common variations
    text = re.sub(r"\bversus\b", "v", text)
    text = re.sub(r"\bvs\.?\b", "v", text)

    # Remove punctuation
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def load_report():

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def main():

    print("=" * 55)
    print("VALIDATING INTERNAL CITATION RELATIONSHIPS")
    print("=" * 55)

    report = load_report()

    citations = report.get("internal_citations", [])

    print()
    print(f"Input citation edges : {len(citations)}")

    validated_edges = []

    self_citations = []

    duplicate_edges = []

    seen_edges = set()

    for citation in citations:

        source_id = citation["source_case_id"]
        target_id = citation["target_case_id"]

        source_name = citation["source_case_name"]
        target_name = citation["target_case_name"]

        # Normalize both names
        source_normalized = normalize_case_name(
            source_name
        )

        target_normalized = normalize_case_name(
            target_name
        )

        # --------------------------------------------------
        # 1. Detect self-citation
        #
        # Check BOTH:
        #   - case ID
        #   - normalized case name
        # --------------------------------------------------

        if (
            source_id == target_id
            or source_normalized == target_normalized
        ):

            self_citations.append(citation)

            continue

        # --------------------------------------------------
        # 2. Remove duplicate relationships
        # --------------------------------------------------

        edge_key = (
            source_id,
            target_id
        )

        if edge_key in seen_edges:

            duplicate_edges.append(citation)

            continue

        seen_edges.add(edge_key)

        validated_edges.append(citation)

    # ------------------------------------------------------
    # Calculate source and target counts
    # ------------------------------------------------------

    source_cases = sorted(
        set(
            citation["source_case_id"]
            for citation in validated_edges
        )
    )

    target_cases = sorted(
        set(
            citation["target_case_id"]
            for citation in validated_edges
        )
    )

    # ------------------------------------------------------
    # Results
    # ------------------------------------------------------

    print()
    print("VALIDATION RESULTS")
    print("-" * 55)

    print(
        f"Valid internal citation edges : "
        f"{len(validated_edges)}"
    )

    print(
        f"Self-citations removed        : "
        f"{len(self_citations)}"
    )

    print(
        f"Duplicate edges removed       : "
        f"{len(duplicate_edges)}"
    )

    print(
        f"Source cases                  : "
        f"{len(source_cases)}"
    )

    print(
        f"Target cases                  : "
        f"{len(target_cases)}"
    )

    # ------------------------------------------------------
    # Show removed self-citations
    # ------------------------------------------------------

    if self_citations:

        print()
        print("SELF-CITATIONS REMOVED")
        print("-" * 55)

        for citation in self_citations:

            print(
                f"{citation['source_case_name']}"
                f" -> "
                f"{citation['target_case_name']}"
            )

    # ------------------------------------------------------
    # Show validated relationships
    # ------------------------------------------------------

    print()
    print("VALIDATED INTERNAL CITATIONS")
    print("-" * 55)

    for citation in validated_edges:

        print(
            f"{citation['source_case_name']}"
        )

        print(
            f"  -> {citation['target_case_name']}"
        )

    # ------------------------------------------------------
    # Save validated report
    # ------------------------------------------------------

    output = {

        "metadata": {

            "input_citation_edges":
                len(citations),

            "validated_internal_citation_edges":
                len(validated_edges),

            "self_citations_removed":
                len(self_citations),

            "duplicate_edges_removed":
                len(duplicate_edges),

            "source_cases":
                len(source_cases),

            "target_cases":
                len(target_cases),

            "method":
                (
                    "Exact normalized corpus case-name "
                    "matching against cleaned judgment text, "
                    "followed by self-citation detection using "
                    "both case IDs and normalized case names, "
                    "and duplicate removal."
                )
        },

        "internal_citations":
            validated_edges
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
    print("REPORT")
    print("-" * 55)

    print(OUTPUT_FILE)

    print()
    print("=" * 55)
    print("VALIDATION COMPLETE")
    print("=" * 55)


if __name__ == "__main__":
    main()