import json
import re
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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "knowledge_graph"
    / "data"
    / "relationships"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "citation_resolution_report.json"
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
# NORMALIZE CASE NAME
# ============================================================

def normalize_case_name(name):

    if not name:
        return ""

    text = name.lower()

    # --------------------------------------------------------
    # Remove file/date information
    # --------------------------------------------------------

    text = re.sub(
        r"_on_\d{1,2}_[a-z]+_\d{4}",
        "",
        text
    )

    # --------------------------------------------------------
    # Convert common case separators
    # --------------------------------------------------------

    text = re.sub(
        r"\bversus\b",
        " v ",
        text
    )

    text = re.sub(
        r"\bvs\.?\b",
        " v ",
        text
    )

    # --------------------------------------------------------
    # Remove common citation suffixes
    # --------------------------------------------------------

    text = re.sub(
        r"\b(?:air|scc|scr|online)\b",
        " ",
        text
    )

    # --------------------------------------------------------
    # Remove punctuation
    # --------------------------------------------------------

    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text
    )

    # --------------------------------------------------------
    # Normalize whitespace
    # --------------------------------------------------------

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# LOAD ORIGINAL CORPUS CASES
# ============================================================

def load_corpus_cases():

    cases = []

    for case_file in sorted(
        CLEANED_DIR.glob("*.json")
    ):

        try:

            with open(
                case_file,
                "r",
                encoding="utf-8"
            ) as f:

                case_data = json.load(f)

            case_id = case_data.get(
                "case_id"
            )

            if not case_id:
                continue

            # Derive case name from case ID.
            if "_on_" in case_id:

                case_name = (
                    case_id
                    .rsplit("_on_", 1)[0]
                    .replace("_", " ")
                )

            else:

                case_name = (
                    case_id
                    .replace("_", " ")
                )

            cases.append({
                "case_id": case_id,
                "case_name": case_name,
                "normalized_name":
                    normalize_case_name(
                        case_name
                    ),
                "source_file":
                    case_data.get(
                        "source_file"
                    )
            })

        except Exception:

            continue

    return cases


# ============================================================
# BUILD NORMALIZED CORPUS INDEX
# ============================================================

def build_corpus_index(corpus_cases):

    index = {}

    duplicate_normalized_names = {}

    for case in corpus_cases:

        normalized = case[
            "normalized_name"
        ]

        if not normalized:
            continue

        if normalized in index:

            duplicate_normalized_names.setdefault(
                normalized,
                []
            ).append(case)

        else:

            index[normalized] = case

    return (
        index,
        duplicate_normalized_names
    )


# ============================================================
# EXTRACT CITATION REFERENCES
# ============================================================

def get_citation_references(graph):

    references = []

    for relationship in graph.get(
        "relationships",
        []
    ):

        if relationship.get(
            "relationship_type"
        ) != "CITES":

            continue

        target = relationship.get(
            "target"
        )

        if not target:
            continue

        # Find corresponding reference node.
        for node in graph.get(
            "nodes",
            []
        ):

            if (
                node.get("node_id")
                == target
                and node.get("node_type")
                == "Case"
            ):

                properties = node.get(
                    "properties",
                    {}
                )

                if properties.get(
                    "reference_only"
                ):

                    references.append({
                        "source_case_id":
                            relationship.get(
                                "source"
                            ),
                        "reference_node_id":
                            target,
                        "reference_name":
                            properties.get(
                                "case_name"
                            )
                    })

                break

    return references


# ============================================================
# RESOLVE REFERENCES
# ============================================================

def resolve_references(
    references,
    corpus_index
):

    resolved = []
    unresolved = []

    for reference in references:

        reference_name = (
            reference.get(
                "reference_name"
            )
            or ""
        )

        normalized = normalize_case_name(
            reference_name
        )

        match = corpus_index.get(
            normalized
        )

        if match:

            resolved.append({
                "source_case_id":
                    reference[
                        "source_case_id"
                    ],
                "reference_name":
                    reference_name,
                "normalized_reference":
                    normalized,
                "matched_case_id":
                    match["case_id"],
                "matched_case_name":
                    match["case_name"],
                "match_type":
                    "exact_normalized"
            })

        else:

            unresolved.append({
                "source_case_id":
                    reference[
                        "source_case_id"
                    ],
                "reference_name":
                    reference_name,
                "normalized_reference":
                    normalized
            })

    return (
        resolved,
        unresolved
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("==========================================")
    print("CITATION ENTITY RESOLUTION")
    print("==========================================")
    print()


    # --------------------------------------------------------
    # Load graph
    # --------------------------------------------------------

    if not GRAPH_FILE.exists():

        print(
            "[ERROR] Graph file not found:"
        )

        print(GRAPH_FILE)

        return


    graph = load_graph()

    print(
        "[OK] Graph loaded successfully"
    )


    # --------------------------------------------------------
    # Load corpus
    # --------------------------------------------------------

    corpus_cases = load_corpus_cases()

    print(
        f"Corpus cases loaded : "
        f"{len(corpus_cases)}"
    )


    # --------------------------------------------------------
    # Build index
    # --------------------------------------------------------

    (
        corpus_index,
        duplicate_names
    ) = build_corpus_index(
        corpus_cases
    )

    print(
        f"Unique normalized names : "
        f"{len(corpus_index)}"
    )

    print(
        f"Duplicate normalized names : "
        f"{len(duplicate_names)}"
    )


    # --------------------------------------------------------
    # Get citations
    # --------------------------------------------------------

    references = get_citation_references(
        graph
    )

    print(
        f"Citation references : "
        f"{len(references)}"
    )


    # --------------------------------------------------------
    # Resolve
    # --------------------------------------------------------

    (
        resolved,
        unresolved
    ) = resolve_references(
        references,
        corpus_index
    )


    print()
    print("RESOLUTION RESULTS")
    print("----------------")

    print(
        f"Resolved exactly : "
        f"{len(resolved)}"
    )

    print(
        f"Unresolved        : "
        f"{len(unresolved)}"
    )


    total = len(references)

    if total:

        percentage = (
            len(resolved)
            / total
        ) * 100

    else:

        percentage = 0


    print(
        f"Resolution rate   : "
        f"{percentage:.2f}%"
    )


    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    report = {

        "metadata": {
            "corpus_cases":
                len(corpus_cases),

            "citation_references":
                total,

            "resolved_exact_normalized":
                len(resolved),

            "unresolved":
                len(unresolved),

            "resolution_rate_percent":
                percentage
        },

        "resolved": resolved,

        "unresolved": unresolved,

        "duplicate_normalized_names":
            duplicate_names
    }


    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False
        )


    # --------------------------------------------------------
    # Show examples
    # --------------------------------------------------------

    print()
    print("RESOLVED EXAMPLES")
    print("----------------")

    for item in resolved[:15]:

        print(
            f"{item['reference_name']}"
        )

        print(
            f"  -> "
            f"{item['matched_case_name']}"
        )

        print()


    print("UNRESOLVED EXAMPLES")
    print("----------------")

    for item in unresolved[:15]:

        print(
            item["reference_name"]
        )


    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    print()
    print("REPORT")
    print("----------------")

    print(OUTPUT_FILE)

    print()
    print("==========================================")
    print()


if __name__ == "__main__":
    main()