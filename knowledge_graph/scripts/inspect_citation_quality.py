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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "knowledge_graph"
    / "data"
    / "relationships"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "citation_quality_report.json"
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
# GET CITATION REFERENCES
# ============================================================

def get_citations(graph):

    nodes = {
        node.get("node_id"): node
        for node in graph.get("nodes", [])
    }

    citations = []

    for relationship in graph.get(
        "relationships",
        []
    ):

        if relationship.get(
            "relationship_type"
        ) != "CITES":

            continue

        source = relationship.get("source")
        target = relationship.get("target")

        target_node = nodes.get(target)

        if not target_node:
            continue

        properties = target_node.get(
            "properties",
            {}
        )

        reference_name = (
            properties.get("case_name")
            or ""
        )

        citations.append({
            "source_case_id": source,
            "reference_node_id": target,
            "reference_name": reference_name
        })

    return citations


# ============================================================
# QUALITY CHECKS
# ============================================================

def quality_flags(name):

    flags = []

    text = name.strip()

    lower = text.lower()

    # --------------------------------------------------------
    # Too short
    # --------------------------------------------------------

    if len(text) < 12:
        flags.append("too_short")

    # --------------------------------------------------------
    # Missing case separator
    # --------------------------------------------------------

    if not re.search(
        r"\b(v\.|vs\.|versus)\b",
        text,
        re.IGNORECASE
    ):

        flags.append("missing_case_separator")

    # --------------------------------------------------------
    # Starts with obvious prose
    # --------------------------------------------------------

    prose_starts = [
        "the ",
        "of the ",
        "in the ",
        "where ",
        "that ",
        "and ",
        "from ",
        "under ",
        "before ",
        "against ",
        "by ",
        "with ",
        "often ",
        "such ",
        "this ",
        "these "
    ]

    if any(
        lower.startswith(prefix)
        for prefix in prose_starts
    ):

        flags.append("prose_start")

    # --------------------------------------------------------
    # Contains obvious extraction noise
    # --------------------------------------------------------

    noise_patterns = [
        "advocate)",
        "applicant)",
        "respondent)",
        "petitioner)",
        "paragraph",
        "judgment",
        "decision",
        "context",
        "principle",
        "present case",
        "following",
        "the court",
        "this court",
        "the supreme court",
        "the high court",
        "where the",
        "which the",
        "has been",
        "had been",
        "was held"
    ]

    for pattern in noise_patterns:

        if pattern in lower:

            flags.append(
                f"contains_noise:{pattern}"
            )

    # --------------------------------------------------------
    # Excessive length
    # --------------------------------------------------------

    if len(text) > 120:

        flags.append("too_long")

    # --------------------------------------------------------
    # Excessive word count
    # --------------------------------------------------------

    word_count = len(
        text.split()
    )

    if word_count > 18:

        flags.append("too_many_words")

    # --------------------------------------------------------
    # Unusual punctuation
    # --------------------------------------------------------

    if re.search(
        r"[\(\)\[\]]",
        text
    ):

        flags.append("contains_brackets")

    return flags


# ============================================================
# CLASSIFY
# ============================================================

def classify_citations(citations):

    clean = []
    noisy = []

    for citation in citations:

        flags = quality_flags(
            citation["reference_name"]
        )

        item = {
            **citation,
            "quality_flags": flags
        }

        if flags:

            noisy.append(item)

        else:

            clean.append(item)

    return clean, noisy


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("==========================================")
    print("CITATION QUALITY INSPECTION")
    print("==========================================")
    print()

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

    citations = get_citations(graph)

    print(
        f"Total CITES references : "
        f"{len(citations)}"
    )

    clean, noisy = classify_citations(
        citations
    )

    print()
    print("QUALITY RESULTS")
    print("----------------")

    print(
        f"Potentially clean : "
        f"{len(clean)}"
    )

    print(
        f"Potentially noisy : "
        f"{len(noisy)}"
    )

    total = len(citations)

    if total:

        clean_percentage = (
            len(clean) / total
        ) * 100

        noisy_percentage = (
            len(noisy) / total
        ) * 100

    else:

        clean_percentage = 0
        noisy_percentage = 0

    print(
        f"Clean percentage  : "
        f"{clean_percentage:.2f}%"
    )

    print(
        f"Noisy percentage  : "
        f"{noisy_percentage:.2f}%"
    )

    # --------------------------------------------------------
    # Show examples
    # --------------------------------------------------------

    print()
    print("POTENTIALLY CLEAN EXAMPLES")
    print("---------------------------")

    for item in clean[:20]:

        print(
            item["reference_name"]
        )

    print()
    print("POTENTIALLY NOISY EXAMPLES")
    print("---------------------------")

    for item in noisy[:30]:

        print(
            item["reference_name"]
        )

        print(
            "  Flags:",
            ", ".join(
                item["quality_flags"]
            )
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
            "total_citations":
                total,

            "potentially_clean":
                len(clean),

            "potentially_noisy":
                len(noisy),

            "clean_percentage":
                clean_percentage,

            "noisy_percentage":
                noisy_percentage
        },

        "clean": clean,

        "noisy": noisy
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

    print()
    print("REPORT")
    print("----------------")

    print(OUTPUT_FILE)

    print()
    print("==========================================")
    print()


if __name__ == "__main__":
    main()