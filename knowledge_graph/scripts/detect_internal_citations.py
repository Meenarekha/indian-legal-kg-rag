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
    / "internal_citation_report.json"
)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_name(name):

    if not name:
        return ""

    text = name.lower()

    # --------------------------------------------------------
    # Remove date suffix from corpus case IDs
    # --------------------------------------------------------

    text = re.sub(
        r"\s+on\s+\d{1,2}"
        r"(?:st|nd|rd|th)?"
        r"(?:\s+[a-z]+)?"
        r"(?:\s+\d{4})?$",
        "",
        text
    )

    # --------------------------------------------------------
    # Normalize case separators
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
    # Remove common legal suffixes
    # --------------------------------------------------------

    text = re.sub(
        r"\b(and others|and ors|ors|others)\b",
        " ",
        text
    )

    # --------------------------------------------------------
    # Remove reporting citations
    # --------------------------------------------------------

    text = re.sub(
        r"\(\s*\d{4}.*?\)",
        " ",
        text
    )

    text = re.sub(
        r"\b\d{4}\s+\d+.*$",
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
# LOAD CORPUS CASES
# ============================================================

def load_corpus_cases():

    cases = []

    for path in sorted(
        CLEANED_DIR.glob("*.json")
    ):

        try:

            with open(
                path,
                "r",
                encoding="utf-8"
            ) as f:

                data = json.load(f)

            case_id = data.get(
                "case_id"
            )

            if not case_id:
                continue

            # ------------------------------------------------
            # Convert case ID to readable case name
            # ------------------------------------------------

            if "_on_" in case_id:

                case_name = (
                    case_id
                    .rsplit("_on_", 1)[0]
                    .replace("_", " ")
                )

            else:

                case_name = case_id.replace(
                    "_",
                    " "
                )

            cases.append({
                "case_id": case_id,
                "case_name": case_name,
                "normalized_name":
                    normalize_name(
                        case_name
                    )
            })

        except Exception:
            continue

    return cases


# ============================================================
# LOAD EXISTING CITATION REFERENCES
# ============================================================

def load_citations():

    with open(
        GRAPH_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        graph = json.load(f)

    nodes = {
        node.get("node_id"): node
        for node in graph.get(
            "nodes",
            []
        )
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

        source = relationship.get(
            "source"
        )

        target = relationship.get(
            "target"
        )

        target_node = nodes.get(
            target
        )

        if not target_node:
            continue

        properties = target_node.get(
            "properties",
            {}
        )

        reference_name = properties.get(
            "case_name"
        )

        if not reference_name:
            continue

        citations.append({
            "source_case_id":
                source,
            "reference_node_id":
                target,
            "reference_name":
                reference_name
        })

    return citations


# ============================================================
# SPLIT CASE NAME
# ============================================================

def split_case_name(name):

    normalized = normalize_name(
        name
    )

    parts = re.split(
        r"\s+v\s+",
        normalized
    )

    if len(parts) != 2:
        return None

    petitioner = parts[0].strip()
    respondent = parts[1].strip()

    if not petitioner or not respondent:
        return None

    return (
        petitioner,
        respondent
    )


# ============================================================
# PARTY TOKENIZATION
# ============================================================

def party_tokens(text):

    if not text:
        return set()

    tokens = set()

    for token in text.split():

        token = token.strip()

        if len(token) >= 3:
            tokens.add(token)

    return tokens


# ============================================================
# STRICT PARTY MATCH
# ============================================================

def party_match(
    citation_party,
    corpus_party
):

    citation_tokens = party_tokens(
        citation_party
    )

    corpus_tokens = party_tokens(
        corpus_party
    )

    if not citation_tokens:
        return False

    if not corpus_tokens:
        return False

    # Every meaningful token in the shorter side
    # must occur in the other side.

    common = (
        citation_tokens
        & corpus_tokens
    )

    smaller_size = min(
        len(citation_tokens),
        len(corpus_tokens)
    )

    if smaller_size == 0:
        return False

    overlap = (
        len(common)
        / smaller_size
    )

    return overlap >= 0.90


# ============================================================
# STRICT CASE MATCH
# ============================================================

def strict_case_match(
    citation_name,
    corpus_case
):

    citation_parts = split_case_name(
        citation_name
    )

    corpus_parts = split_case_name(
        corpus_case["case_name"]
    )

    if not citation_parts:
        return False

    if not corpus_parts:
        return False

    citation_petitioner, citation_respondent = (
        citation_parts
    )

    corpus_petitioner, corpus_respondent = (
        corpus_parts
    )

    # BOTH sides must match strongly.
    petitioner_ok = party_match(
        citation_petitioner,
        corpus_petitioner
    )

    respondent_ok = party_match(
        citation_respondent,
        corpus_respondent
    )

    return (
        petitioner_ok
        and respondent_ok
    )


# ============================================================
# FIND INTERNAL CITATIONS
# ============================================================

def find_internal_citations(
    citations,
    corpus_cases
):

    internal = []
    external = []
    ambiguous = []

    for citation in citations:

        matches = []

        for corpus_case in corpus_cases:

            if strict_case_match(
                citation["reference_name"],
                corpus_case
            ):

                matches.append(
                    corpus_case
                )

        # ----------------------------------------------------
        # Exactly one match
        # ----------------------------------------------------

        if len(matches) == 1:

            match = matches[0]

            # Prevent self-citation.
            if (
                citation["source_case_id"]
                == match["case_id"]
            ):
                continue

            internal.append({
                "source_case_id":
                    citation[
                        "source_case_id"
                    ],

                "reference_name":
                    citation[
                        "reference_name"
                    ],

                "target_case_id":
                    match[
                        "case_id"
                    ],

                "target_case_name":
                    match[
                        "case_name"
                    ],

                "match_type":
                    "strict_internal"
            })

        # ----------------------------------------------------
        # Multiple matches
        # ----------------------------------------------------

        elif len(matches) > 1:

            ambiguous.append({
                "source_case_id":
                    citation[
                        "source_case_id"
                    ],

                "reference_name":
                    citation[
                        "reference_name"
                    ],

                "possible_matches": [
                    {
                        "case_id":
                            case["case_id"],

                        "case_name":
                            case["case_name"]
                    }

                    for case in matches
                ]
            })

        # ----------------------------------------------------
        # No match
        # ----------------------------------------------------

        else:

            external.append({
                "source_case_id":
                    citation[
                        "source_case_id"
                    ],

                "reference_name":
                    citation[
                        "reference_name"
                    ]
            })

    return (
        internal,
        external,
        ambiguous
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("==========================================")
    print("STRICT INTERNAL CITATION DETECTION")
    print("==========================================")
    print()

    corpus_cases = load_corpus_cases()

    citations = load_citations()

    print(
        f"Corpus cases : "
        f"{len(corpus_cases)}"
    )

    print(
        f"Citation relationships : "
        f"{len(citations)}"
    )

    (
        internal,
        external,
        ambiguous
    ) = find_internal_citations(
        citations,
        corpus_cases
    )

    print()
    print("RESULTS")
    print("------------------------------------------")

    print(
        f"Internal citations : "
        f"{len(internal)}"
    )

    print(
        f"External/unresolved : "
        f"{len(external)}"
    )

    print(
        f"Ambiguous matches : "
        f"{len(ambiguous)}"
    )

    total = len(citations)

    if total:

        internal_rate = (
            len(internal)
            / total
        ) * 100

    else:

        internal_rate = 0

    print(
        f"Internal citation rate : "
        f"{internal_rate:.2f}%"
    )

    # --------------------------------------------------------
    # EXAMPLES
    # --------------------------------------------------------

    print()
    print("INTERNAL CITATION EXAMPLES")
    print("------------------------------------------")

    for item in internal[:30]:

        print(
            item["reference_name"]
        )

        print(
            f"  -> {item['target_case_name']}"
        )

    print()
    print("AMBIGUOUS EXAMPLES")
    print("------------------------------------------")

    for item in ambiguous[:15]:

        print(
            item["reference_name"]
        )

    # --------------------------------------------------------
    # SAVE REPORT
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    report = {

        "metadata": {

            "corpus_cases":
                len(corpus_cases),

            "citation_relationships":
                len(citations),

            "internal_citations":
                len(internal),

            "external_or_unresolved":
                len(external),

            "ambiguous":
                len(ambiguous),

            "internal_citation_rate_percent":
                internal_rate
        },

        "internal_citations":
            internal,

        "external_or_unresolved":
            external,

        "ambiguous":
            ambiguous
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
    print("------------------------------------------")

    print(OUTPUT_FILE)

    print()
    print("==========================================")
    print()


if __name__ == "__main__":
    main()