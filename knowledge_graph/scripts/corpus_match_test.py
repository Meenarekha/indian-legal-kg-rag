import json
import re
from pathlib import Path


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


def normalize_name(name):

    if not name:
        return ""

    text = name.lower()

    # Remove reporting information
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

    # Normalize versus forms
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

    # Normalize common suffixes
    text = re.sub(
        r"\b(and others|and ors|ors|others)\b",
        " ",
        text
    )

    # Remove punctuation
    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text
    )

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


def load_corpus_cases():

    cases = []

    for path in sorted(
        CLEANED_DIR.glob("*.json")
    ):

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        case_id = data.get("case_id")

        if not case_id:
            continue

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
            "normalized": normalize_name(
                case_name
            )
        })

    return cases


def load_citations():

    with open(
        GRAPH_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        graph = json.load(f)

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

        target = relationship.get(
            "target"
        )

        node = nodes.get(target)

        if not node:
            continue

        properties = node.get(
            "properties",
            {}
        )

        name = properties.get(
            "case_name"
        )

        if name:

            citations.append(name)

    return list(dict.fromkeys(citations))


def split_case_name(name):

    normalized = normalize_name(name)

    parts = re.split(
        r"\s+v\s+",
        normalized
    )

    if len(parts) != 2:
        return None, None

    return (
        parts[0].strip(),
        parts[1].strip()
    )


def party_tokens(text):

    if not text:
        return set()

    return {
        token
        for token in text.split()
        if len(token) >= 3
    }

def conservative_match(
    citation,
    corpus_case
):

    citation_parts = split_case_name(
        citation
    )

    corpus_parts = split_case_name(
        corpus_case["case_name"]
    )

    if not citation_parts:
        return False

    if not corpus_parts:
        return False

    cit_petitioner, cit_respondent = (
        citation_parts
    )

    cor_petitioner, cor_respondent = (
        corpus_parts
    )

    cit_p = party_tokens(
        cit_petitioner
    )

    cor_p = party_tokens(
        cor_petitioner
    )

    cit_r = party_tokens(
        cit_respondent
    )

    cor_r = party_tokens(
        cor_respondent
    )

    if not cit_p or not cit_r:
        return False

    if not cor_p or not cor_r:
        return False

    petitioner_overlap = (
        len(cit_p & cor_p)
        / min(
            len(cit_p),
            len(cor_p)
        )
    )

    respondent_overlap = (
        len(cit_r & cor_r)
        / min(
            len(cit_r),
            len(cor_r)
        )
    )

    # Very conservative:
    # both parties must have strong token overlap.
    return (
        petitioner_overlap >= 0.80
        and
        respondent_overlap >= 0.80
    )


def main():

    print()
    print("==========================================")
    print("CORPUS-AWARE CITATION MATCH TEST")
    print("==========================================")
    print()

    corpus_cases = load_corpus_cases()

    citations = load_citations()

    print(
        f"Corpus cases : {len(corpus_cases)}"
    )

    print(
        f"Unique citation names : {len(citations)}"
    )

    matches = []
    unmatched = []

    for citation in citations:

        found = []

        for case in corpus_cases:

            if conservative_match(
                citation,
                case
            ):

                found.append(case)

        if len(found) == 1:

            matches.append({
                "citation": citation,
                "matched_case":
                    found[0]["case_name"],
                "case_id":
                    found[0]["case_id"]
            })

        else:

            unmatched.append({
                "citation": citation,
                "possible_matches":
                    len(found)
            })

    print()
    print("RESULTS")
    print("------------------------------------------")

    print(
        f"Unique citations tested : "
        f"{len(citations)}"
    )

    print(
        f"Unique citations matched : "
        f"{len(matches)}"
    )

    print(
        f"Unmatched / ambiguous : "
        f"{len(unmatched)}"
    )

    if citations:

        rate = (
            len(matches)
            / len(citations)
        ) * 100

    else:

        rate = 0

    print(
        f"Resolution rate : "
        f"{rate:.2f}%"
    )

    print()
    print("MATCH EXAMPLES")
    print("------------------------------------------")

    for item in matches[:30]:

        print(
            f"{item['citation']}"
        )

        print(
            f"  -> {item['matched_case']}"
        )

    print()
    print("UNMATCHED EXAMPLES")
    print("------------------------------------------")

    for item in unmatched[:30]:

        print(
            item["citation"]
        )

    print()
    print("==========================================")
    print()


if __name__ == "__main__":
    main()