import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORT_FILE = (
    PROJECT_ROOT
    / "knowledge_graph"
    / "data"
    / "relationships"
    / "validated_internal_citation_report.json"
)

CLEANED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cleaned"
)


STRONG_PATTERNS = [
    r"\bthis court\b",
    r"\bheld\b",
    r"\bheld that\b",
    r"\breiterated\b",
    r"\breaffirmed\b",
    r"\brelied upon\b",
    r"\brelied on\b",
    r"\bfollowed\b",
    r"\bendorsed\b",
    r"\breferred to\b",
    r"\bconsidered\b",
    r"\bdecision in\b",
    r"\bjudgment in\b",
    r"\bprecedent\b",
    r"\bauthority\b",
    r"\bcase of\b",
    r"\bin .*? v .*?\b",
]

WEAK_PATTERNS = [
    r"\bw\.?\s*p\.?\s*c\.?\b",
    r"\bwrit petition\b",
    r"\bquestion answer\b",
    r"\bsi question\b",
    r"\bannexure\b",
    r"\bserial\b",
    r"\bsl no\b",
    r"\bcase number\b",
]


def normalize_text(text):

    text = text.lower()

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

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def load_case_text(case_id):

    for file_path in CLEANED_DIR.glob("*.json"):

        try:

            with open(
                file_path,
                "r",
                encoding="utf-8"
            ) as f:

                data = json.load(f)

            if data.get("case_id") == case_id:

                pages = data.get("pages", [])

                return "\n".join(
                    page.get("cleaned_text", "")
                    for page in pages
                )

        except Exception:

            continue

    return None


def find_context(
    text,
    target_name,
    window=350
):

    normalized_text = normalize_text(text)

    normalized_target = normalize_text(
        target_name
    )

    position = normalized_text.find(
        normalized_target
    )

    if position == -1:

        return None

    start = max(
        0,
        position - window
    )

    end = min(
        len(normalized_text),
        position
        + len(normalized_target)
        + window
    )

    return normalized_text[start:end]


def count_patterns(
    context,
    patterns
):

    count = 0

    for pattern in patterns:

        if re.search(
            pattern,
            context,
            re.IGNORECASE
        ):

            count += 1

    return count


def classify_context(context):

    strong_score = count_patterns(
        context,
        STRONG_PATTERNS
    )

    weak_score = count_patterns(
        context,
        WEAK_PATTERNS
    )

    if strong_score > 0 and strong_score >= weak_score:

        return "strong_candidate", strong_score, weak_score

    if weak_score > 0 and weak_score > strong_score:

        return "weak_candidate", strong_score, weak_score

    return "uncertain", strong_score, weak_score


def main():

    print("=" * 70)
    print("CLASSIFYING INTERNAL CITATION CONTEXTS")
    print("=" * 70)

    with open(
        REPORT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        report = json.load(f)

    citations = report.get(
        "internal_citations",
        []
    )

    results = []

    strong = 0
    weak = 0
    uncertain = 0

    for citation in citations:

        source_id = citation[
            "source_case_id"
        ]

        target_name = citation[
            "target_case_name"
        ]

        text = load_case_text(
            source_id
        )

        if text is None:

            classification = (
                "source_text_not_found"
            )

            result = {
                **citation,
                "classification":
                    classification
            }

            results.append(result)

            continue

        context = find_context(
            text,
            target_name
        )

        if context is None:

            classification = (
                "target_not_found"
            )

            result = {
                **citation,
                "classification":
                    classification
            }

            results.append(result)

            continue

        classification, strong_score, weak_score = (
            classify_context(context)
        )

        if classification == "strong_candidate":
            strong += 1

        elif classification == "weak_candidate":
            weak += 1

        else:
            uncertain += 1

        results.append({

            **citation,

            "classification":
                classification,

            "strong_signal_count":
                strong_score,

            "weak_signal_count":
                weak_score,

            "context":
                context
        })

    output_file = (
        PROJECT_ROOT
        / "knowledge_graph"
        / "data"
        / "relationships"
        / "classified_citation_contexts.json"
    )

    output = {

        "metadata": {

            "input_edges":
                len(citations),

            "strong_candidates":
                strong,

            "weak_candidates":
                weak,

            "uncertain":
                uncertain,

            "method":
                (
                    "Deterministic context classification "
                    "using legal-reference and "
                    "list/procedural indicator patterns."
                )
        },

        "internal_citations":
            results
    }

    with open(
        output_file,
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
        f"Input relationships : {len(citations)}"
    )

    print(
        f"Strong candidates   : {strong}"
    )

    print(
        f"Weak candidates     : {weak}"
    )

    print(
        f"Uncertain           : {uncertain}"
    )

    print()
    print("OUTPUT")
    print("-" * 70)

    print(output_file)

    print()
    print("=" * 70)
    print("CLASSIFICATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()