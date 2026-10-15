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


def normalize_text(text):
    text = text.lower()

    text = re.sub(r"\bversus\b", " v ", text)
    text = re.sub(r"\bvs\.?\b", " v ", text)

    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def load_validated_report():

    with open(REPORT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_case_text(case_id):

    possible_files = list(
        CLEANED_DIR.glob("*.json")
    )

    for file_path in possible_files:

        try:

            with open(
                file_path,
                "r",
                encoding="utf-8"
            ) as f:

                data = json.load(f)

            if data.get("case_id") == case_id:

                pages = data.get("pages", [])

                full_text = "\n".join(
                    page.get("cleaned_text", "")
                    for page in pages
                )

                return full_text

        except Exception:
            continue

    return None


def extract_context(text, target_name, window=250):

    normalized_text = normalize_text(text)
    normalized_target = normalize_text(target_name)

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
        position + len(normalized_target) + window
    )

    return normalized_text[start:end]


def main():

    print("=" * 70)
    print("INSPECTING VALIDATED INTERNAL CITATIONS")
    print("=" * 70)

    report = load_validated_report()

    citations = report.get(
        "internal_citations",
        []
    )

    print()
    print(
        f"Validated citation candidates : "
        f"{len(citations)}"
    )

    print()
    print("CITATION CONTEXTS")
    print("-" * 70)

    results = []

    for index, citation in enumerate(citations, start=1):

        source_id = citation[
            "source_case_id"
        ]

        source_name = citation[
            "source_case_name"
        ]

        target_name = citation[
            "target_case_name"
        ]

        text = load_case_text(
            source_id
        )

        print()
        print(
            f"[{index}] "
            f"{source_name}"
        )

        print(
            f"     -> {target_name}"
        )

        if text is None:

            print(
                "     STATUS: SOURCE TEXT NOT FOUND"
            )

            results.append({
                "source_case_id": source_id,
                "source_case_name": source_name,
                "target_case_id":
                    citation["target_case_id"],
                "target_case_name": target_name,
                "status": "source_text_not_found"
            })

            continue

        context = extract_context(
            text,
            target_name
        )

        if context is None:

            print(
                "     STATUS: TARGET NAME NOT FOUND"
            )

            results.append({
                "source_case_id": source_id,
                "source_case_name": source_name,
                "target_case_id":
                    citation["target_case_id"],
                "target_case_name": target_name,
                "status": "target_name_not_found"
            })

            continue

        print(
            "     STATUS: MATCH FOUND"
        )

        print(
            "     CONTEXT:"
        )

        print(
            f"     ... {context} ..."
        )

        results.append({
            "source_case_id": source_id,
            "source_case_name": source_name,
            "target_case_id":
                citation["target_case_id"],
            "target_case_name": target_name,
            "status": "match_found",
            "context": context
        })

    print()
    print("=" * 70)
    print("INSPECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()