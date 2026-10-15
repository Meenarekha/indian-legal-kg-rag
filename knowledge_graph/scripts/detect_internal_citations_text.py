import json
import re
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

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
    / "internal_citation_text_report.json"
)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text):

    if not text:
        return ""

    text = text.lower()

    # Normalize common citation separators.
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

    # Remove punctuation.
    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text
    )

    # Normalize whitespace.
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# CASE NAME FROM CASE ID
# ============================================================

def case_name_from_id(case_id):

    if "_on_" in case_id:

        name = (
            case_id
            .rsplit("_on_", 1)[0]
        )

    else:

        name = case_id

    return name.replace(
        "_",
        " "
    )


# ============================================================
# LOAD CORPUS
# ============================================================

def load_corpus():

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

            case_name = case_name_from_id(
                case_id
            )

            pages = data.get(
                "pages",
                []
            )

            full_text = "\n".join(
                page.get(
                    "cleaned_text",
                    ""
                )
                for page in pages
            )

            cases.append({
                "case_id": case_id,
                "case_name": case_name,
                "normalized_name":
                    normalize_text(
                        case_name
                    ),
                "full_text":
                    full_text
            })

        except Exception as error:

            print(
                f"[WARNING] Could not load "
                f"{path.name}: {error}"
            )

    return cases


# ============================================================
# FIND INTERNAL CITATIONS
# ============================================================

def find_internal_citations(cases):

    results = []

    # --------------------------------------------------------
    # Create searchable corpus index.
    # --------------------------------------------------------

    corpus_index = {}

    for case in cases:

        normalized = case[
            "normalized_name"
        ]

        if not normalized:
            continue

        # Avoid duplicate normalized names.
        corpus_index.setdefault(
            normalized,
            []
        ).append(case)

    # --------------------------------------------------------
    # Search each judgment.
    # --------------------------------------------------------

    for source_case in cases:

        source_text = normalize_text(
            source_case["full_text"]
        )

        if not source_text:
            continue

        for normalized_name, targets in (
            corpus_index.items()
        ):

            # ------------------------------------------------
            # Do not count self citation.
            # ------------------------------------------------

            for target_case in targets:

                if (
                    target_case["case_id"]
                    == source_case["case_id"]
                ):
                    continue

                # ------------------------------------------------
                # Exact normalized phrase search.
                # ------------------------------------------------

                pattern = (
                    r"(?<![a-z0-9])"
                    + re.escape(
                        normalized_name
                    )
                    + r"(?![a-z0-9])"
                )

                if re.search(
                    pattern,
                    source_text
                ):

                    results.append({
                        "source_case_id":
                            source_case[
                                "case_id"
                            ],

                        "source_case_name":
                            source_case[
                                "case_name"
                            ],

                        "target_case_id":
                            target_case[
                                "case_id"
                            ],

                        "target_case_name":
                            target_case[
                                "case_name"
                            ],

                        "match_type":
                            "exact_normalized_text"
                    })

    return results


# ============================================================
# REMOVE DUPLICATES
# ============================================================

def remove_duplicates(results):

    unique = []

    seen = set()

    for item in results:

        key = (
            item["source_case_id"],
            item["target_case_id"]
        )

        if key in seen:
            continue

        seen.add(key)

        unique.append(item)

    return unique


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("==========================================")
    print("TEXT-BASED INTERNAL CITATION DETECTION")
    print("==========================================")
    print()

    cases = load_corpus()

    print(
        f"Corpus cases : {len(cases)}"
    )

    results = find_internal_citations(
        cases
    )

    results = remove_duplicates(
        results
    )

    print()
    print("RESULTS")
    print("------------------------------------------")

    print(
        f"Internal citation edges : "
        f"{len(results)}"
    )

    # --------------------------------------------------------
    # Number of source cases with citations.
    # --------------------------------------------------------

    source_cases = {
        item["source_case_id"]
        for item in results
    }

    target_cases = {
        item["target_case_id"]
        for item in results
    }

    print(
        f"Source cases with internal citations : "
        f"{len(source_cases)}"
    )

    print(
        f"Target cases cited internally : "
        f"{len(target_cases)}"
    )

    # --------------------------------------------------------
    # Examples
    # --------------------------------------------------------

    print()
    print("INTERNAL CITATION EXAMPLES")
    print("------------------------------------------")

    for item in results[:50]:

        print(
            f"{item['source_case_name']}"
        )

        print(
            f"  -> {item['target_case_name']}"
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
                len(cases),

            "internal_citation_edges":
                len(results),

            "source_cases_with_internal_citations":
                len(source_cases),

            "target_cases_cited_internally":
                len(target_cases),

            "method":
                "exact normalized case-name search in cleaned judgment text"
        },

        "internal_citations":
            results
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