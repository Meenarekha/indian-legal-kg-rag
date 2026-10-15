from pathlib import Path
import json
import re


PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXTRACTED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "extracted"
)

CLEANED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cleaned"
)


# ============================================================
# ARTIFACT PATTERNS
# ============================================================

ARTIFACT_PATTERNS = {

    "Indian Kanoon footer":
        r"Indian Kanoon\s*-\s*http://indiankanoon\.org/",

    "DOC marker":
        r"(?m)^\s*[A-Za-z0-9_\-()]+\.DOC\s*$",

    "Uploaded marker":
        r"Uploaded\s+on\s*-",

    "Downloaded marker":
        r"Downloaded\s+on\s*-",

}


# ============================================================
# LEGAL CONTENT PATTERNS
# ============================================================

LEGAL_PATTERNS = {

    "Section reference":
        r"\bSection\s+\d+[A-Za-z]?(?:\s*\(\s*\d+\s*\))?",

    "Paragraph number":
        r"(?m)^\s*\d+\.\s+",

    "Legal citation":
        r"\b\d{4}:[A-Z]{2,10}-[A-Z]{2,10}:\d+",

    "Petition reference":
        r"\b(?:WRIT|CIVIL|CRIMINAL|SPECIAL|PUBLIC INTEREST)\b",

    "Court reference":
        r"\b(?:HIGH COURT|SUPREME COURT)\b",

}


# ============================================================
# LOAD CASE
# ============================================================

def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ============================================================
# COMBINE PAGE TEXT
# ============================================================

def combine_original_text(case):

    return "\n".join(
        page.get("text", "")
        for page in case.get("pages", [])
    )


def combine_cleaned_text(case):

    return "\n".join(
        page.get("cleaned_text", "")
        for page in case.get("pages", [])
    )


# ============================================================
# COUNT PATTERNS
# ============================================================

def count_pattern(pattern, text):

    return len(
        re.findall(
            pattern,
            text,
            flags=re.IGNORECASE
        )
    )


# ============================================================
# MAIN VALIDATION
# ============================================================

def main():

    print("=" * 80)
    print("CLEANING VALIDATION")
    print("=" * 80)

    extracted_files = sorted(
        EXTRACTED_DIR.glob("*.json")
    )

    extracted_files = [
        file
        for file in extracted_files
        if file.name != "extraction_summary.json"
    ]

    cleaned_files = sorted(
        CLEANED_DIR.glob("*.json")
    )

    print(
        f"\nExtracted case files: {len(extracted_files)}"
    )

    print(
        f"Cleaned case files:   {len(cleaned_files)}"
    )

    # --------------------------------------------------------
    # File count check
    # --------------------------------------------------------

    if len(extracted_files) != len(cleaned_files):

        print(
            "\nWARNING: File counts do not match."
        )

    else:

        print(
            "\nPASS: All extracted cases have cleaned versions."
        )

    # --------------------------------------------------------
    # Aggregate statistics
    # --------------------------------------------------------

    artifact_counts = {
        name: {
            "before": 0,
            "after": 0
        }
        for name in ARTIFACT_PATTERNS
    }

    legal_counts = {
        name: {
            "before": 0,
            "after": 0
        }
        for name in LEGAL_PATTERNS
    }

    cases_with_empty_cleaned_pages = 0

    total_pages = 0

    # --------------------------------------------------------
    # Compare cases
    # --------------------------------------------------------

    for extracted_path in extracted_files:

        cleaned_path = (
            CLEANED_DIR
            / extracted_path.name
        )

        if not cleaned_path.exists():
            continue

        extracted_case = load_json(
            extracted_path
        )

        cleaned_case = load_json(
            cleaned_path
        )

        original_text = combine_original_text(
            extracted_case
        )

        cleaned_text = combine_cleaned_text(
            cleaned_case
        )

        # ----------------------------------------------------
        # Artifact checks
        # ----------------------------------------------------

        for name, pattern in ARTIFACT_PATTERNS.items():

            artifact_counts[name]["before"] += (
                count_pattern(
                    pattern,
                    original_text
                )
            )

            artifact_counts[name]["after"] += (
                count_pattern(
                    pattern,
                    cleaned_text
                )
            )

        # ----------------------------------------------------
        # Legal-content checks
        # ----------------------------------------------------

        for name, pattern in LEGAL_PATTERNS.items():

            legal_counts[name]["before"] += (
                count_pattern(
                    pattern,
                    original_text
                )
            )

            legal_counts[name]["after"] += (
                count_pattern(
                    pattern,
                    cleaned_text
                )
            )

        # ----------------------------------------------------
        # Empty page check
        # ----------------------------------------------------

        for page in cleaned_case.get(
            "pages",
            []
        ):

            total_pages += 1

            if not page.get(
                "cleaned_text",
                ""
            ).strip():

                cases_with_empty_cleaned_pages += 1

    # ========================================================
    # PRINT ARTIFACT RESULTS
    # ========================================================

    print("\n" + "=" * 80)
    print("ARTIFACT REMOVAL")
    print("=" * 80)

    for name, counts in artifact_counts.items():

        print(
            f"\n{name}"
        )

        print(
            f"Before: {counts['before']:,}"
        )

        print(
            f"After:  {counts['after']:,}"
        )

        if counts["after"] == 0:

            print("Status: PASS")

        else:

            print(
                "Status: REVIEW"
            )

    # ========================================================
    # PRINT LEGAL CONTENT RESULTS
    # ========================================================

    print("\n" + "=" * 80)
    print("LEGAL CONTENT PRESERVATION")
    print("=" * 80)

    for name, counts in legal_counts.items():

        print(
            f"\n{name}"
        )

        print(
            f"Before: {counts['before']:,}"
        )

        print(
            f"After:  {counts['after']:,}"
        )

        if counts["after"] > 0:

            print("Status: PRESENT")

        else:

            print("Status: REVIEW")

    # ========================================================
    # PAGE VALIDATION
    # ========================================================

    print("\n" + "=" * 80)
    print("PAGE VALIDATION")
    print("=" * 80)

    print(
        f"\nTotal cleaned pages checked: "
        f"{total_pages:,}"
    )

    print(
        f"Pages with empty cleaned text: "
        f"{cases_with_empty_cleaned_pages:,}"
    )

    if cases_with_empty_cleaned_pages == 0:

        print(
            "Status: PASS"
        )

    else:

        print(
            "Status: REVIEW"
        )

    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()