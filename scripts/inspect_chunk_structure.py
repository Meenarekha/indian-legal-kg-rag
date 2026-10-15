from pathlib import Path
import json
import re
import statistics


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CLEANED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cleaned"
)


# ============================================================
# PARAGRAPH DETECTION
# ============================================================

PARAGRAPH_PATTERN = re.compile(
    r"(?m)^\s*(\d{1,4})\.\s+"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("LEGAL CHUNK STRUCTURE INSPECTION")
    print("=" * 80)

    if not CLEANED_DIR.exists():

        print("\nERROR: Cleaned directory not found:")
        print(CLEANED_DIR)

        return

    json_files = sorted(
        CLEANED_DIR.glob("*.json")
    )

    if not json_files:

        print("\nERROR: No cleaned case files found.")

        return

    print(
        f"\nCleaned cases found: {len(json_files)}"
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    case_character_counts = []
    page_character_counts = []
    paragraph_counts = []
    paragraph_lengths = []

    cases_with_paragraphs = 0
    cases_without_paragraphs = 0

    total_pages = 0

    # --------------------------------------------------------
    # Inspect cases
    # --------------------------------------------------------

    for json_path in json_files:

        with open(
            json_path,
            "r",
            encoding="utf-8"
        ) as file:

            case = json.load(file)

        case_text_parts = []

        case_paragraphs = []

        for page in case.get(
            "pages",
            []
        ):

            text = page.get(
                "cleaned_text",
                ""
            )

            if not text:
                continue

            total_pages += 1

            page_character_counts.append(
                len(text)
            )

            case_text_parts.append(text)

            matches = list(
                PARAGRAPH_PATTERN.finditer(text)
            )

            for match in matches:

                case_paragraphs.append(
                    match.group(1)
                )

        full_text = "\n".join(
            case_text_parts
        )

        case_character_counts.append(
            len(full_text)
        )

        paragraph_counts.append(
            len(case_paragraphs)
        )

        if case_paragraphs:

            cases_with_paragraphs += 1

        else:

            cases_without_paragraphs += 1

        # ----------------------------------------------------
        # Estimate paragraph lengths
        # ----------------------------------------------------

        matches = list(
            PARAGRAPH_PATTERN.finditer(
                full_text
            )
        )

        for index, match in enumerate(matches):

            start = match.start()

            if index + 1 < len(matches):

                end = matches[
                    index + 1
                ].start()

            else:

                end = len(full_text)

            paragraph_text = (
                full_text[start:end]
            )

            paragraph_lengths.append(
                len(paragraph_text)
            )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 80)
    print("DOCUMENT STATISTICS")
    print("=" * 80)

    print(
        f"\nTotal cases:              {len(json_files):,}"
    )

    print(
        f"Total pages inspected:    {total_pages:,}"
    )

    print(
        f"Cases with paragraphs:     {cases_with_paragraphs:,}"
    )

    print(
        f"Cases without paragraphs:  {cases_without_paragraphs:,}"
    )

    # --------------------------------------------------------
    # Case size
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("CASE CHARACTERISTICS")
    print("=" * 80)

    if case_character_counts:

        print(
            f"\nMinimum characters: "
            f"{min(case_character_counts):,}"
        )

        print(
            f"Maximum characters: "
            f"{max(case_character_counts):,}"
        )

        print(
            f"Average characters: "
            f"{statistics.mean(case_character_counts):,.0f}"
        )

        print(
            f"Median characters: "
            f"{statistics.median(case_character_counts):,.0f}"
        )

    # --------------------------------------------------------
    # Page size
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("PAGE CHARACTERISTICS")
    print("=" * 80)

    if page_character_counts:

        print(
            f"\nMinimum characters: "
            f"{min(page_character_counts):,}"
        )

        print(
            f"Maximum characters: "
            f"{max(page_character_counts):,}"
        )

        print(
            f"Average characters: "
            f"{statistics.mean(page_character_counts):,.0f}"
        )

        print(
            f"Median characters: "
            f"{statistics.median(page_character_counts):,.0f}"
        )

    # --------------------------------------------------------
    # Paragraph statistics
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("PARAGRAPH CHARACTERISTICS")
    print("=" * 80)

    if paragraph_counts:

        print(
            f"\nMinimum paragraphs/case: "
            f"{min(paragraph_counts):,}"
        )

        print(
            f"Maximum paragraphs/case: "
            f"{max(paragraph_counts):,}"
        )

        print(
            f"Average paragraphs/case: "
            f"{statistics.mean(paragraph_counts):,.1f}"
        )

        print(
            f"Median paragraphs/case: "
            f"{statistics.median(paragraph_counts):,.1f}"
        )

    if paragraph_lengths:

        print(
            f"\nMinimum paragraph length: "
            f"{min(paragraph_lengths):,}"
        )

        print(
            f"Maximum paragraph length: "
            f"{max(paragraph_lengths):,}"
        )

        print(
            f"Average paragraph length: "
            f"{statistics.mean(paragraph_lengths):,.0f}"
        )

        print(
            f"Median paragraph length: "
            f"{statistics.median(paragraph_lengths):,.0f}"
        )

    print("\n" + "=" * 80)
    print("INSPECTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()