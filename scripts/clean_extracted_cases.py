from pathlib import Path
import json
import re


# ============================================================
# PROJECT PATHS
# ============================================================

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
# DIRECTORY SETUP
# ============================================================

CLEANED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_page_text(text):
    if not text:
        return ""

    # ========================================================
    # SOURCE FOOTER
    # ========================================================

    text = re.sub(
        r"Indian Kanoon\s*-\s*http://indiankanoon\.org/\S*",
        "",
        text,
        flags=re.IGNORECASE
    )

    # ========================================================
    # STANDALONE PAGE NUMBERS
    # ========================================================

    text = re.sub(
        r"(?m)^\s*\d+\s*$",
        "",
        text
    )

    # ========================================================
    # DOCUMENT FILE MARKERS
    #
    # Examples:
    #   something.DOC
    #   Rane SMWP1_2024.DOC
    # ========================================================

    text = re.sub(
        r"(?mi)^\s*[A-Za-z0-9_\-() .]+\.DOC\s*$",
        "",
        text
    )

    # ========================================================
    # UPLOADED / DOWNLOADED METADATA
    # ========================================================

    text = re.sub(
        r"(?mi)^\s*:::\s*Uploaded\s+on\s*-.*$",
        "",
        text
    )

    text = re.sub(
        r"(?mi)^\s*:::\s*Downloaded\s+on\s*-.*$",
        "",
        text
    )

    # ========================================================
    # DIGITAL SIGNATURE LABEL
    # ========================================================

    text = re.sub(
        r"(?mi)^\s*Digitally\s+signed\s+by\s*$",
        "",
        text
    )

    # ========================================================
    # DIGITAL SIGNATURE DATE
    #
    # Example:
    # Date: 2024.05.06
    # ========================================================

    text = re.sub(
        r"(?mi)^\s*Date:\s*\d{4}\.\d{2}\.\d{2}\s*$",
        "",
        text
    )

    # ========================================================
    # DIGITAL SIGNATURE TIMESTAMP
    #
    # Examples:
    # 13:54:49 IST
    # 16:52:02 +0530
    # ========================================================

    text = re.sub(
        r"(?mi)^\s*\d{2}:\d{2}:\d{2}(?:\s+[A-Z]+|\s+[+-]\d{4})?\s*$",
        "",
        text
    )

    # ========================================================
    # SIGNATURE REASON LABEL
    # ========================================================

    text = re.sub(
        r"(?mi)^\s*Reason:\s*$",
        "",
        text
    )

    # ========================================================
    # SIGNATURE TIMESTAMP + NAME
    #
    # Example:
    # 05.07 GARG 16:52:02 +0530
    #
    # We remove only lines that strongly resemble
    # document/signature metadata.
    # ========================================================

    text = re.sub(
        r"(?mi)^\s*\d{2}\.\d{2}\s+[A-Z][A-Za-z .'-]{1,40}\s+\d{2}:\d{2}:\d{2}\s+[+-]\d{4}\s*$",
        "",
        text
    )

    # ========================================================
    # COMMON DIGITAL-SIGNATURE METADATA LINES
    # ========================================================

    text = re.sub(
        r"(?mi)^\s*Digitally\s+signed\s+by\s+.*$",
        "",
        text
    )

    # ========================================================
    # NORMALIZE SPACING
    # ========================================================

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    # ========================================================
    # REDUCE EXCESSIVE BLANK LINES
    # ========================================================

    text = re.sub(
        r"\n\s*\n\s*\n+",
        "\n\n",
        text
    )

    # ========================================================
    # STRIP INDIVIDUAL LINES
    # ========================================================

    lines = []

    for line in text.splitlines():

        line = line.strip()

        if line:
            lines.append(line)

    return "\n".join(lines).strip()

# ============================================================
# CASE PROCESSING
# ============================================================

def process_case(json_path):
    """
    Read one extracted case and create its cleaned version.
    """

    with open(
        json_path,
        "r",
        encoding="utf-8"
    ) as file:

        case = json.load(file)

    cleaned_pages = []

    for page in case.get("pages", []):

        original_text = page.get(
            "text",
            ""
        )

        cleaned_text = clean_page_text(
            original_text
        )

        cleaned_page = {
            "page_number": page.get(
                "page_number"
            ),

            "original_character_count": len(
                original_text
            ),

            "cleaned_character_count": len(
                cleaned_text
            ),

            "cleaned_text": cleaned_text
        }

        cleaned_pages.append(
            cleaned_page
        )

    # --------------------------------------------------------
    # Preserve document-level metadata
    # --------------------------------------------------------

    cleaned_case = {
        "case_id": case.get(
            "case_id"
        ),

        "court": case.get(
            "court"
        ),

        "source_file": case.get(
            "source_file"
        ),

        "source_path": case.get(
            "source_path"
        ),

        "page_count": case.get(
            "page_count"
        ),

        "pages_with_text": case.get(
            "pages_with_text"
        ),

        "pages_without_text": case.get(
            "pages_without_text"
        ),

        "total_characters_original": case.get(
            "total_characters"
        ),

        "pages": cleaned_pages
    }

    return cleaned_case


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("LEGAL CASE CLEANING PIPELINE")
    print("=" * 80)

    # --------------------------------------------------------
    # Check extracted directory
    # --------------------------------------------------------

    if not EXTRACTED_DIR.exists():

        print(
            "\nERROR: Extracted directory not found:"
        )

        print(
            EXTRACTED_DIR
        )

        return

    # --------------------------------------------------------
    # Find extracted case files
    # --------------------------------------------------------

    json_files = sorted(
        EXTRACTED_DIR.glob("*.json")
    )

    # Do not treat extraction_summary.json as a case.
    json_files = [
        file
        for file in json_files
        if file.name != "extraction_summary.json"
    ]

    if not json_files:

        print(
            "\nERROR: No extracted case JSON files found."
        )

        return

    print(
        f"\nFound {len(json_files)} extracted case files."
    )

    # --------------------------------------------------------
    # Processing counters
    # --------------------------------------------------------

    successful = 0
    failed = 0

    total_original_chars = 0
    total_cleaned_chars = 0

    # --------------------------------------------------------
    # Process every case
    # --------------------------------------------------------

    for index, json_path in enumerate(
        json_files,
        start=1
    ):

        try:

            cleaned_case = process_case(
                json_path
            )

            output_path = (
                CLEANED_DIR
                / json_path.name
            )

            with open(
                output_path,
                "w",
                encoding="utf-8"
            ) as file:

                json.dump(
                    cleaned_case,
                    file,
                    ensure_ascii=False,
                    indent=2
                )

            # ------------------------------------------------
            # Character statistics
            # ------------------------------------------------

            total_original_chars += cleaned_case.get(
                "total_characters_original",
                0
            )

            for page in cleaned_case.get(
                "pages",
                []
            ):

                total_cleaned_chars += page.get(
                    "cleaned_character_count",
                    0
                )

            successful += 1

            print(
                f"[{index}/{len(json_files)}] "
                f"Cleaned: {json_path.name}"
            )

        except Exception as error:

            failed += 1

            print(
                f"[{index}/{len(json_files)}] "
                f"FAILED: {json_path.name}"
            )

            print(
                f"Error: {error}"
            )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 80)
    print("CLEANING COMPLETE")
    print("=" * 80)

    print(
        f"\nTotal cases:              {len(json_files)}"
    )

    print(
        f"Successfully cleaned:     {successful}"
    )

    print(
        f"Failed:                   {failed}"
    )

    print(
        f"Original characters:      {total_original_chars:,}"
    )

    print(
        f"Cleaned characters:       {total_cleaned_chars:,}"
    )

    print(
        "\nOutput directory:"
    )

    print(
        CLEANED_DIR
    )

    print("=" * 80)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()