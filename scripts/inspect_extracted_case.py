from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXTRACTED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "extracted"
)


def inspect_case(json_path: Path):
    print("=" * 80)
    print("EXTRACTED CASE INSPECTION")
    print("=" * 80)

    print(f"\nFile:")
    print(json_path)

    with open(
        json_path,
        "r",
        encoding="utf-8"
    ) as file:
        case = json.load(file)

    print("\nCASE METADATA")
    print("-" * 80)

    print(f"Case ID:              {case.get('case_id')}")
    print(f"Court:                {case.get('court')}")
    print(f"Source file:          {case.get('source_file')}")
    print(f"Source path:          {case.get('source_path')}")
    print(f"Page count:           {case.get('page_count')}")
    print(f"Pages with text:      {case.get('pages_with_text')}")
    print(f"Pages without text:   {case.get('pages_without_text')}")
    print(f"Total characters:     {case.get('total_characters')}")

    pages = case.get("pages", [])

    print("\nFIRST PAGE")
    print("-" * 80)

    if pages:

        first_page = pages[0]

        print(
            f"Page number: "
            f"{first_page.get('page_number')}"
        )

        print(
            f"Character count: "
            f"{first_page.get('character_count')}"
        )

        print("\nExtracted text:\n")

        print(
            first_page.get("text", "")[:8000]
        )

    print("\n" + "=" * 80)


def main():

    if not EXTRACTED_DIR.exists():

        print(
            "ERROR: Extracted directory not found:"
        )

        print(EXTRACTED_DIR)

        return

    json_files = sorted(
        EXTRACTED_DIR.glob("*.json")
    )

    # Do not inspect the summary file.
    json_files = [
        file
        for file in json_files
        if file.name != "extraction_summary.json"
    ]

    if not json_files:

        print(
            "ERROR: No extracted case JSON files found."
        )

        return

    print(
        f"Found {len(json_files)} extracted case files."
    )

    print(
        "\nThe first extracted case will be inspected."
    )

    inspect_case(json_files[0])


if __name__ == "__main__":
    main()