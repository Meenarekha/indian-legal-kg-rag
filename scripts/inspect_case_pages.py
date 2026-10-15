from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTRACTED_DIR = PROJECT_ROOT / "data" / "processed" / "extracted"


def main():

    json_files = sorted(EXTRACTED_DIR.glob("*.json"))
    json_files = [
        file for file in json_files
        if file.name != "extraction_summary.json"
    ]

    if not json_files:
        print("ERROR: No extracted case files found.")
        return

    json_path = json_files[0]

    with open(json_path, "r", encoding="utf-8") as file:
        case = json.load(file)

    print("=" * 80)
    print("LEGAL CASE PAGE INSPECTION")
    print("=" * 80)

    print(f"\nCase: {case.get('case_id')}")
    print(f"Court: {case.get('court')}")
    print(f"Total pages: {case.get('page_count')}")

    pages = case.get("pages", [])

    pages_to_show = [1, 2, 3, 10, 50, 100, 200, 300, 380]

    for page_number in pages_to_show:

        matching_pages = [
            page for page in pages
            if page.get("page_number") == page_number
        ]

        if not matching_pages:
            continue

        page = matching_pages[0]

        print("\n" + "=" * 80)
        print(f"PAGE {page_number}")
        print("=" * 80)

        text = page.get("text", "")

        print(text[:5000])

    print("\n" + "=" * 80)
    print("INSPECTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()