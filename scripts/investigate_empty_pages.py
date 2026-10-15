from pathlib import Path
import json


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


def main():

    print("=" * 80)
    print("EMPTY CLEANED PAGE INVESTIGATION")
    print("=" * 80)

    found = 0

    extracted_files = sorted(
        EXTRACTED_DIR.glob("*.json")
    )

    extracted_files = [
        file
        for file in extracted_files
        if file.name != "extraction_summary.json"
    ]

    for extracted_path in extracted_files:

        cleaned_path = CLEANED_DIR / extracted_path.name

        if not cleaned_path.exists():
            continue

        with open(
            extracted_path,
            "r",
            encoding="utf-8"
        ) as file:

            extracted_case = json.load(file)

        with open(
            cleaned_path,
            "r",
            encoding="utf-8"
        ) as file:

            cleaned_case = json.load(file)

        extracted_pages = extracted_case.get(
            "pages",
            []
        )

        cleaned_pages = cleaned_case.get(
            "pages",
            []
        )

        for original_page, cleaned_page in zip(
            extracted_pages,
            cleaned_pages
        ):

            cleaned_text = cleaned_page.get(
                "cleaned_text",
                ""
            )

            if not cleaned_text.strip():

                found += 1

                print("\n" + "=" * 80)
                print(f"EMPTY PAGE #{found}")
                print("=" * 80)

                print(
                    f"Case ID: "
                    f"{extracted_case.get('case_id')}"
                )

                print(
                    f"Court: "
                    f"{extracted_case.get('court')}"
                )

                print(
                    f"Page number: "
                    f"{original_page.get('page_number')}"
                )

                original_text = original_page.get(
                    "text",
                    ""
                )

                print(
                    f"Original characters: "
                    f"{len(original_text)}"
                )

                print(
                    "\nORIGINAL TEXT:"
                )

                print("-" * 80)

                print(
                    repr(original_text)
                )

                print("-" * 80)

    print("\n" + "=" * 80)
    print("INVESTIGATION COMPLETE")
    print("=" * 80)

    print(
        f"\nTotal empty cleaned pages: {found}"
    )


if __name__ == "__main__":
    main()