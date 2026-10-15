from pathlib import Path
import json
import pymupdf


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "extracted"


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pdf(pdf_path: Path, court_name: str):
    """
    Extract text from a PDF page by page.

    The original page boundaries are preserved so that
    page-level evidence can be traced later.
    """

    document = pymupdf.open(pdf_path)

    pages = []
    total_characters = 0
    pages_with_text = 0

    for page_number, page in enumerate(document, start=1):

        text = page.get_text("text")

        text = text.replace("\x00", "")

        character_count = len(text)

        total_characters += character_count

        if text.strip():
            pages_with_text += 1

        pages.append({
            "page_number": page_number,
            "text": text,
            "character_count": character_count
        })

    total_pages = len(document)

    document.close()

    return {
        "case_id": pdf_path.stem,
        "court": court_name,
        "source_file": pdf_path.name,
        "source_path": str(pdf_path.relative_to(PROJECT_ROOT)),
        "page_count": total_pages,
        "pages_with_text": pages_with_text,
        "pages_without_text": total_pages - pages_with_text,
        "total_characters": total_characters,
        "pages": pages
    }


# ============================================================
# SAVE JSON
# ============================================================

def save_extracted_case(case_data):
    """
    Save one extracted case as a JSON file.
    """

    output_file = OUTPUT_DIR / f"{case_data['case_id']}.json"

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(
            case_data,
            file,
            ensure_ascii=False,
            indent=2
        )


# ============================================================
# DATASET EXTRACTION
# ============================================================

def main():

    print("=" * 70)
    print("LEGAL QUERY RAG - PDF EXTRACTION")
    print("=" * 70)

    if not RAW_DATA_DIR.exists():
        print("\nERROR:")
        print(f"Raw data directory does not exist:\n{RAW_DATA_DIR}")
        return

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    pdf_files = []

    # Search recursively so nested folders are also supported.
    for pdf_path in RAW_DATA_DIR.rglob("*"):

        if pdf_path.is_file() and pdf_path.suffix.lower() == ".pdf":
            pdf_files.append(pdf_path)

    pdf_files.sort()

    total_files = len(pdf_files)

    print(f"\nRaw data directory:")
    print(RAW_DATA_DIR)

    print(f"\nPDF files found: {total_files}")

    if total_files == 0:
        print("\nERROR: No PDF files found.")
        return

    successful = 0
    failed = 0

    suspicious_files = []

    total_pages = 0
    total_characters = 0

    print("\nStarting extraction...")
    print("-" * 70)

    for index, pdf_path in enumerate(pdf_files, start=1):

        # Court name is the immediate parent folder.
        court_name = pdf_path.parent.name

        print(
            f"[{index:03d}/{total_files}] "
            f"{court_name} / {pdf_path.name}"
        )

        try:

            case_data = extract_pdf(
                pdf_path,
                court_name
            )

            save_extracted_case(case_data)

            successful += 1

            total_pages += case_data["page_count"]
            total_characters += case_data["total_characters"]

            # Flag documents where pages contain no extractable text.
            if case_data["pages_without_text"] > 0:

                suspicious_files.append({
                    "file": str(
                        pdf_path.relative_to(PROJECT_ROOT)
                    ),
                    "court": court_name,
                    "pages": case_data["page_count"],
                    "pages_without_text": case_data["pages_without_text"],
                    "total_characters": case_data["total_characters"]
                })

        except Exception as error:

            failed += 1

            print(
                f"    ERROR: {error}"
            )

    # ========================================================
    # EXTRACTION SUMMARY
    # ========================================================

    summary = {
        "total_pdf_files": total_files,
        "successful": successful,
        "failed": failed,
        "total_pages": total_pages,
        "total_characters": total_characters,
        "documents_with_empty_pages": len(suspicious_files),
        "suspicious_files": suspicious_files
    }

    summary_file = OUTPUT_DIR / "extraction_summary.json"

    with open(
        summary_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2
        )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)

    print(f"Total PDFs:              {total_files}")
    print(f"Successfully extracted:  {successful}")
    print(f"Failed:                  {failed}")
    print(f"Total pages:             {total_pages}")
    print(f"Total characters:        {total_characters}")
    print(
        f"Documents with empty pages: "
        f"{len(suspicious_files)}"
    )

    print("\nOutput directory:")
    print(OUTPUT_DIR)

    print("\nExtraction summary:")
    print(summary_file)

    print("=" * 70)


if __name__ == "__main__":
    main()