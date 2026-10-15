from pathlib import Path
import fitz


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def extract_pdf_text(pdf_path: Path):
    """
    Extract text page by page from a PDF using PyMuPDF.
    """

    document = fitz.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document, start=1):
        text = page.get_text("text")

        pages.append({
            "page_number": page_number,
            "text": text
        })

    document.close()

    return pages


def inspect_pdf(pdf_path: Path):
    print("=" * 70)
    print("PDF INSPECTION")
    print("=" * 70)

    print(f"\nFile:")
    print(pdf_path)

    print(f"\nFile size:")
    print(f"{pdf_path.stat().st_size / 1024:.2f} KB")

    pages = extract_pdf_text(pdf_path)

    print(f"\nNumber of pages: {len(pages)}")

    total_characters = 0
    pages_with_text = 0
    pages_without_text = 0

    for page in pages:
        text = page["text"]
        character_count = len(text)

        total_characters += character_count

        if text.strip():
            pages_with_text += 1
        else:
            pages_without_text += 1

    print(f"Pages containing text: {pages_with_text}")
    print(f"Pages without text: {pages_without_text}")
    print(f"Total extracted characters: {total_characters}")

    print("\n" + "-" * 70)
    print("FIRST PAGE EXTRACTED TEXT")
    print("-" * 70)

    if pages:
        first_page_text = pages[0]["text"]

        if first_page_text.strip():
            print(first_page_text[:4000])
        else:
            print("[NO TEXT EXTRACTED FROM FIRST PAGE]")

    print("\n" + "=" * 70)


def main():

    sample_files = []

    court_directories = [
        RAW_DATA_DIR / "Supreme Court",
        RAW_DATA_DIR / "high court",
        RAW_DATA_DIR / "District Court",
    ]

    for court_directory in court_directories:

        if not court_directory.exists():
            print(f"WARNING: Directory not found: {court_directory}")
            continue

        pdf_files = sorted(court_directory.glob("*.pdf"))

        if pdf_files:
            sample_files.append(pdf_files[0])

    if not sample_files:
        print("ERROR: No PDF files found.")
        return

    for pdf_path in sample_files:
        inspect_pdf(pdf_path)


if __name__ == "__main__":
    main()