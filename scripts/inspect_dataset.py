from pathlib import Path


# Project root = folder containing this script's parent directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def main():
    print("=" * 60)
    print("LEGAL QUERY RAG - DATASET INSPECTION")
    print("=" * 60)

    if not RAW_DATA_DIR.exists():
        print(f"ERROR: Raw data directory not found:")
        print(RAW_DATA_DIR)
        return

    court_folders = [
        folder for folder in RAW_DATA_DIR.iterdir()
        if folder.is_dir()
    ]

    total_pdfs = 0

    print(f"\nRaw data directory:")
    print(RAW_DATA_DIR)

    print("\nCourt-wise PDF count:")
    print("-" * 40)

    for court_folder in sorted(court_folders):
        pdf_files = list(court_folder.glob("*.pdf"))
        count = len(pdf_files)

        total_pdfs += count

        print(f"{court_folder.name:<25} {count}")

    print("-" * 40)
    print(f"{'TOTAL':<25} {total_pdfs}")

    print("\nDataset inspection completed.")


if __name__ == "__main__":
    main()