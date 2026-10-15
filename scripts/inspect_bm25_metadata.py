import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

METADATA_PATH = (
    BASE_DIR
    / "data"
    / "indexes"
    / "bm25"
    / "bm25_metadata.json"
)


def main():

    print("=" * 80)
    print("BM25 METADATA INSPECTION")
    print("=" * 80)

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print(f"\nTotal metadata records: {len(metadata)}")

    if not metadata:
        print("ERROR: Metadata file is empty.")
        return

    first_record = metadata[0]

    print("\nFirst metadata record:")
    print("-" * 80)

    for key, value in first_record.items():
        print(f"{key}: {value}")

    print("\nMetadata keys:")
    print("-" * 80)

    for key in first_record.keys():
        print(key)


if __name__ == "__main__":
    main()