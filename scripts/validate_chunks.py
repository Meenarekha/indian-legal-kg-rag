import json
from pathlib import Path
from collections import Counter

# ============================================================
# PATH CONFIGURATION
# ============================================================

CHUNK_DIR = Path("data/processed/chunks")

# Expected chunk-size boundaries
MIN_EXPECTED_SIZE = 300
MAX_EXPECTED_SIZE = 2400


# ============================================================
# LOAD CHUNKS
# ============================================================

def load_chunks():
    files = sorted(CHUNK_DIR.glob("*.json"))

    all_chunks = []

    for file in files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                case_data = json.load(f)

            chunks = case_data.get("chunks", [])

            for chunk in chunks:
                chunk["_source_file"] = file.name
                all_chunks.append(chunk)

        except Exception as e:
            print(f"ERROR reading {file.name}: {e}")

    return files, all_chunks


# ============================================================
# PRINT CHUNK
# ============================================================

def print_chunk(chunk, number):
    print("\n" + "=" * 80)
    print(f"CHUNK #{number}")
    print("=" * 80)

    print(f"Chunk ID       : {chunk.get('chunk_id')}")
    print(f"Case ID        : {chunk.get('case_id')}")
    print(f"Court          : {chunk.get('court')}")
    print(f"Pages          : {chunk.get('pages')}")
    print(f"Start page     : {chunk.get('start_page')}")
    print(f"End page       : {chunk.get('end_page')}")
    print(f"Paragraph start: {chunk.get('paragraph_start')}")
    print(f"Paragraph end  : {chunk.get('paragraph_end')}")
    print(f"Character count: {chunk.get('character_count')}")

    print("\nTEXT:")
    print("-" * 80)
    print(chunk.get("text", ""))


# ============================================================
# MAIN VALIDATION
# ============================================================

def main():

    print("=" * 80)
    print("LEGAL CHUNK QUALITY VALIDATION")
    print("=" * 80)

    if not CHUNK_DIR.exists():
        print("\nERROR: Chunk directory does not exist.")
        print(f"Expected: {CHUNK_DIR.resolve()}")
        return

    files, chunks = load_chunks()

    print(f"\nCase files found: {len(files)}")
    print(f"Total chunks   : {len(chunks)}")

    if not chunks:
        print("\nERROR: No chunks found.")
        return

    # --------------------------------------------------------
    # CHUNK SIZE ANALYSIS
    # --------------------------------------------------------

    sizes = [
        len(chunk.get("text", "").strip())
        for chunk in chunks
    ]

    below_300 = [
        chunk for chunk in chunks
        if len(chunk.get("text", "").strip()) < MIN_EXPECTED_SIZE
    ]

    below_100 = [
        chunk for chunk in chunks
        if len(chunk.get("text", "").strip()) < 100
    ]

    below_50 = [
        chunk for chunk in chunks
        if len(chunk.get("text", "").strip()) < 50
    ]

    above_max = [
        chunk for chunk in chunks
        if len(chunk.get("text", "").strip()) > MAX_EXPECTED_SIZE
    ]

    print("\n" + "-" * 80)
    print("CHUNK SIZE ANALYSIS")
    print("-" * 80)

    print(f"Minimum chunk size       : {min(sizes)}")
    print(f"Maximum chunk size       : {max(sizes)}")
    print(f"Average chunk size       : {sum(sizes) / len(sizes):.2f}")

    sorted_sizes = sorted(sizes)
    median = sorted_sizes[len(sorted_sizes) // 2]

    print(f"Median chunk size        : {median}")

    print(f"\nChunks below 300 chars   : {len(below_300)}")
    print(f"Chunks below 100 chars   : {len(below_100)}")
    print(f"Chunks below 50 chars    : {len(below_50)}")
    print(f"Chunks above 2400 chars  : {len(above_max)}")

    # --------------------------------------------------------
    # SHOW SMALLEST CHUNKS
    # --------------------------------------------------------

    smallest_chunks = sorted(
        chunks,
        key=lambda x: len(x.get("text", "").strip())
    )[:20]

    print("\n" + "=" * 80)
    print("20 SMALLEST CHUNKS")
    print("=" * 80)

    for i, chunk in enumerate(smallest_chunks, start=1):
        print_chunk(chunk, i)

    # --------------------------------------------------------
    # CHECK REQUIRED METADATA
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("METADATA VALIDATION")
    print("=" * 80)

    required_fields = [
        "chunk_id",
        "case_id",
        "court",
        "source_file",
        "start_page",
        "end_page",
        "pages",
        "paragraph_start",
        "paragraph_end",
        "character_count",
        "text"
    ]

    missing_metadata = []

    for chunk in chunks:

        missing = [
            field
            for field in required_fields
            if field not in chunk
        ]

        if missing:
            missing_metadata.append(
                (chunk.get("chunk_id"), missing)
            )

    if missing_metadata:
        print(f"Chunks with missing metadata: {len(missing_metadata)}")

        for chunk_id, missing in missing_metadata[:20]:
            print(f"  {chunk_id}: missing {missing}")

    else:
        print("All chunks contain required metadata: PASS")

    # --------------------------------------------------------
    # EMPTY TEXT CHECK
    # --------------------------------------------------------

    empty_chunks = [
        chunk
        for chunk in chunks
        if not chunk.get("text", "").strip()
    ]

    print("\n" + "=" * 80)
    print("EMPTY CHUNK CHECK")
    print("=" * 80)

    print(f"Empty chunks: {len(empty_chunks)}")

    # --------------------------------------------------------
    # DUPLICATE CHECK
    # --------------------------------------------------------

    text_counter = Counter(
        chunk.get("text", "").strip()
        for chunk in chunks
    )

    duplicates = {
        text: count
        for text, count in text_counter.items()
        if text and count > 1
    }

    print("\n" + "=" * 80)
    print("DUPLICATE TEXT CHECK")
    print("=" * 80)

    print(f"Unique chunk texts : {len(text_counter)}")
    print(f"Duplicated texts   : {len(duplicates)}")

    if duplicates:
        print("\nTop duplicated chunk texts:")

        duplicate_items = sorted(
            duplicates.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        for text, count in duplicate_items:

            preview = text.replace("\n", " ")

            if len(preview) > 200:
                preview = preview[:200] + "..."

            print(f"\nCount: {count}")
            print(f"Text : {preview}")

    # --------------------------------------------------------
    # COURT DISTRIBUTION
    # --------------------------------------------------------

    court_counts = Counter(
        chunk.get("court", "UNKNOWN")
        for chunk in chunks
    )

    print("\n" + "=" * 80)
    print("CHUNK DISTRIBUTION BY COURT")
    print("=" * 80)

    for court, count in court_counts.items():
        print(f"{court:<25} : {count}")

    # --------------------------------------------------------
    # CASE DISTRIBUTION
    # --------------------------------------------------------

    case_chunk_counts = Counter(
        chunk.get("case_id", "UNKNOWN")
        for chunk in chunks
    )

    print("\n" + "=" * 80)
    print("CASE-LEVEL CHUNK DISTRIBUTION")
    print("=" * 80)

    print(f"Cases represented: {len(case_chunk_counts)}")

    counts = list(case_chunk_counts.values())

    print(f"Minimum chunks/case: {min(counts)}")
    print(f"Maximum chunks/case: {max(counts)}")
    print(f"Average chunks/case: {sum(counts) / len(counts):.2f}")

    # --------------------------------------------------------
    # FINAL STATUS
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    print(f"Total chunks             : {len(chunks)}")
    print(f"Below 300 characters     : {len(below_300)}")
    print(f"Above 2400 characters    : {len(above_max)}")
    print(f"Empty chunks             : {len(empty_chunks)}")
    print(f"Missing metadata chunks  : {len(missing_metadata)}")
    print(f"Duplicated text patterns : {len(duplicates)}")

    print("\n" + "=" * 80)
    print("NEXT DECISION")
    print("=" * 80)

    if len(below_300) > 0:
        print(
            "WARNING: Small chunks exist."
        )
        print(
            "Inspect the smallest chunks before modifying the chunker."
        )
    else:
        print(
            "No chunks below the expected minimum."
        )

    if len(above_max) == 0:
        print("Maximum-size validation: PASS")
    else:
        print("Maximum-size validation: FAIL")

    if len(empty_chunks) == 0:
        print("Empty-chunk validation: PASS")
    else:
        print("Empty-chunk validation: FAIL")

    if len(missing_metadata) == 0:
        print("Metadata validation: PASS")
    else:
        print("Metadata validation: FAIL")

    print("=" * 80)


if __name__ == "__main__":
    main()