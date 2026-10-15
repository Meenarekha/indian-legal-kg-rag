import json
import re
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

CLEANED_DIR = Path("data/processed/cleaned")
EXTRACTED_DIR = Path("data/processed/extracted")
CHUNKS_DIR = Path("data/processed/chunks")


# ============================================================
# SUSPICIOUS CHARACTER DETECTION
# ============================================================

def suspicious_character_score(text):
    """
    Estimate whether extracted text contains an unusually high
    amount of suspicious/non-standard characters.

    This is only a diagnostic score.
    It does NOT decide whether text should be deleted.
    """

    if not text:
        return 0.0

    suspicious = 0

    for char in text:

        code = ord(char)

        # Replacement character
        if code == 0xFFFD:
            suspicious += 1

        # Control characters
        elif code < 32 and char not in "\n\t\r":
            suspicious += 1

        # Common mojibake / unusual symbol ranges
        elif 0x0080 <= code <= 0x009F:
            suspicious += 1

        elif 0x2000 <= code <= 0x206F:
            # Some punctuation is legitimate, so count lightly
            suspicious += 0.25

        elif 0x20A0 <= code <= 0x20CF:
            # Currency symbols can be legitimate
            suspicious += 0.1

    return suspicious / len(text)


# ============================================================
# NON-ASCII CHARACTER REPORT
# ============================================================

def character_report(text):
    """
    Display non-ASCII characters and their Unicode code points.
    """

    counts = {}

    for char in text:

        if ord(char) > 127:

            key = (
                char,
                f"U+{ord(char):04X}"
            )

            counts[key] = counts.get(key, 0) + 1

    return sorted(
        counts.items(),
        key=lambda item: item[1],
        reverse=True
    )


# ============================================================
# FIND SUSPICIOUS CHUNKS
# ============================================================

def find_suspicious_chunks():

    suspicious_chunks = []

    chunk_files = sorted(
        CHUNKS_DIR.glob("*.json")
    )

    for chunk_file in chunk_files:

        try:

            with open(
                chunk_file,
                "r",
                encoding="utf-8"
            ) as f:

                case_data = json.load(f)

        except Exception as e:

            print(
                f"Could not read {chunk_file.name}: {e}"
            )

            continue

        for chunk in case_data.get("chunks", []):

            text = chunk.get("text", "")

            if not text:
                continue

            score = suspicious_character_score(text)

            # Diagnostic threshold
            if score >= 0.15:

                suspicious_chunks.append({
                    "chunk_id": chunk.get("chunk_id"),
                    "case_id": chunk.get("case_id"),
                    "court": chunk.get("court"),
                    "pages": chunk.get("pages", []),
                    "paragraph_start": chunk.get(
                        "paragraph_start"
                    ),
                    "paragraph_end": chunk.get(
                        "paragraph_end"
                    ),
                    "character_count": len(text),
                    "score": score,
                    "text": text
                })

    return suspicious_chunks


# ============================================================
# TRACE PAGE BACK TO SOURCE
# ============================================================

def load_page_text(directory, case_id, page_number):

    file_path = directory / f"{case_id}.json"

    if not file_path.exists():
        return None

    try:

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as f:

            case_data = json.load(f)

    except Exception:
        return None

    for page in case_data.get("pages", []):

        if page.get("page_number") == page_number:

            return page

    return None


# ============================================================
# PRINT INVESTIGATION
# ============================================================

def investigate_chunk(chunk):

    print()
    print("=" * 80)
    print("SUSPICIOUS CHUNK")
    print("=" * 80)

    print(
        f"Chunk ID          : {chunk['chunk_id']}"
    )

    print(
        f"Case ID           : {chunk['case_id']}"
    )

    print(
        f"Court             : {chunk['court']}"
    )

    print(
        f"Pages             : {chunk['pages']}"
    )

    print(
        f"Paragraph         : "
        f"{chunk['paragraph_start']} - "
        f"{chunk['paragraph_end']}"
    )

    print(
        f"Characters        : {chunk['character_count']}"
    )

    print(
        f"Suspicious score  : "
        f"{chunk['score']:.3f}"
    )

    print()
    print("CHUNK TEXT")
    print("-" * 80)
    print(chunk["text"])

    print()
    print("CHARACTER ANALYSIS")
    print("-" * 80)

    report = character_report(
        chunk["text"]
    )

    if not report:

        print("No non-ASCII characters.")

    else:

        for (char, code), count in report[:30]:

            print(
                f"{repr(char):12} "
                f"{code:8} "
                f"count={count}"
            )

    # --------------------------------------------------------
    # Trace every page
    # --------------------------------------------------------

    for page_number in chunk["pages"]:

        print()
        print("=" * 80)
        print(
            f"SOURCE PAGE {page_number}"
        )
        print("=" * 80)

        cleaned_page = load_page_text(
            CLEANED_DIR,
            chunk["case_id"],
            page_number
        )

        extracted_page = load_page_text(
            EXTRACTED_DIR,
            chunk["case_id"],
            page_number
        )

        # ----------------------------------------------------
        # Cleaned page
        # ----------------------------------------------------

        print()
        print("CLEANED TEXT")
        print("-" * 80)

        if cleaned_page:

            print(
                cleaned_page.get(
                    "cleaned_text",
                    ""
                )
            )

        else:

            print(
                "Cleaned page could not be found."
            )

        # ----------------------------------------------------
        # Original extracted page
        # ----------------------------------------------------

        print()
        print("ORIGINAL EXTRACTED TEXT")
        print("-" * 80)

        if extracted_page:

            print(
                extracted_page.get(
                    "text",
                    ""
                )
            )

        else:

            print(
                "Original extracted page could not be found."
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("SUSPICIOUS ENCODING INVESTIGATION")
    print("=" * 80)

    suspicious_chunks = find_suspicious_chunks()

    print()
    print(
        f"Suspicious chunks found: "
        f"{len(suspicious_chunks)}"
    )

    # --------------------------------------------------------
    # Limit detailed output.
    #
    # We don't want hundreds of pages of output if many
    # documents contain unusual but legitimate Unicode.
    # --------------------------------------------------------

    inspect_count = min(
        len(suspicious_chunks),
        10
    )

    print(
        f"Detailed inspection: "
        f"first {inspect_count}"
    )

    for chunk in suspicious_chunks[:inspect_count]:

        investigate_chunk(chunk)

    print()
    print("=" * 80)
    print("INVESTIGATION COMPLETE")
    print("=" * 80)

    print()
    print(
        "No data was modified by this script."
    )


if __name__ == "__main__":
    main()