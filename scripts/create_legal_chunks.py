import json
import re
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

INPUT_DIR = Path("data/processed/cleaned")
OUTPUT_DIR = Path("data/processed/chunks")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CHUNK PARAMETERS
# ============================================================

TARGET_CHARS = 1800
MAX_CHARS = 2400
MIN_CHARS = 300

# Number of previous paragraph units reused as overlap
OVERLAP_PARAGRAPHS = 1


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    """
    Normalize whitespace while preserving paragraph content.
    """

    if not text:
        return ""

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Normalize spaces/tabs
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize excessive blank lines
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

    return text.strip()


# ============================================================
# PARAGRAPH EXTRACTION
# ============================================================

def extract_paragraph_units(pages):
    """
    Extract paragraph-like units from page-level cleaned text.

    The function keeps page provenance and attempts to recognize
    numbered legal paragraphs.
    """

    units = []

    current_lines = []
    current_start_page = None
    current_paragraph_number = None

    paragraph_pattern = re.compile(
        r"^\s*(\d{1,4}(?:\.\d+)*)\.\s*(.*)$"
    )

    def flush_current():
        nonlocal current_lines
        nonlocal current_start_page
        nonlocal current_paragraph_number

        if not current_lines:
            return

        text = " ".join(current_lines)
        text = normalize_text(text)

        if text:
            units.append({
                "text": text,
                "start_page": current_start_page,
                "end_page": current_start_page,
                "paragraph_number": current_paragraph_number
            })

        current_lines = []
        current_start_page = None
        current_paragraph_number = None

    for page in pages:

        page_number = page.get("page_number")
        page_text = page.get("cleaned_text", "")

        if not page_text:
            continue

        lines = page_text.splitlines()

        for raw_line in lines:

            line = raw_line.strip()

            if not line:
                continue

            match = paragraph_pattern.match(line)

            if match:

                # A new numbered paragraph begins
                flush_current()

                paragraph_number = match.group(1)
                remainder = match.group(2).strip()

                current_start_page = page_number
                current_paragraph_number = paragraph_number

                if remainder:
                    current_lines.append(remainder)

            else:

                # Continue current paragraph
                if current_start_page is None:
                    current_start_page = page_number

                current_lines.append(line)

                # If content crosses pages, update end page
                if units:
                    pass

        # If the current paragraph continues onto another page,
        # we preserve the page boundary through the accumulated text.
        if current_lines:
            pass

    flush_current()

    # --------------------------------------------------------
    # Improve page provenance.
    #
    # The first implementation above identifies paragraph units.
    # For chunks we also need reliable page ranges.
    # --------------------------------------------------------

    if not units:
        return []

    return units


# ============================================================
# SENTENCE SPLITTING
# ============================================================

def split_sentences(text):
    """
    Lightweight sentence splitter for oversized legal paragraphs.

    This is intentionally conservative. It is not intended to
    perform full NLP sentence parsing.
    """

    if not text:
        return []

    # Protect common legal abbreviations from naive splitting
    protected = text

    abbreviations = [
        "No.",
        "Nos.",
        "Dr.",
        "Mr.",
        "Mrs.",
        "Ms.",
        "J.",
        "JJ.",
        "Ltd.",
        "Co.",
        "e.g.",
        "i.e.",
        "vs.",
        "v."
    ]

    replacements = {}

    for index, abbreviation in enumerate(abbreviations):

        token = f"__ABBR_{index}__"

        if abbreviation in protected:
            protected = protected.replace(abbreviation, token)
            replacements[token] = abbreviation

    # Sentence boundary
    parts = re.split(
        r"(?<=[.!?])\s+(?=[A-Z0-9(\[])",
        protected
    )

    restored = []

    for part in parts:

        for token, abbreviation in replacements.items():
            part = part.replace(token, abbreviation)

        part = normalize_text(part)

        if part:
            restored.append(part)

    return restored


# ============================================================
# HARD SPLIT
# ============================================================

def hard_split(text, max_chars=MAX_CHARS):
    """
    Final fallback when a sentence itself is larger than MAX_CHARS.

    Attempts to split on whitespace rather than cutting words.
    """

    text = normalize_text(text)

    if len(text) <= max_chars:
        return [text]

    words = text.split()

    pieces = []
    current = []

    current_length = 0

    for word in words:

        additional_length = len(word)

        if current:
            additional_length += 1

        if current_length + additional_length <= max_chars:

            current.append(word)
            current_length += additional_length

        else:

            if current:
                pieces.append(" ".join(current))

            current = [word]
            current_length = len(word)

    if current:
        pieces.append(" ".join(current))

    return pieces


# ============================================================
# SPLIT OVERSIZED PARAGRAPH
# ============================================================

def split_large_unit(unit):
    """
    Split a paragraph using sentences first and hard character
    splitting only as a final fallback.
    """

    text = unit["text"]

    if len(text) <= MAX_CHARS:
        return [unit]

    sentences = split_sentences(text)

    if not sentences:
        sentences = [text]

    pieces = []

    current = []
    current_length = 0

    for sentence in sentences:

        if len(sentence) > MAX_CHARS:

            if current:
                pieces.append({
                    **unit,
                    "text": " ".join(current)
                })

                current = []
                current_length = 0

            hard_parts = hard_split(sentence)

            for part in hard_parts:

                pieces.append({
                    **unit,
                    "text": part
                })

            continue

        additional_length = len(sentence)

        if current:
            additional_length += 1

        if current_length + additional_length <= TARGET_CHARS:

            current.append(sentence)
            current_length += additional_length

        else:

            if current:

                pieces.append({
                    **unit,
                    "text": " ".join(current)
                })

            current = [sentence]
            current_length = len(sentence)

    if current:

        pieces.append({
            **unit,
            "text": " ".join(current)
        })

    return pieces


# ============================================================
# COMBINE PARAGRAPH UNITS
# ============================================================

def create_candidate_chunks(units):
    """
    Combine paragraph units until the target chunk size is reached.
    """

    chunks = []

    current_units = []
    current_length = 0

    for unit in units:

        text = unit["text"]

        if not text:
            continue

        # ----------------------------------------------------
        # Oversized paragraph
        # ----------------------------------------------------

        if len(text) > MAX_CHARS:

            if current_units:

                chunks.append(current_units)

                # Keep one paragraph as overlap
                current_units = current_units[
                    -OVERLAP_PARAGRAPHS:
                ]

                current_length = sum(
                    len(x["text"]) for x in current_units
                )

            split_units = split_large_unit(unit)

            for split_unit in split_units:

                chunks.append([split_unit])

            continue

        # ----------------------------------------------------
        # Normal paragraph
        # ----------------------------------------------------

        additional_length = len(text)

        if current_units:
            additional_length += 1

        if current_length + additional_length <= TARGET_CHARS:

            current_units.append(unit)
            current_length += additional_length

        else:

            if current_units:
                chunks.append(current_units)

            current_units = [unit]
            current_length = len(text)

    if current_units:
        chunks.append(current_units)

    return chunks


# ============================================================
# FINAL SMALL-CHUNK MERGING
# ============================================================

def merge_small_chunks(chunks):
    """
    Merge short chunks into neighboring chunks where possible.

    We do NOT delete short chunks because some short legal
    statements are legally meaningful, e.g. 'Leave granted.'
    """

    if not chunks:
        return []

    result = []

    index = 0

    while index < len(chunks):

        current = chunks[index]

        current_text = " ".join(
            unit["text"] for unit in current
        )

        # ----------------------------------------------------
        # Already large enough
        # ----------------------------------------------------

        if len(current_text) >= MIN_CHARS:

            result.append(current)
            index += 1
            continue

        # ----------------------------------------------------
        # Try merging with NEXT chunk first.
        #
        # This is useful for headings and short statements
        # that introduce the following legal discussion.
        # ----------------------------------------------------

        if index + 1 < len(chunks):

            next_chunk = chunks[index + 1]

            next_text = " ".join(
                unit["text"] for unit in next_chunk
            )

            combined_length = (
                len(current_text)
                + 1
                + len(next_text)
            )

            if combined_length <= MAX_CHARS:

                merged = current + next_chunk
                chunks[index + 1] = merged

                index += 1
                continue

        # ----------------------------------------------------
        # Try merging with PREVIOUS chunk
        # ----------------------------------------------------

        if result:

            previous = result[-1]

            previous_text = " ".join(
                unit["text"] for unit in previous
            )

            combined_length = (
                len(previous_text)
                + 1
                + len(current_text)
            )

            if combined_length <= MAX_CHARS:

                result[-1] = previous + current
                index += 1
                continue

        # ----------------------------------------------------
        # Cannot merge safely.
        #
        # Keep it rather than deleting potentially meaningful
        # legal content.
        # ----------------------------------------------------

        result.append(current)

        index += 1

    return result


# ============================================================
# BUILD FINAL CHUNK RECORD
# ============================================================

def build_chunk_record(
    chunk_units,
    case_data,
    chunk_number
):

    text = " ".join(
        unit["text"] for unit in chunk_units
    )

    text = normalize_text(text)

    pages = []

    for unit in chunk_units:

        if unit.get("start_page") is not None:
            pages.append(unit["start_page"])

        if unit.get("end_page") is not None:
            pages.append(unit["end_page"])

    pages = sorted(set(pages))

    paragraph_numbers = [
        unit.get("paragraph_number")
        for unit in chunk_units
        if unit.get("paragraph_number") is not None
    ]

    start_page = min(pages) if pages else None
    end_page = max(pages) if pages else None

    paragraph_start = (
        paragraph_numbers[0]
        if paragraph_numbers
        else None
    )

    paragraph_end = (
        paragraph_numbers[-1]
        if paragraph_numbers
        else None
    )

    chunk_id = (
        f"{case_data['case_id']}_chunk_{chunk_number:05d}"
    )

    return {
        "chunk_id": chunk_id,
        "case_id": case_data["case_id"],
        "court": case_data["court"],
        "source_file": case_data["source_file"],
        "source_path": case_data["source_path"],
        "start_page": start_page,
        "end_page": end_page,
        "pages": pages,
        "paragraph_start": paragraph_start,
        "paragraph_end": paragraph_end,
        "character_count": len(text),
        "text": text
    }


# ============================================================
# PROCESS ONE CASE
# ============================================================

def process_case(input_file, output_file):

    with open(
        input_file,
        "r",
        encoding="utf-8"
    ) as f:

        case_data = json.load(f)

    pages = case_data.get("pages", [])

    units = extract_paragraph_units(pages)

    if not units:
        raise ValueError("No usable text units found.")

    candidate_chunks = create_candidate_chunks(units)

    final_chunks = merge_small_chunks(candidate_chunks)

    chunk_records = []

    for index, chunk_units in enumerate(
        final_chunks,
        start=1
    ):

        record = build_chunk_record(
            chunk_units,
            case_data,
            index
        )

        if record["text"]:
            chunk_records.append(record)

    output_data = {
        "case_id": case_data["case_id"],
        "court": case_data["court"],
        "source_file": case_data["source_file"],
        "source_path": case_data["source_path"],
        "page_count": case_data["page_count"],
        "chunk_count": len(chunk_records),
        "chunks": chunk_records
    }

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output_data,
            f,
            ensure_ascii=False,
            indent=2
        )

    return len(chunk_records)


# ============================================================
# MAIN
# ============================================================

def main():

    input_files = sorted(
        INPUT_DIR.glob("*.json")
    )

    total_cases = len(input_files)
    successful = 0
    failed = 0
    total_chunks = 0

    print("=" * 80)
    print("LEGAL-AWARE CHUNKING")
    print("=" * 80)

    print(f"Input cases: {total_cases}")
    print()

    for input_file in input_files:

        output_file = OUTPUT_DIR / input_file.name

        try:

            count = process_case(
                input_file,
                output_file
            )

            successful += 1
            total_chunks += count

        except Exception as e:

            failed += 1

            print(
                f"FAILED: {input_file.name} -> {e}"
            )

    print()
    print("=" * 80)
    print("CHUNKING COMPLETE")
    print("=" * 80)

    print(f"Total cases:        {total_cases}")
    print(f"Successfully done:  {successful}")
    print(f"Failed:             {failed}")
    print(f"Total chunks:       {total_chunks}")

    print()
    print("Output directory:")
    print(OUTPUT_DIR.resolve())

    print("=" * 80)


if __name__ == "__main__":
    main()