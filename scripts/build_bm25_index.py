import json
import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


# ============================================================
# PATHS
# ============================================================

CHUNKS_DIR = Path("data/processed/chunks")

INDEX_DIR = Path("data/indexes/bm25")

INDEX_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize(text):
    """
    Convert legal text into BM25 tokens.

    The tokenizer:
    - converts text to lowercase
    - keeps letters and numbers
    - separates punctuation
    - preserves important numeric legal references
      such as section numbers
    """

    if not text:
        return []

    text = text.lower()

    # Extract words and numbers.
    #
    # Examples:
    # "Section 302 IPC"
    # becomes:
    # ["section", "302", "ipc"]
    #
    # "Article 21"
    # becomes:
    # ["article", "21"]
    tokens = re.findall(
        r"[a-z]+|\d+(?:\.\d+)*",
        text
    )

    return tokens


# ============================================================
# LOAD CHUNKS
# ============================================================

def load_chunks():

    all_chunks = []

    chunk_files = sorted(
        CHUNKS_DIR.glob("*.json")
    )

    print(
        f"Chunk files found: {len(chunk_files)}"
    )

    for chunk_file in chunk_files:

        with open(
            chunk_file,
            "r",
            encoding="utf-8"
        ) as f:

            case_data = json.load(f)

        for chunk in case_data.get(
            "chunks",
            []
        ):

            text = chunk.get(
                "text",
                ""
            )

            if not text.strip():
                continue

            all_chunks.append({
                "chunk_id": chunk.get(
                    "chunk_id"
                ),
                "case_id": chunk.get(
                    "case_id"
                ),
                "court": chunk.get(
                    "court"
                ),
                "source_file": chunk.get(
                    "source_file"
                ),
                "start_page": chunk.get(
                    "start_page"
                ),
                "end_page": chunk.get(
                    "end_page"
                ),
                "pages": chunk.get(
                    "pages",
                    []
                ),
                "paragraph_start": chunk.get(
                    "paragraph_start"
                ),
                "paragraph_end": chunk.get(
                    "paragraph_end"
                ),
                "text": text
            })

    return all_chunks


# ============================================================
# BUILD BM25
# ============================================================

def build_index(chunks):

    tokenized_documents = []

    valid_chunks = []

    for chunk in chunks:

        tokens = tokenize(
            chunk["text"]
        )

        if not tokens:
            continue

        tokenized_documents.append(
            tokens
        )

        valid_chunks.append(
            chunk
        )

    print(
        f"Valid chunks for BM25: "
        f"{len(valid_chunks)}"
    )

    print(
        "Building BM25 index..."
    )

    bm25 = BM25Okapi(
        tokenized_documents
    )

    return bm25, valid_chunks


# ============================================================
# SAVE INDEX
# ============================================================

def save_index(
    bm25,
    chunks
):

    index_path = (
        INDEX_DIR /
        "bm25_index.pkl"
    )

    metadata_path = (
        INDEX_DIR /
        "bm25_metadata.json"
    )

    # --------------------------------------------------------
    # Save BM25 object
    # --------------------------------------------------------

    with open(
        index_path,
        "wb"
    ) as f:

        pickle.dump(
            bm25,
            f
        )

    # --------------------------------------------------------
    # Save chunk metadata separately
    # --------------------------------------------------------

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            chunks,
            f,
            ensure_ascii=False,
            indent=2
        )

    print()
    print(
        f"BM25 index saved to:"
    )

    print(
        index_path.resolve()
    )

    print()
    print(
        "BM25 metadata saved to:"
    )

    print(
        metadata_path.resolve()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("BM25 INDEX BUILDING")
    print("=" * 80)

    # --------------------------------------------------------
    # Load chunks
    # --------------------------------------------------------

    chunks = load_chunks()

    print(
        f"Total chunks loaded: "
        f"{len(chunks)}"
    )

    if not chunks:

        raise ValueError(
            "No chunks found."
        )

    # --------------------------------------------------------
    # Build index
    # --------------------------------------------------------

    bm25, valid_chunks = build_index(
        chunks
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_index(
        bm25,
        valid_chunks
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("BM25 INDEX COMPLETE")
    print("=" * 80)

    print(
        f"Input chunks:       {len(chunks)}"
    )

    print(
        f"Indexed chunks:     {len(valid_chunks)}"
    )

    print(
        f"Skipped chunks:     "
        f"{len(chunks) - len(valid_chunks)}"
    )

    print()
    print(
        "Index directory:"
    )

    print(
        INDEX_DIR.resolve()
    )

    print("=" * 80)


if __name__ == "__main__":
    main()