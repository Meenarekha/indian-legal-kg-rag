import json
import pickle
import re
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

INDEX_DIR = Path("data/indexes/bm25")

INDEX_PATH = INDEX_DIR / "bm25_index.pkl"
METADATA_PATH = INDEX_DIR / "bm25_metadata.json"


# ============================================================
# TOKENIZER
# ============================================================

def tokenize(text):
    """
    Tokenize a legal query using the same tokenization
    strategy used while building the BM25 index.
    """

    if not text:
        return []

    text = text.lower()

    tokens = re.findall(
        r"[a-z]+|\d+(?:\.\d+)*",
        text
    )

    return tokens


# ============================================================
# LOAD INDEX
# ============================================================

def load_bm25():

    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"BM25 index not found: {INDEX_PATH}"
        )

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"BM25 metadata not found: {METADATA_PATH}"
        )

    with open(
        INDEX_PATH,
        "rb"
    ) as f:

        bm25 = pickle.load(f)

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        metadata = json.load(f)

    return bm25, metadata


# ============================================================
# SEARCH
# ============================================================

def search_bm25(
    query,
    top_k=10
):

    bm25, metadata = load_bm25()

    query_tokens = tokenize(query)

    if not query_tokens:

        return []

    scores = bm25.get_scores(
        query_tokens
    )

    # --------------------------------------------------------
    # Sort by descending BM25 score
    # --------------------------------------------------------

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True
    )

    results = []

    for rank, index in enumerate(
        ranked_indices[:top_k],
        start=1
    ):

        chunk = metadata[index]

        results.append({
            "rank": rank,
            "score": float(
                scores[index]
            ),
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
            "text": chunk.get(
                "text",
                ""
            )
        })

    return results


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(
    query,
    results
):

    print()
    print("=" * 80)
    print("BM25 SEARCH RESULTS")
    print("=" * 80)

    print()
    print(
        f"Query: {query}"
    )

    print(
        f"Results: {len(results)}"
    )

    for result in results:

        print()
        print("-" * 80)

        print(
            f"Rank       : {result['rank']}"
        )

        print(
            f"BM25 Score : {result['score']:.4f}"
        )

        print(
            f"Case ID    : {result['case_id']}"
        )

        print(
            f"Court      : {result['court']}"
        )

        print(
            f"Pages      : {result['pages']}"
        )

        print(
            f"Paragraph  : "
            f"{result['paragraph_start']} - "
            f"{result['paragraph_end']}"
        )

        print()

        print(
            result["text"][:700]
        )

        if len(result["text"]) > 700:

            print("...")


# ============================================================
# INTERACTIVE SEARCH
# ============================================================

def main():

    print("=" * 80)
    print("BM25 LEGAL SEARCH")
    print("=" * 80)

    print()
    print(
        "Enter a legal query."
    )

    print(
        "Type 'exit' to stop."
    )

    while True:

        query = input(
            "\nQuery: "
        ).strip()

        if query.lower() == "exit":

            print(
                "Exiting BM25 search."
            )

            break

        if not query:

            print(
                "Please enter a query."
            )

            continue

        try:

            results = search_bm25(
                query,
                top_k=5
            )

            display_results(
                query,
                results
            )

        except Exception as e:

            print()
            print(
                f"Search error: {e}"
            )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    main()