import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

INDEX_DIR = Path("data/indexes/faiss")

INDEX_PATH = INDEX_DIR / "faiss.index"
METADATA_PATH = INDEX_DIR / "faiss_metadata.json"
CONFIG_PATH = INDEX_DIR / "embedding_config.json"


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ============================================================
# LOAD FAISS
# ============================================================

def load_index():

    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {INDEX_PATH}"
        )

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"FAISS metadata not found: {METADATA_PATH}"
        )

    index = faiss.read_index(
        str(INDEX_PATH)
    )

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        metadata = json.load(f)

    return index, metadata


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print()
    print(
        "Loading embedding model:"
    )

    print(
        MODEL_NAME
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    print(
        f"Embedding dimension: "
        f"{model.get_embedding_dimension()}"
    )

    return model


# ============================================================
# SEARCH
# ============================================================

def search_faiss(
    query,
    model,
    index,
    metadata,
    top_k=5
):

    # --------------------------------------------------------
    # Convert query to embedding
    # --------------------------------------------------------

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # --------------------------------------------------------
    # FAISS expects float32
    # --------------------------------------------------------

    query_embedding = query_embedding.astype(
        "float32"
    )

    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for rank, (score, index_id) in enumerate(
        zip(
            scores[0],
            indices[0]
        ),
        start=1
    ):

        # FAISS uses -1 when no result exists
        if index_id < 0:
            continue

        chunk = metadata[index_id]

        results.append({
            "rank": rank,
            "score": float(score),
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
    print("FAISS SEMANTIC SEARCH RESULTS")
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
            f"Rank          : {result['rank']}"
        )

        print(
            f"Similarity    : "
            f"{result['score']:.4f}"
        )

        print(
            f"Case ID       : {result['case_id']}"
        )

        print(
            f"Court         : {result['court']}"
        )

        print(
            f"Pages         : {result['pages']}"
        )

        print(
            f"Paragraph     : "
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
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("FAISS LEGAL SEMANTIC SEARCH")
    print("=" * 80)

    index, metadata = load_index()

    print()
    print(
        f"FAISS vectors loaded: "
        f"{index.ntotal}"
    )

    model = load_model()

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
                "Exiting FAISS search."
            )

            break

        if not query:

            print(
                "Please enter a query."
            )

            continue

        try:

            results = search_faiss(
                query,
                model,
                index,
                metadata,
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


if __name__ == "__main__":
    main()