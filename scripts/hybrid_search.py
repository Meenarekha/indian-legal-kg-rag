import json
import pickle
import re
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

BM25_INDEX_PATH = (
    BASE_DIR
    / "data"
    / "indexes"
    / "bm25"
    / "bm25_index.pkl"
)

BM25_METADATA_PATH = (
    BASE_DIR
    / "data"
    / "indexes"
    / "bm25"
    / "bm25_metadata.json"
)

FAISS_INDEX_PATH = (
    BASE_DIR
    / "data"
    / "indexes"
    / "faiss"
    / "faiss.index"
)

FAISS_METADATA_PATH = (
    BASE_DIR
    / "data"
    / "indexes"
    / "faiss"
    / "faiss_metadata.json"
)

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


# Number of candidates retrieved from each system
BM25_TOP_K = 20
FAISS_TOP_K = 20

# RRF constant
RRF_K = 60

# Final hybrid results
FINAL_TOP_K = 10


# ============================================================
# BM25 TOKENIZER
# ============================================================

def tokenize(text):
    """
    Tokenizer used for BM25 retrieval.

    Extracts:
    - alphabetic tokens
    - numeric/legal-number tokens
    """

    text = text.lower()

    tokens = re.findall(
        r"[a-z]+|\d+(?:\.\d+)*",
        text
    )

    return tokens


# ============================================================
# LOAD BM25 INDEX
# ============================================================

def load_bm25():

    print("\nLoading BM25 index...")

    with open(BM25_INDEX_PATH, "rb") as f:
        bm25 = pickle.load(f)

    with open(BM25_METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print(
        f"BM25 documents loaded: {len(metadata)}"
    )

    return bm25, metadata


# ============================================================
# LOAD FAISS INDEX
# ============================================================

def load_faiss():

    print("\nLoading FAISS index...")

    index = faiss.read_index(
        str(FAISS_INDEX_PATH)
    )

    with open(
        FAISS_METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        metadata = json.load(f)

    print(
        f"FAISS vectors loaded: {index.ntotal}"
    )

    return index, metadata


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

def load_embedding_model():

    print("\nLoading embedding model:")
    print(EMBEDDING_MODEL_NAME)

    model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )

    print(
        "Embedding dimension:",
        model.get_embedding_dimension()
    )

    return model


# ============================================================
# CREATE STANDARD RESULT
# ============================================================

def create_result(
    metadata,
    idx,
    rank,
    score_name,
    score
):
    """
    Convert metadata from either BM25 or FAISS
    into one common result structure.

    BM25 and FAISS metadata may have slightly
    different field names, so this function
    handles both safely.
    """

    item = metadata[idx]

    return {
        "rank": rank,
        "score": float(score),

        "chunk_id": item.get("chunk_id"),
        "case_id": item.get("case_id"),
        "court": item.get("court"),
        "source_file": item.get("source_file"),

        "start_page": item.get(
            "start_page",
            item.get("pages", [None])[0]
            if item.get("pages")
            else None
        ),

        "end_page": item.get(
            "end_page",
            item.get("pages", [None])[-1]
            if item.get("pages")
            else None
        ),

        "pages": item.get("pages", []),

        "paragraph_start": item.get(
            "paragraph_start"
        ),

        "paragraph_end": item.get(
            "paragraph_end"
        ),

        "text": item.get("text", "")
    }


# ============================================================
# BM25 SEARCH
# ============================================================

def bm25_search(
    query,
    bm25,
    metadata,
    top_k=20
):

    query_tokens = tokenize(query)

    scores = bm25.get_scores(
        query_tokens
    )

    ranked_indices = np.argsort(
        scores
    )[::-1][:top_k]

    results = []

    for rank, idx in enumerate(
        ranked_indices,
        start=1
    ):

        idx = int(idx)

        result = create_result(
            metadata,
            idx,
            rank,
            "bm25",
            scores[idx]
        )

        results.append(result)

    return results


# ============================================================
# FAISS SEARCH
# ============================================================

def faiss_search(
    query,
    index,
    metadata,
    model,
    top_k=20
):

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    query_embedding = (
        query_embedding.astype("float32")
    )

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for rank, (score, idx) in enumerate(
        zip(scores[0], indices[0]),
        start=1
    ):

        idx = int(idx)

        if idx < 0:
            continue

        result = create_result(
            metadata,
            idx,
            rank,
            "faiss",
            score
        )

        results.append(result)

    return results


# ============================================================
# RECIPROCAL RANK FUSION
# ============================================================

def reciprocal_rank_fusion(
    bm25_results,
    faiss_results,
    rrf_k=60
):

    fused = {}

    # --------------------------------------------------------
    # BM25 results
    # --------------------------------------------------------

    for result in bm25_results:

        chunk_id = result["chunk_id"]

        rank = result["rank"]

        if chunk_id not in fused:

            fused[chunk_id] = {
                "chunk_id":
                    result["chunk_id"],

                "case_id":
                    result["case_id"],

                "court":
                    result["court"],

                "source_file":
                    result["source_file"],

                "start_page":
                    result["start_page"],

                "end_page":
                    result["end_page"],

                "pages":
                    result["pages"],

                "paragraph_start":
                    result["paragraph_start"],

                "paragraph_end":
                    result["paragraph_end"],

                "text":
                    result["text"],

                "rrf_score": 0.0,

                "bm25_rank": None,
                "bm25_score": None,

                "faiss_rank": None,
                "faiss_score": None
            }

        fused[chunk_id]["rrf_score"] += (
            1.0 / (rrf_k + rank)
        )

        fused[chunk_id]["bm25_rank"] = rank

        fused[chunk_id]["bm25_score"] = (
            result["score"]
        )

    # --------------------------------------------------------
    # FAISS results
    # --------------------------------------------------------

    for result in faiss_results:

        chunk_id = result["chunk_id"]

        rank = result["rank"]

        if chunk_id not in fused:

            fused[chunk_id] = {
                "chunk_id":
                    result["chunk_id"],

                "case_id":
                    result["case_id"],

                "court":
                    result["court"],

                "source_file":
                    result["source_file"],

                "start_page":
                    result["start_page"],

                "end_page":
                    result["end_page"],

                "pages":
                    result["pages"],

                "paragraph_start":
                    result["paragraph_start"],

                "paragraph_end":
                    result["paragraph_end"],

                "text":
                    result["text"],

                "rrf_score": 0.0,

                "bm25_rank": None,
                "bm25_score": None,

                "faiss_rank": None,
                "faiss_score": None
            }

        fused[chunk_id]["rrf_score"] += (
            1.0 / (rrf_k + rank)
        )

        fused[chunk_id]["faiss_rank"] = rank

        fused[chunk_id]["faiss_score"] = (
            result["score"]
        )

    # --------------------------------------------------------
    # Sort by RRF score
    # --------------------------------------------------------

    ranked_results = sorted(
        fused.values(),
        key=lambda x: x["rrf_score"],
        reverse=True
    )

    return ranked_results


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(
    query,
    results
):

    print("\n")
    print("=" * 80)
    print("HYBRID BM25 + FAISS RESULTS")
    print("=" * 80)

    print(f"\nQuery: {query}")
    print(f"Results: {len(results)}")

    for rank, result in enumerate(
        results,
        start=1
    ):

        print("\n" + "-" * 80)

        print(
            f"Hybrid Rank    : {rank}"
        )

        print(
            f"RRF Score      : "
            f"{result['rrf_score']:.6f}"
        )

        print(
            f"BM25 Rank      : "
            f"{result['bm25_rank']}"
        )

        if result["bm25_score"] is not None:
            print(
                f"BM25 Score     : "
                f"{result['bm25_score']:.4f}"
            )
        else:
            print(
                "BM25 Score     : None"
            )

        print(
            f"FAISS Rank     : "
            f"{result['faiss_rank']}"
        )

        if result["faiss_score"] is not None:
            print(
                f"FAISS Score    : "
                f"{result['faiss_score']:.4f}"
            )
        else:
            print(
                "FAISS Score    : None"
            )

        print(
            f"Chunk ID       : "
            f"{result['chunk_id']}"
        )

        print(
            f"Case ID        : "
            f"{result['case_id']}"
        )

        print(
            f"Court          : "
            f"{result['court']}"
        )

        print(
            f"Pages          : "
            f"{result['pages']}"
        )

        print(
            f"Paragraph      : "
            f"{result['paragraph_start']} - "
            f"{result['paragraph_end']}"
        )

        print("\nText:")

        print(
            result["text"][:1200]
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("HYBRID LEGAL RETRIEVAL")
    print("BM25 + FAISS + Reciprocal Rank Fusion")
    print("=" * 80)

    # --------------------------------------------------------
    # Load BM25
    # --------------------------------------------------------

    bm25, bm25_metadata = load_bm25()

    # --------------------------------------------------------
    # Load FAISS
    # --------------------------------------------------------

    faiss_index, faiss_metadata = load_faiss()

    # --------------------------------------------------------
    # Verify metadata alignment
    # --------------------------------------------------------

    print(
        "\nBM25 metadata records:",
        len(bm25_metadata)
    )

    print(
        "FAISS metadata records:",
        len(faiss_metadata)
    )

    print(
        "FAISS index vectors:",
        faiss_index.ntotal
    )

    # --------------------------------------------------------
    # Load embedding model
    # --------------------------------------------------------

    embedding_model = (
        load_embedding_model()
    )

    print(
        "\nHybrid retrieval system ready."
    )

    print(
        "\nEnter a legal query."
    )

    print(
        "Type 'exit' to stop."
    )

    # --------------------------------------------------------
    # Interactive search
    # --------------------------------------------------------

    while True:

        query = input(
            "\nQuery: "
        ).strip()

        if query.lower() == "exit":

            print(
                "\nExiting hybrid search."
            )

            break

        if not query:

            print(
                "Please enter a legal query."
            )

            continue

        # ----------------------------------------------------
        # BM25 retrieval
        # ----------------------------------------------------

        bm25_results = bm25_search(
            query,
            bm25,
            bm25_metadata,
            BM25_TOP_K
        )

        # ----------------------------------------------------
        # FAISS retrieval
        # ----------------------------------------------------

        faiss_results = faiss_search(
            query,
            faiss_index,
            faiss_metadata,
            embedding_model,
            FAISS_TOP_K
        )

        # ----------------------------------------------------
        # RRF fusion
        # ----------------------------------------------------

        hybrid_results = (
            reciprocal_rank_fusion(
                bm25_results,
                faiss_results,
                RRF_K
            )
        )

        # ----------------------------------------------------
        # Final top K
        # ----------------------------------------------------

        hybrid_results = (
            hybrid_results[:FINAL_TOP_K]
        )

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        display_results(
            query,
            hybrid_results
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()