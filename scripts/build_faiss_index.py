import json
import numpy as np
import faiss

from pathlib import Path
from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

CHUNKS_DIR = Path("data/processed/chunks")

INDEX_DIR = Path("data/indexes/faiss")

INDEX_DIR.mkdir(
    parents=True,
    exist_ok=True
)


INDEX_PATH = INDEX_DIR / "faiss.index"
METADATA_PATH = INDEX_DIR / "faiss_metadata.json"
CONFIG_PATH = INDEX_DIR / "embedding_config.json"


# ============================================================
# EMBEDDING MODEL
# ============================================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Number of chunks embedded at once.
BATCH_SIZE = 32


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
            ).strip()

            if not text:
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
                "character_count": chunk.get(
                    "character_count"
                ),
                "text": text
            })

    return all_chunks


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

def load_model():

    print()
    print("Loading embedding model:")
    print(MODEL_NAME)

    model = SentenceTransformer(
        MODEL_NAME
    )

    print(
        f"Embedding dimension: "
        f"{model.get_sentence_embedding_dimension()}"
    )

    return model


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

def create_embeddings(
    model,
    chunks
):

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print()
    print(
        "Creating dense embeddings..."
    )

    print(
        f"Total texts: {len(texts)}"
    )

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    return embeddings


# ============================================================
# BUILD FAISS INDEX
# ============================================================

def build_faiss_index(
    embeddings
):

    dimension = embeddings.shape[1]

    print()
    print(
        "Building FAISS index..."
    )

    print(
        f"Vector dimension: {dimension}"
    )

    print(
        "Index type: IndexFlatIP"
    )

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    print(
        f"Vectors added: {index.ntotal}"
    )

    return index


# ============================================================
# SAVE INDEX
# ============================================================

def save_index(
    index,
    chunks,
    model
):

    print()
    print(
        "Saving FAISS index..."
    )

    faiss.write_index(
        index,
        str(INDEX_PATH)
    )

    # --------------------------------------------------------
    # Save metadata in exactly the same order as vectors.
    #
    # Vector position 0 → metadata[0]
    # Vector position 1 → metadata[1]
    # etc.
    # --------------------------------------------------------

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            chunks,
            f,
            ensure_ascii=False,
            indent=2
        )

    # --------------------------------------------------------
    # Save configuration
    # --------------------------------------------------------

    config = {
        "model_name": MODEL_NAME,
        "embedding_dimension":
            model.get_sentence_embedding_dimension(),
        "index_type": "IndexFlatIP",
        "normalized_embeddings": True,
        "similarity": "cosine_similarity",
        "batch_size": BATCH_SIZE,
        "number_of_vectors": len(chunks)
    }

    with open(
        CONFIG_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            config,
            f,
            indent=2
        )

    print()
    print(
        f"FAISS index saved to:"
    )

    print(
        INDEX_PATH.resolve()
    )

    print()
    print(
        "FAISS metadata saved to:"
    )

    print(
        METADATA_PATH.resolve()
    )

    print()
    print(
        "Embedding configuration saved to:"
    )

    print(
        CONFIG_PATH.resolve()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("DENSE EMBEDDING + FAISS INDEX BUILDING")
    print("=" * 80)

    # --------------------------------------------------------
    # Load chunks
    # --------------------------------------------------------

    chunks = load_chunks()

    print()
    print(
        f"Total chunks loaded: {len(chunks)}"
    )

    if not chunks:

        raise ValueError(
            "No chunks found."
        )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Create embeddings
    # --------------------------------------------------------

    embeddings = create_embeddings(
        model,
        chunks
    )

    # --------------------------------------------------------
    # Validate embedding matrix
    # --------------------------------------------------------

    print()
    print(
        "Embedding matrix shape:"
    )

    print(
        embeddings.shape
    )

    if embeddings.shape[0] != len(chunks):

        raise ValueError(
            "Number of embeddings does not "
            "match number of chunks."
        )

    # --------------------------------------------------------
    # Build FAISS
    # --------------------------------------------------------

    index = build_faiss_index(
        embeddings
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_index(
        index,
        chunks,
        model
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("FAISS INDEX COMPLETE")
    print("=" * 80)

    print(
        f"Chunks embedded:       {len(chunks)}"
    )

    print(
        f"Vectors indexed:       {index.ntotal}"
    )

    print(
        f"Vector dimension:      "
        f"{embeddings.shape[1]}"
    )

    print(
        "Normalization:          Enabled"
    )

    print(
        "Similarity:             Cosine similarity"
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