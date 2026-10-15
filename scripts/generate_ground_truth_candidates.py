import json
import pickle
import re
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

EVALUATION_FILE = Path(
    "data/evaluation/retrieval_eval.json"
)

BM25_INDEX_FILE = Path(
    "data/indexes/bm25/bm25_index.pkl"
)

BM25_METADATA_FILE = Path(
    "data/indexes/bm25/bm25_metadata.json"
)

FAISS_INDEX_FILE = Path(
    "data/indexes/faiss/faiss.index"
)

FAISS_METADATA_FILE = Path(
    "data/indexes/faiss/faiss_metadata.json"
)

OUTPUT_FILE = Path(
    "data/evaluation/retrieval_candidates.json"
)

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

BM25_TOP_K = 30
FAISS_TOP_K = 30

MAX_CANDIDATE_CASES = 20
SUPPORTING_CHUNKS_PER_CASE = 3


# ============================================================
# TOKENIZER
# ============================================================

def tokenize(text):
    """
    Tokenizer consistent with the BM25 index.
    """

    text = text.lower()

    return re.findall(
        r"[a-z]+|\d+(?:\.\d+)*",
        text
    )


# ============================================================
# LOAD BM25
# ============================================================

def load_bm25():

    print("Loading BM25...")

    with open(
        BM25_INDEX_FILE,
        "rb"
    ) as file:

        bm25 = pickle.load(file)

    with open(
        BM25_METADATA_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        metadata = json.load(file)

    print(
        f"BM25 documents: {len(metadata)}"
    )

    return bm25, metadata


# ============================================================
# LOAD FAISS
# ============================================================

def load_faiss():

    print("Loading FAISS...")

    index = faiss.read_index(
        str(FAISS_INDEX_FILE)
    )

    with open(
        FAISS_METADATA_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        metadata = json.load(file)

    print(
        f"FAISS vectors: {index.ntotal}"
    )

    print(
        f"FAISS metadata: {len(metadata)}"
    )

    return index, metadata


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

def load_embedding_model():

    print("Loading embedding model...")

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    print(
        "Embedding dimension:",
        model.get_embedding_dimension()
    )

    return model


# ============================================================
# BM25 SEARCH
# ============================================================

def bm25_search(
    query,
    bm25,
    metadata,
    top_k
):

    tokens = tokenize(query)

    scores = bm25.get_scores(
        tokens
    )

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:top_k]

    results = []

    for rank, index in enumerate(
        ranked_indices,
        start=1
    ):

        record = metadata[index]

        results.append({
            "rank": rank,
            "score": float(
                scores[index]
            ),
            "case_id": record.get(
                "case_id",
                ""
            ),
            "court": record.get(
                "court",
                ""
            ),
            "source_file": record.get(
                "source_file",
                ""
            ),
            "start_page": record.get(
                "start_page"
            ),
            "end_page": record.get(
                "end_page"
            ),
            "pages": record.get(
                "pages",
                []
            ),
            "paragraph_start": record.get(
                "paragraph_start"
            ),
            "paragraph_end": record.get(
                "paragraph_end"
            ),
            "text": record.get(
                "text",
                ""
            )
        })

    return results


# ============================================================
# FAISS SEARCH
# ============================================================

def faiss_search(
    query,
    model,
    index,
    metadata,
    top_k
):

    embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")

    scores, indices = index.search(
        embedding,
        top_k
    )

    results = []

    for rank, (score, index_id) in enumerate(
        zip(scores[0], indices[0]),
        start=1
    ):

        if index_id < 0:
            continue

        record = metadata[index_id]

        results.append({
            "rank": rank,
            "score": float(score),
            "case_id": record.get(
                "case_id",
                ""
            ),
            "court": record.get(
                "court",
                ""
            ),
            "source_file": record.get(
                "source_file",
                ""
            ),
            "start_page": record.get(
                "start_page"
            ),
            "end_page": record.get(
                "end_page"
            ),
            "pages": record.get(
                "pages",
                []
            ),
            "paragraph_start": record.get(
                "paragraph_start"
            ),
            "paragraph_end": record.get(
                "paragraph_end"
            ),
            "text": record.get(
                "text",
                ""
            )
        })

    return results


# ============================================================
# GROUP CANDIDATES BY CASE
# ============================================================

def build_candidate_cases(
    bm25_results,
    faiss_results
):

    case_map = {}

    # --------------------------------------------------------
    # Add BM25 results
    # --------------------------------------------------------

    for result in bm25_results:

        case_id = result["case_id"]

        if not case_id:
            continue

        if case_id not in case_map:

            case_map[case_id] = {
                "case_id": case_id,
                "court": result["court"],
                "source_file": result[
                    "source_file"
                ],
                "bm25_best_rank": None,
                "faiss_best_rank": None,
                "supporting_chunks": []
            }

        case = case_map[case_id]

        if (
            case["bm25_best_rank"] is None
            or result["rank"]
            < case["bm25_best_rank"]
        ):

            case["bm25_best_rank"] = (
                result["rank"]
            )

        case["supporting_chunks"].append({
            "retriever": "BM25",
            "rank": result["rank"],
            "score": result["score"],
            "pages": result["pages"],
            "paragraph_start": result[
                "paragraph_start"
            ],
            "paragraph_end": result[
                "paragraph_end"
            ],
            "text": result["text"]
        })


    # --------------------------------------------------------
    # Add FAISS results
    # --------------------------------------------------------

    for result in faiss_results:

        case_id = result["case_id"]

        if not case_id:
            continue

        if case_id not in case_map:

            case_map[case_id] = {
                "case_id": case_id,
                "court": result["court"],
                "source_file": result[
                    "source_file"
                ],
                "bm25_best_rank": None,
                "faiss_best_rank": None,
                "supporting_chunks": []
            }

        case = case_map[case_id]

        if (
            case["faiss_best_rank"] is None
            or result["rank"]
            < case["faiss_best_rank"]
        ):

            case["faiss_best_rank"] = (
                result["rank"]
            )

        case["supporting_chunks"].append({
            "retriever": "FAISS",
            "rank": result["rank"],
            "score": result["score"],
            "pages": result["pages"],
            "paragraph_start": result[
                "paragraph_start"
            ],
            "paragraph_end": result[
                "paragraph_end"
            ],
            "text": result["text"]
        })


    # --------------------------------------------------------
    # Create a candidate ordering
    # --------------------------------------------------------
    #
    # A case appearing strongly in both retrieval systems
    # should appear early in the annotation list.
    #
    # This is ONLY candidate ordering.
    # It is NOT the ground-truth relevance score.
    # --------------------------------------------------------

    cases = list(
        case_map.values()
    )

    def candidate_priority(case):

        bm25_rank = (
            case["bm25_best_rank"]
            if case["bm25_best_rank"] is not None
            else 999
        )

        faiss_rank = (
            case["faiss_best_rank"]
            if case["faiss_best_rank"] is not None
            else 999
        )

        return (
            min(
                bm25_rank,
                faiss_rank
            ),
            bm25_rank + faiss_rank
        )

    cases.sort(
        key=candidate_priority
    )


    # --------------------------------------------------------
    # Keep strongest supporting chunks
    # --------------------------------------------------------

    for case in cases:

        chunks = case[
            "supporting_chunks"
        ]

        chunks.sort(
            key=lambda x: (
                x["rank"],
                -x["score"]
            )
        )

        case[
            "supporting_chunks"
        ] = chunks[
            :SUPPORTING_CHUNKS_PER_CASE
        ]


    return cases[
        :MAX_CANDIDATE_CASES
    ]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print(
        "GROUND-TRUTH CANDIDATE GENERATION"
    )
    print("=" * 80)

    # --------------------------------------------------------
    # Load evaluation questions
    # --------------------------------------------------------

    print(
        "\nLoading evaluation questions..."
    )

    with open(
        EVALUATION_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        evaluation_data = json.load(
            file
        )

    queries = evaluation_data[
        "queries"
    ]

    print(
        f"Queries: {len(queries)}"
    )

    # --------------------------------------------------------
    # Load retrieval systems
    # --------------------------------------------------------

    bm25, bm25_metadata = load_bm25()

    faiss_index, faiss_metadata = (
        load_faiss()
    )

    embedding_model = (
        load_embedding_model()
    )

    # --------------------------------------------------------
    # Validate index sizes
    # --------------------------------------------------------

    if len(bm25_metadata) != (
        len(bm25.doc_len)
    ):

        print(
            "\nWARNING: BM25 metadata/document "
            "count mismatch."
        )

    if faiss_index.ntotal != (
        len(faiss_metadata)
    ):

        raise ValueError(
            "FAISS index and metadata counts "
            "do not match."
        )

    # --------------------------------------------------------
    # Process queries
    # --------------------------------------------------------

    output_queries = []

    for query_number, query_record in enumerate(
        queries,
        start=1
    ):

        query_id = query_record[
            "query_id"
        ]

        query = query_record[
            "query"
        ]

        print("\n" + "-" * 80)

        print(
            f"[{query_number:02d}/{len(queries):02d}] "
            f"{query_id}"
        )

        print(
            f"Query: {query}"
        )

        # ----------------------------------------------------
        # BM25
        # ----------------------------------------------------

        bm25_results = bm25_search(
            query,
            bm25,
            bm25_metadata,
            BM25_TOP_K
        )

        # ----------------------------------------------------
        # FAISS
        # ----------------------------------------------------

        faiss_results = faiss_search(
            query,
            embedding_model,
            faiss_index,
            faiss_metadata,
            FAISS_TOP_K
        )

        # ----------------------------------------------------
        # Case aggregation
        # ----------------------------------------------------

        candidate_cases = (
            build_candidate_cases(
                bm25_results,
                faiss_results
            )
        )

        print(
            f"BM25 chunks: {len(bm25_results)}"
        )

        print(
            f"FAISS chunks: {len(faiss_results)}"
        )

        print(
            f"Unique candidate cases: "
            f"{len(candidate_cases)}"
        )

        # ----------------------------------------------------
        # Display candidate cases
        # ----------------------------------------------------

        for rank, case in enumerate(
            candidate_cases,
            start=1
        ):

            print(
                f"\n{rank:02d}. "
                f"{case['case_id']}"
            )

            print(
                f"    Court: {case['court']}"
            )

            print(
                f"    BM25 best rank: "
                f"{case['bm25_best_rank']}"
            )

            print(
                f"    FAISS best rank: "
                f"{case['faiss_best_rank']}"
            )

            for chunk in case[
                "supporting_chunks"
            ]:

                text = chunk["text"]

                preview = (
                    text[:350]
                    .replace("\n", " ")
                )

                print(
                    f"    [{chunk['retriever']}] "
                    f"rank={chunk['rank']}"
                )

                print(
                    f"      Pages: "
                    f"{chunk['pages']}"
                )

                print(
                    f"      Paragraphs: "
                    f"{chunk['paragraph_start']}"
                    f"-"
                    f"{chunk['paragraph_end']}"
                )

                print(
                    f"      Evidence: "
                    f"{preview}"
                )


        output_queries.append({
            "query_id": query_id,
            "domain": query_record[
                "domain"
            ],
            "query": query,

            "candidate_cases": candidate_cases,

            "ground_truth": {
                "relevant_cases": [],
                "annotation_status": "PENDING"
            }
        })


    # --------------------------------------------------------
    # Save output
    # --------------------------------------------------------

    output_data = {

        "dataset_name":
            "Indian Legal RAG Ground Truth Candidate Set",

        "version":
            "1.0",

        "purpose":
            (
                "Candidate cases generated for independent "
                "ground-truth annotation."
            ),

        "candidate_generation_policy":
            (
                "Candidates are generated using BM25 and "
                "dense semantic retrieval. These candidates "
                "are only used to facilitate annotation and "
                "must not automatically be treated as "
                "ground truth."
            ),

        "annotation_status":
            "PENDING",

        "queries":
            output_queries
    }


    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output_data,
            file,
            indent=4,
            ensure_ascii=False
        )


    print("\n" + "=" * 80)

    print(
        "CANDIDATE GENERATION COMPLETED"
    )

    print("=" * 80)

    print(
        f"\nOutput:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        "\nGround truth remains PENDING."
    )

    print(
        "\nThe candidate list must be independently "
        "validated using the underlying judgment evidence."
    )


if __name__ == "__main__":
    main()