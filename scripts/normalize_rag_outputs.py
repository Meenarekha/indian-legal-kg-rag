import json
import re
from pathlib import Path
from datetime import datetime


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

GENERATION_DIR = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "generation_outputs"
)

CONTEXT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "contexts"
)


# ============================================================
# JSON HELPERS
# ============================================================

def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def save_json(path, data):

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False
        )


# ============================================================
# PARSE TIMESTAMP FROM FILENAME
# ============================================================

def get_timestamp(path):

    """
    Extract YYYYMMDD_HHMMSS from filename.
    """

    match = re.search(
        r"(\d{8}_\d{6})",
        path.name
    )

    if not match:
        return None

    return match.group(1)


# ============================================================
# GET FILE TIME AS NUMBER
# ============================================================

def timestamp_to_number(
    timestamp
):

    if not timestamp:
        return None

    return int(
        timestamp.replace(
            "_",
            ""
        )
    )


# ============================================================
# FIND RAG ANSWER FILES
# ============================================================

def find_rag_answer_files(
    query_id
):

    files = []

    pattern = (
        f"rag_answer_{query_id}_*.json"
    )

    # Case-insensitive handling
    patterns = [
        f"rag_answer_{query_id}_*.json",
        f"rag_answer_{query_id.lower()}_*.json"
    ]

    for pattern in patterns:

        for path in GENERATION_DIR.glob(
            pattern
        ):

            try:

                data = load_json(
                    path
                )

            except Exception:

                continue

            if not data.get(
                "answer"
            ):

                continue

            files.append(
                (
                    path.stat().st_mtime,
                    path,
                    data
                )
            )

    # Remove duplicates
    unique = {}

    for item in files:

        unique[
            str(item[1])
        ] = item

    files = list(
        unique.values()
    )

    files.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return files


# ============================================================
# FIND CONTEXT FILES
# ============================================================

def find_context_files(
    query_id
):

    pattern = (
        f"rag_context_{query_id}_*.json"
    )

    files = []

    for path in CONTEXT_DIR.glob(
        pattern
    ):

        files.append(
            path
        )

    return files


# ============================================================
# MATCH CONTEXT TO ANSWER
# ============================================================

def find_best_context(
    answer_path,
    context_files
):

    if not context_files:
        return None

    answer_timestamp = (
        get_timestamp(
            answer_path
        )
    )

    answer_number = (
        timestamp_to_number(
            answer_timestamp
        )
    )

    # --------------------------------------------------------
    # If timestamp unavailable, use newest context
    # --------------------------------------------------------

    if answer_number is None:

        context_files.sort(
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        return context_files[0]

    candidates = []

    for context_path in context_files:

        context_timestamp = (
            get_timestamp(
                context_path
            )
        )

        context_number = (
            timestamp_to_number(
                context_timestamp
            )
        )

        if context_number is None:
            continue

        difference = abs(
            answer_number
            -
            context_number
        )

        # Context should normally be generated
        # immediately before the answer.
        candidates.append(
            (
                difference,
                context_number,
                context_path
            )
        )

    if not candidates:

        return None

    # --------------------------------------------------------
    # Prefer context immediately BEFORE answer.
    # --------------------------------------------------------

    before = [
        item
        for item in candidates
        if item[1] <= answer_number
    ]

    if before:

        before.sort(
            key=lambda x: (
                answer_number - x[1]
            )
        )

        return before[0][2]

    # Otherwise choose closest
    candidates.sort(
        key=lambda x: x[0]
    )

    return candidates[0][2]


# ============================================================
# EXTRACT CONTEXT DATA
# ============================================================

def extract_context(
    context_path
):

    if not context_path:
        return None

    try:

        data = load_json(
            context_path
        )

    except Exception as error:

        print(
            f"Could not read context: "
            f"{context_path.name}"
        )

        print(
            f"Reason: {error}"
        )

        return None

    evidence_records = data.get(
        "evidence_records",
        []
    )

    if not isinstance(
        evidence_records,
        list
    ):

        evidence_records = []

    # --------------------------------------------------------
    # Context character count
    # --------------------------------------------------------

    context_characters = data.get(
        "total_context_characters"
    )

    if context_characters is None:

        context_characters = data.get(
            "context_characters"
        )

    # --------------------------------------------------------
    # Cases considered
    # --------------------------------------------------------

    cases = data.get(
        "cases_considered"
    )

    if isinstance(
        cases,
        list
    ):

        cases_considered = len(
            cases
        )

    elif isinstance(
        cases,
        int
    ):

        cases_considered = cases

    else:

        unique_cases = set()

        for record in evidence_records:

            case = (
                record.get(
                    "case_name"
                )
                or
                record.get(
                    "case_id"
                )
            )

            if case:
                unique_cases.add(
                    str(case)
                )

        cases_considered = len(
            unique_cases
        )

    # --------------------------------------------------------
    # Truncation
    # --------------------------------------------------------

    truncated = data.get(
        "truncated"
    )

    if truncated is None:

        truncated = data.get(
            "context_truncated",
            False
        )

    return {

        "evidence_records":
            evidence_records,

        "cases_considered":
            cases_considered,

        "context_characters":
            context_characters,

        "context_truncated":
            truncated,

        "configuration":
            data.get(
                "configuration",
                {}
            )
    }


# ============================================================
# BUILD STANDARDIZED RAG FILE
# ============================================================

def normalize_query(
    query_id
):

    answer_files = (
        find_rag_answer_files(
            query_id
        )
    )

    if not answer_files:

        print(
            f"\n{query_id}: "
            "NO ANSWER FILE"
        )

        return None

    # Latest generated answer
    _, answer_path, answer_data = (
        answer_files[0]
    )

    print(
        f"\n{query_id}"
    )

    print(
        f"Answer: "
        f"{answer_path.name}"
    )

    # --------------------------------------------------------
    # Context candidates
    # --------------------------------------------------------

    context_files = (
        find_context_files(
            query_id
        )
    )

    context_path = (
        find_best_context(
            answer_path,
            context_files
        )
    )

    if context_path:

        print(
            f"Context: "
            f"{context_path.name}"
        )

    else:

        print(
            "Context: NOT FOUND"
        )

    # --------------------------------------------------------
    # Extract context
    # --------------------------------------------------------

    context_data = (
        extract_context(
            context_path
        )
    )

    if context_data is None:

        context_data = {

            "evidence_records": [],

            "cases_considered":
                answer_data.get(
                    "cases_considered",
                    0
                ),

            "context_characters":
                answer_data.get(
                    "context_characters",
                    0
                ),

            "context_truncated":
                answer_data.get(
                    "context_truncated",
                    False
                ),

            "configuration":
                answer_data.get(
                    "retrieval_configuration",
                    {}
                )
        }

    evidence = (
        context_data[
            "evidence_records"
        ]
    )

    cases = (
        context_data[
            "cases_considered"
        ]
    )

    characters = (
        context_data[
            "context_characters"
        ]
    )

    truncated = (
        context_data[
            "context_truncated"
        ]
    )

    configuration = (
        context_data[
            "configuration"
        ]
    )

    # --------------------------------------------------------
    # Fallback configuration
    # --------------------------------------------------------

    if not configuration:

        configuration = (
            answer_data.get(
                "retrieval_configuration",
                {
                    "retrieval_pipeline":
                        "BM25 + Dense FAISS + RRF + "
                        "case aggregation + relevance filtering + "
                        "authority-aware reranking",

                    "max_cases":
                        5,

                    "supporting_chunks_per_case":
                        3,

                    "max_context_characters":
                        30000
                }
            )
        )

    # --------------------------------------------------------
    # Standard structure
    # --------------------------------------------------------

    normalized = {

        "query_id":
            query_id,

        "question":
            answer_data.get(
                "question",
                ""
            ),

        "generation_type":
            "RAG",

        "provider":
            answer_data.get(
                "provider",
                "provider_manager"
            ),

        "primary_provider":
            answer_data.get(
                "primary_provider",
                "gemini"
            ),

        "fallback_enabled":
            answer_data.get(
                "fallback_enabled",
                True
            ),

        "retrieval_used":
            True,

        "reference_answer_used":
            False,

        "retrieval_configuration":
            configuration,

        "cases_considered":
            cases,

        "evidence_records":
            evidence,

        "context_characters":
            characters,

        "context_truncated":
            truncated,

        "answer":
            answer_data.get(
                "answer",
                ""
            ),

        "generated_at":
            answer_data.get(
                "generated_at",
                ""
            ),

        "normalized_at":
            datetime.now().isoformat(),

        "original_answer_file":
            answer_path.name,

        "context_source_file":
            (
                context_path.name
                if context_path
                else None
            )
    }

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = (
        GENERATION_DIR
        /
        f"rag_answer_{query_id}.json"
    )

    save_json(
        output_path,
        normalized
    )

    print(
        f"Cases considered: "
        f"{cases}"
    )

    print(
        f"Evidence records: "
        f"{len(evidence)}"
    )

    print(
        f"Context characters: "
        f"{characters}"
    )

    print(
        f"Context truncated: "
        f"{truncated}"
    )

    print(
        f"Saved: "
        f"{output_path.name}"
    )

    return normalized


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("REBUILDING ALL 30 RAG OUTPUTS WITH ACTUAL EVIDENCE")
    print("=" * 80)

    results = []

    for number in range(
        1,
        31
    ):

        query_id = (
            f"Q{number:03d}"
        )

        result = normalize_query(
            query_id
        )

        if result:

            results.append(
                result
            )

    # ========================================================
    # VALIDATION
    # ========================================================

    print("\n")
    print("=" * 80)
    print("FINAL EVIDENCE VALIDATION")
    print("=" * 80)

    zero_evidence = []

    for number in range(
        1,
        31
    ):

        query_id = (
            f"Q{number:03d}"
        )

        path = (
            GENERATION_DIR
            /
            f"rag_answer_{query_id}.json"
        )

        if not path.exists():

            print(
                f"{query_id}: MISSING"
            )

            continue

        data = load_json(
            path
        )

        evidence = data.get(
            "evidence_records",
            []
        )

        cases = data.get(
            "cases_considered"
        )

        characters = data.get(
            "context_characters"
        )

        print(
            f"{query_id}: "
            f"{len(evidence)} evidence | "
            f"{cases} cases | "
            f"{characters} chars"
        )

        if not evidence:

            zero_evidence.append(
                query_id
            )

    # ========================================================
    # RESULT
    # ========================================================

    print("\n")

    if zero_evidence:

        print(
            "WARNING: These queries still have zero evidence:"
        )

        print(
            ", ".join(
                zero_evidence
            )
        )

        print(
            "\nDO NOT RUN GROUNDEDNESS YET."
        )

    else:

        print(
            "SUCCESS!"
        )

        print(
            "All 30 RAG files contain retrieved evidence."
        )

        print(
            "Groundedness evaluation can now be rerun."
        )

    print("\n")
    print(
        f"Normalized: {len(results)}/30"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()