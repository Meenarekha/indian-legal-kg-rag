import json
import re
from pathlib import Path
from difflib import SequenceMatcher


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
# JSON
# ============================================================

def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def save_json(path, data):

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False
        )


# ============================================================
# NORMALIZE TEXT
# ============================================================

def normalize_text(text):

    if not text:
        return ""

    text = str(text).lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# SIMILARITY
# ============================================================

def similarity(
    text1,
    text2
):

    a = normalize_text(
        text1
    )

    b = normalize_text(
        text2
    )

    if not a or not b:
        return 0.0

    return SequenceMatcher(
        None,
        a,
        b
    ).ratio()


# ============================================================
# FIND OLD CONTEXT FILES
# ============================================================

def get_old_contexts():

    contexts = []

    for path in CONTEXT_DIR.glob(
        "context_*.json"
    ):

        try:

            data = load_json(
                path
            )

        except Exception:

            continue

        query = data.get(
            "query"
        )

        evidence = data.get(
            "evidence_records",
            []
        )

        if not query:
            continue

        if not isinstance(
            evidence,
            list
        ):
            continue

        if not evidence:
            continue

        contexts.append(
            {
                "path": path,
                "data": data,
                "query": query,
                "evidence": evidence
            }
        )

    return contexts


# ============================================================
# FIND Q FILE
# ============================================================

def get_standardized_file(
    query_id
):

    path = (
        GENERATION_DIR
        /
        f"rag_answer_{query_id}.json"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Missing: {path}"
        )

    return path


# ============================================================
# EXTRACT CONTEXT INFORMATION
# ============================================================

def extract_context_info(
    context_data
):

    evidence = context_data.get(
        "evidence_records",
        []
    )

    if not isinstance(
        evidence,
        list
    ):

        evidence = []

    # --------------------------------------------------------
    # Cases
    # --------------------------------------------------------

    cases = context_data.get(
        "cases_considered"
    )

    if isinstance(
        cases,
        list
    ):

        cases_count = len(
            cases
        )

    elif isinstance(
        cases,
        int
    ):

        cases_count = cases

    else:

        unique_cases = set()

        for record in evidence:

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

        cases_count = len(
            unique_cases
        )

    # --------------------------------------------------------
    # Characters
    # --------------------------------------------------------

    characters = context_data.get(
        "total_context_characters"
    )

    if characters is None:

        characters = context_data.get(
            "context_characters"
        )

    if characters is None:

        total = 0

        for record in evidence:

            text = (
                record.get(
                    "text"
                )
                or
                record.get(
                    "chunk_text"
                )
                or
                record.get(
                    "content"
                )
                or
                ""
            )

            total += len(
                str(text)
            )

        characters = total

    # --------------------------------------------------------
    # Truncation
    # --------------------------------------------------------

    truncated = context_data.get(
        "truncated",
        context_data.get(
            "context_truncated",
            False
        )
    )

    return (
        evidence,
        cases_count,
        characters,
        truncated
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("RECOVERING Q001-Q003 RETRIEVED EVIDENCE")
    print("=" * 80)

    old_contexts = get_old_contexts()

    print(
        f"\nOld context files found: "
        f"{len(old_contexts)}"
    )

    if not old_contexts:

        print(
            "No old context files found."
        )

        return

    # ========================================================
    # PROCESS Q001-Q003
    # ========================================================

    for query_number in range(
        1,
        4
    ):

        query_id = (
            f"Q{query_number:03d}"
        )

        print("\n")
        print("-" * 80)
        print(query_id)

        answer_path = get_standardized_file(
            query_id
        )

        answer_data = load_json(
            answer_path
        )

        question = answer_data.get(
            "question",
            ""
        )

        print(
            f"Question:\n{question}"
        )

        # ----------------------------------------------------
        # Find best context
        # ----------------------------------------------------

        candidates = []

        for context in old_contexts:

            score = similarity(
                question,
                context["query"]
            )

            candidates.append(
                (
                    score,
                    context
                )
            )

        candidates.sort(
            key=lambda x: x[0],
            reverse=True
        )

        # ----------------------------------------------------
        # Show top matches
        # ----------------------------------------------------

        print(
            "\nTop context matches:"
        )

        for score, context in candidates[:5]:

            print(
                f"\nScore: {score:.4f}"
            )

            print(
                f"File: "
                f"{context['path'].name}"
            )

            print(
                f"Query: "
                f"{context['query']}"
            )

            print(
                f"Evidence records: "
                f"{len(context['evidence'])}"
            )

        # ----------------------------------------------------
        # Best match
        # ----------------------------------------------------

        best_score, best_context = (
            candidates[0]
        )

        # Safety threshold
        if best_score < 0.40:

            print(
                "\nWARNING:"
            )

            print(
                f"Best similarity is only "
                f"{best_score:.4f}"
            )

            print(
                "Skipping this query rather than "
                "making an unsafe context assignment."
            )

            continue

        context_data = (
            best_context["data"]
        )

        (
            evidence,
            cases_count,
            characters,
            truncated
        ) = extract_context_info(
            context_data
        )

        # ----------------------------------------------------
        # Update standardized answer
        # ----------------------------------------------------

        answer_data[
            "cases_considered"
        ] = cases_count

        answer_data[
            "evidence_records"
        ] = evidence

        answer_data[
            "context_characters"
        ] = characters

        answer_data[
            "context_truncated"
        ] = truncated

        answer_data[
            "context_source_file"
        ] = best_context[
            "path"
        ].name

        answer_data[
            "recovered_from_legacy_context"
        ] = True

        answer_data[
            "context_match_score"
        ] = round(
            best_score,
            4
        )

        save_json(
            answer_path,
            answer_data
        )

        print(
            "\nSELECTED CONTEXT:"
        )

        print(
            best_context[
                "path"
            ].name
        )

        print(
            f"Match score: "
            f"{best_score:.4f}"
        )

        print(
            f"Cases: "
            f"{cases_count}"
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
            "\nSaved successfully."
        )

    # ========================================================
    # VALIDATION
    # ========================================================

    print("\n")
    print("=" * 80)
    print("Q001-Q003 VALIDATION")
    print("=" * 80)

    success = True

    for query_number in range(
        1,
        4
    ):

        query_id = (
            f"Q{query_number:03d}"
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

            success = False

            continue

        data = load_json(
            path
        )

        evidence = data.get(
            "evidence_records",
            []
        )

        cases = data.get(
            "cases_considered",
            0
        )

        chars = data.get(
            "context_characters",
            0
        )

        print(
            f"{query_id}: "
            f"{len(evidence)} evidence | "
            f"{cases} cases | "
            f"{chars} chars"
        )

        if not evidence:

            success = False

    print("\n")

    if success:

        print(
            "SUCCESS!"
        )

        print(
            "Q001-Q003 now contain retrieved evidence."
        )

        print(
            "\nAll 30 RAG files are ready for groundedness evaluation."
        )

    else:

        print(
            "WARNING:"
        )

        print(
            "One or more Q001-Q003 files still have zero evidence."
        )

        print(
            "DO NOT run groundedness yet."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()