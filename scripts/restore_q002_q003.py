import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

GENERATION_DIR = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "generation_outputs"
)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False
        )


# ============================================================
# ORIGINAL FILES
# ============================================================

original_files = {
    "Q002": "rag_answer_q002_20260914_010008.json",
    "Q003": "rag_answer_q003_20260914_010116.json",
}


# ============================================================
# RESTORE
# ============================================================

print("=" * 80)
print("RESTORING ORIGINAL Q002 AND Q003 RETRIEVAL EVIDENCE")
print("=" * 80)


for query_id, filename in original_files.items():

    print("\n" + "-" * 80)
    print(query_id)

    original_path = GENERATION_DIR / filename
    output_path = (
        GENERATION_DIR
        / f"rag_answer_{query_id}.json"
    )

    # --------------------------------------------------------
    # Check original file
    # --------------------------------------------------------

    if not original_path.exists():

        print(
            f"ERROR: Original file not found:\n"
            f"{original_path}"
        )

        continue

    original = load_json(
        original_path
    )

    print(
        f"Original file: {filename}"
    )

    print(
        f"Question: {original.get('question')}"
    )

    # --------------------------------------------------------
    # Extract actual original evidence
    # --------------------------------------------------------

    evidence = original.get(
        "evidence_records",
        []
    )

    context_file = original.get(
        "context_file"
    )

    context_characters = original.get(
        "total_context_characters",
        0
    )

    print(
        f"Original context file: "
        f"{context_file}"
    )

    print(
        f"Original evidence records: "
        f"{len(evidence)}"
    )

    print(
        f"Original context characters: "
        f"{context_characters}"
    )

    # --------------------------------------------------------
    # Load current standardized answer
    # --------------------------------------------------------

    if output_path.exists():

        standardized = load_json(
            output_path
        )

    else:

        standardized = original.copy()

    # --------------------------------------------------------
    # Restore actual evidence
    # --------------------------------------------------------

    standardized[
        "evidence_records"
    ] = evidence

    standardized[
        "context_file"
    ] = context_file

    standardized[
        "context_source_file"
    ] = context_file

    standardized[
        "context_characters"
    ] = context_characters

    standardized[
        "cases_considered"
    ] = 0

    # --------------------------------------------------------
    # Calculate unique cases from evidence
    # --------------------------------------------------------

    unique_cases = set()

    for record in evidence:

        case_id = (
            record.get("case_id")
            or record.get("case_name")
            or record.get("source_file")
        )

        if case_id:
            unique_cases.add(
                str(case_id)
            )

    standardized[
        "cases_considered"
    ] = len(unique_cases)

    # --------------------------------------------------------
    # Preserve provenance
    # --------------------------------------------------------

    standardized[
        "original_answer_file"
    ] = filename

    standardized[
        "recovered_from_original_generation"
    ] = True

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_json(
        output_path,
        standardized
    )

    print(
        f"Restored cases: "
        f"{len(unique_cases)}"
    )

    print(
        f"Restored evidence: "
        f"{len(evidence)}"
    )

    print(
        f"Restored context characters: "
        f"{context_characters}"
    )

    print(
        f"Saved: "
        f"{output_path.name}"
    )


# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n")
print("=" * 80)
print("FINAL Q002/Q003 VALIDATION")
print("=" * 80)


all_good = True


for query_id in ["Q002", "Q003"]:

    path = (
        GENERATION_DIR
        / f"rag_answer_{query_id}.json"
    )

    if not path.exists():

        print(
            f"{query_id}: FILE MISSING"
        )

        all_good = False
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

    context = data.get(
        "context_source_file"
    )

    print(
        f"{query_id}: "
        f"{len(evidence)} evidence | "
        f"{cases} cases | "
        f"{chars} chars"
    )

    print(
        f"  Context: {context}"
    )

    if not evidence:

        all_good = False


print("\n")

if all_good:

    print(
        "SUCCESS!"
    )

    print(
        "Q002 and Q003 have their ORIGINAL retrieval evidence."
    )

    print(
        "Do not regenerate these answers."
    )

else:

    print(
        "WARNING: One or more queries still have missing evidence."
    )

    print(
        "DO NOT run groundedness yet."
    )