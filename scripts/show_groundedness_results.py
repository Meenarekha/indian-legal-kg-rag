import json
from pathlib import Path


# ============================================================
# PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULT_FILE = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "groundedness_results.json"
)


# ============================================================
# LOAD RESULTS
# ============================================================

if not RESULT_FILE.exists():

    print("ERROR: groundedness_results.json not found.")
    print()
    print("Expected location:")
    print(RESULT_FILE)
    raise SystemExit(1)


with open(
    RESULT_FILE,
    "r",
    encoding="utf-8"
) as f:

    data = json.load(f)


results = {
    item["query_id"]: item
    for item in data.get("results", [])
}


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 90)
print("                 RAG GROUNDEDNESS EVALUATION RESULTS")
print("=" * 90)

print()
print("Evaluation basis : Actual retrieved judicial evidence")
print("Evaluator model   : " + data.get(
    "evaluator_model",
    "N/A"
))

print()


# ============================================================
# TABLE HEADER
# ============================================================

print(
    f"{'Query':<8}"
    f"{'Groundedness':<16}"
    f"{'Supported':<14}"
    f"{'Unsupported':<16}"
    f"{'Total Claims':<15}"
    f"{'Status'}"
)

print("-" * 90)


# ============================================================
# ALL 30 QUESTIONS
# ============================================================

for i in range(1, 31):

    query_id = f"Q{i:03d}"

    result = results.get(query_id)


    # --------------------------------------------------------
    # Not evaluated
    # --------------------------------------------------------

    if result is None:

        print(
            f"{query_id:<8}"
            f"{'--':<16}"
            f"{'--':<14}"
            f"{'--':<16}"
            f"{'--':<15}"
            f"Not evaluated"
        )

        continue


    score = result.get(
        "groundedness_score",
        "--"
    )

    supported = result.get(
        "supported_claims",
        "--"
    )

    unsupported = result.get(
        "unsupported_claims",
        "--"
    )

    total = result.get(
        "total_claims",
        "--"
    )


    print(
        f"{query_id:<8}"
        f"{str(score) + '/5':<16}"
        f"{str(supported):<14}"
        f"{str(unsupported):<16}"
        f"{str(total):<15}"
        f"Evaluated"
    )


# ============================================================
# SUMMARY
# ============================================================

print("-" * 90)

evaluated = data.get(
    "questions_evaluated",
    len(results)
)

requested = 30

average = data.get(
    "average_groundedness",
    0
)

supported_total = data.get(
    "supported_claims",
    0
)

unsupported_total = data.get(
    "unsupported_claims",
    0
)

total_claims = data.get(
    "total_claims",
    0
)

support_rate = data.get(
    "evidence_support_rate_percent",
    0
)


print(
    f"{'Questions evaluated':<30}: "
    f"{evaluated} / {requested}"
)

print(
    f"{'Average groundedness':<30}: "
    f"{average:.3f} / 5"
)

print(
    f"{'Supported claims':<30}: "
    f"{supported_total}"
)

print(
    f"{'Unsupported claims':<30}: "
    f"{unsupported_total}"
)

print(
    f"{'Total evaluated claims':<30}: "
    f"{total_claims}"
)

print(
    f"{'Evidence support rate':<30}: "
    f"{support_rate:.2f}%"
)


# ============================================================
# FAILED / UNEVALUATED QUESTIONS
# ============================================================

failed = []

for i in range(1, 31):

    query_id = f"Q{i:03d}"

    if query_id not in results:

        failed.append(query_id)


print()

if failed:

    print(
        "Not evaluated due to API quota:"
    )

    print(
        ", ".join(failed)
    )

else:

    print(
        "All 30 questions were evaluated."
    )


# ============================================================
# IMPORTANT NOTE
# ============================================================

print()

print("-" * 90)

print(
    "NOTE: Groundedness measures support from retrieved "
    "judicial evidence."
)

print(
    "It does not constitute professional legal validation."
)

print("=" * 90)
print()