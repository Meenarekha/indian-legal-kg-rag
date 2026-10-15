import json
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EVALUATION_DIR = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "evaluation_results"
)

GENERATION_DIR = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "generation_outputs"
)

ANALYSIS_DIR = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "failure_analysis"
)

ANALYSIS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD JSON
# ============================================================

def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ============================================================
# FIND LATEST GENERATION FILE
# ============================================================

def find_latest_generation(
    query_id,
    generation_type
):

    candidates = []

    for path in GENERATION_DIR.glob(
        "*.json"
    ):

        filename = path.name.lower()

        if generation_type == "RAG":

            if not filename.startswith(
                "rag_answer_"
            ):
                continue

        elif generation_type == "BASE_LLM":

            if not filename.startswith(
                "base_answer_"
            ):
                continue

        try:

            data = load_json(path)

        except Exception:

            continue

        file_query_id = str(
            data.get(
                "query_id",
                ""
            )
        ).upper()

        if file_query_id != query_id:

            continue

        if not data.get("answer"):

            continue

        candidates.append(
            (
                data.get(
                    "generated_at",
                    ""
                ),
                path,
                data
            )
        )

    if not candidates:

        return None, None

    candidates.sort(
        key=lambda x: x[0],
        reverse=True
    )

    _, path, data = candidates[0]

    return path, data


# ============================================================
# LOAD RAG CONTEXT SUMMARY
# ============================================================

def get_rag_context_info(
    rag_data
):

    context_file = rag_data.get(
        "context_file"
    )

    if not context_file:

        return {
            "context_file": None,
            "cases_considered": None,
            "evidence_records": None,
            "context_characters": None,
            "context_truncated": None,
            "case_names": []
        }

    context_path = (
        PROJECT_ROOT
        / context_file
    )

    if not context_path.exists():

        return {
            "context_file":
                context_file,

            "cases_considered":
                None,

            "evidence_records":
                None,

            "context_characters":
                None,

            "context_truncated":
                None,

            "case_names": []
        }

    try:

        context_data = load_json(
            context_path
        )

    except Exception:

        return {
            "context_file":
                context_file,

            "cases_considered":
                None,

            "evidence_records":
                None,

            "context_characters":
                None,

            "context_truncated":
                None,

            "case_names": []
        }

    # --------------------------------------------------------
    # Retrieval summary
    # --------------------------------------------------------

    retrieval_summary = rag_data.get(
        "retrieval_summary",
        {}
    )

    cases_considered = (
        retrieval_summary.get(
            "cases_considered"
        )
    )

    evidence_records = (
        retrieval_summary.get(
            "evidence_records"
        )
    )

    context_characters = (
        retrieval_summary.get(
            "context_characters"
        )
    )

    context_truncated = (
        retrieval_summary.get(
            "context_truncated"
        )
    )

    # --------------------------------------------------------
    # Extract case names from evidence
    # --------------------------------------------------------

    case_names = []

    evidence = context_data.get(
        "evidence_records",
        []
    )

    for record in evidence:

        case_name = record.get(
            "case_name"
        )

        if (
            case_name
            and
            case_name not in case_names
        ):

            case_names.append(
                case_name
            )

    return {

        "context_file":
            context_file,

        "cases_considered":
            cases_considered,

        "evidence_records":
            evidence_records,

        "context_characters":
            context_characters,

        "context_truncated":
            context_truncated,

        "case_names":
            case_names
    }


# ============================================================
# CALCULATE COMMON SCORE
# ============================================================

def common_score(
    score_data
):

    values = [
        score_data.get(
            "relevance"
        ),
        score_data.get(
            "reference_consistency"
        ),
        score_data.get(
            "completeness"
        )
    ]

    values = [
        float(value)
        for value in values
        if value is not None
    ]

    if not values:

        return None

    return round(
        sum(values)
        /
        len(values),
        3
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("RAG FAILURE ANALYSIS")
    print("=" * 80)

    evaluation_files = sorted(
        EVALUATION_DIR.glob(
            "evaluation_Q*.json"
        )
    )

    print(
        f"\nEvaluation files found: "
        f"{len(evaluation_files)}"
    )

    records = []

    # ========================================================
    # PROCESS EACH QUESTION
    # ========================================================

    for evaluation_path in evaluation_files:

        try:

            evaluation = load_json(
                evaluation_path
            )

        except Exception as error:

            print(
                f"Could not read "
                f"{evaluation_path.name}: "
                f"{error}"
            )

            continue

        query_id = str(
            evaluation.get(
                "query_id",
                ""
            )
        ).upper()

        if not query_id:

            continue

        # ----------------------------------------------------
        # Scores
        # ----------------------------------------------------

        base = evaluation.get(
            "base",
            {}
        )

        rag = evaluation.get(
            "rag",
            {}
        )

        base_common = common_score(
            base
        )

        rag_common = common_score(
            rag
        )

        # ----------------------------------------------------
        # Difference
        # ----------------------------------------------------

        if (
            base_common is not None
            and
            rag_common is not None
        ):

            difference = round(
                rag_common
                -
                base_common,
                3
            )

        else:

            difference = None

        # ----------------------------------------------------
        # Winner
        # ----------------------------------------------------

        if difference is None:

            winner = "UNKNOWN"

        elif difference > 0:

            winner = "RAG"

        elif difference < 0:

            winner = "BASE"

        else:

            winner = "TIE"

        # ----------------------------------------------------
        # Load RAG generation file
        # ----------------------------------------------------

        rag_path, rag_data = (
            find_latest_generation(
                query_id,
                "RAG"
            )
        )

        # ----------------------------------------------------
        # RAG context
        # ----------------------------------------------------

        if rag_data:

            context_info = (
                get_rag_context_info(
                    rag_data
                )
            )

            rag_provider = (
                rag_data.get(
                    "provider"
                )
            )

        else:

            context_info = {

                "context_file": None,

                "cases_considered": None,

                "evidence_records": None,

                "context_characters": None,

                "context_truncated": None,

                "case_names": []
            }

            rag_provider = None

        # ----------------------------------------------------
        # Build record
        # ----------------------------------------------------

        record = {

            "query_id":
                query_id,

            "question":
                evaluation.get(
                    "question"
                ),

            "base_common_score":
                base_common,

            "rag_common_score":
                rag_common,

            "rag_minus_base":
                difference,

            "winner":
                winner,

            "base_relevance":
                base.get(
                    "relevance"
                ),

            "base_reference_consistency":
                base.get(
                    "reference_consistency"
                ),

            "base_completeness":
                base.get(
                    "completeness"
                ),

            "rag_relevance":
                rag.get(
                    "relevance"
                ),

            "rag_reference_consistency":
                rag.get(
                    "reference_consistency"
                ),

            "rag_completeness":
                rag.get(
                    "completeness"
                ),

            "rag_groundedness":
                rag.get(
                    "groundedness"
                ),

            "base_overall":
                base.get(
                    "overall"
                ),

            "rag_overall":
                rag.get(
                    "overall"
                ),

            "rag_provider":
                rag_provider,

            "cases_considered":
                context_info[
                    "cases_considered"
                ],

            "evidence_records":
                context_info[
                    "evidence_records"
                ],

            "context_characters":
                context_info[
                    "context_characters"
                ],

            "context_truncated":
                context_info[
                    "context_truncated"
                ],

            "case_names":
                context_info[
                    "case_names"
                ],

            "rag_context_file":
                context_info[
                    "context_file"
                ],

            "brief_reason":
                evaluation.get(
                    "brief_reason",
                    ""
                )
        }

        records.append(
            record
        )

    # ========================================================
    # SORT
    # ========================================================

    records.sort(
        key=lambda x:
            int(
                x["query_id"][1:]
            )
    )

    # ========================================================
    # CLASSIFICATION
    # ========================================================

    rag_wins = [
        x
        for x in records
        if x["winner"] == "RAG"
    ]

    base_wins = [
        x
        for x in records
        if x["winner"] == "BASE"
    ]

    ties = [
        x
        for x in records
        if x["winner"] == "TIE"
    ]

    low_groundedness = [
        x
        for x in records
        if (
            x["rag_groundedness"]
            is not None
            and
            float(
                x["rag_groundedness"]
            ) < 3
        )
    ]

    good_groundedness = [
        x
        for x in records
        if (
            x["rag_groundedness"]
            is not None
            and
            float(
                x["rag_groundedness"]
            ) >= 3
        )
    ]

    # ========================================================
    # SAVE COMPLETE ANALYSIS
    # ========================================================

    analysis_file = (
        ANALYSIS_DIR
        /
        "rag_failure_analysis.json"
    )

    analysis = {

        "total_questions":
            len(records),

        "rag_wins":
            len(rag_wins),

        "base_wins":
            len(base_wins),

        "ties":
            len(ties),

        "low_groundedness_count":
            len(low_groundedness),

        "good_groundedness_count":
            len(good_groundedness),

        "records":
            records
    }

    with open(
        analysis_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            analysis,
            file,
            indent=2,
            ensure_ascii=False
        )

    # ========================================================
    # SAVE CSV
    # ========================================================

    csv_file = (
        ANALYSIS_DIR
        /
        "rag_failure_analysis.csv"
    )

    import csv

    if records:

        fieldnames = list(
            records[0].keys()
        )

        with open(
            csv_file,
            "w",
            encoding="utf-8",
            newline=""
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames
            )

            writer.writeheader()

            for record in records:

                row = record.copy()

                row["case_names"] = (
                    " | ".join(
                        row["case_names"]
                    )
                )

                writer.writerow(
                    row
                )

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print("\n")
    print("=" * 80)
    print("FAILURE ANALYSIS SUMMARY")
    print("=" * 80)

    print(
        f"\nQuestions analyzed: "
        f"{len(records)}"
    )

    print(
        f"RAG wins: "
        f"{len(rag_wins)}"
    )

    print(
        f"Base wins: "
        f"{len(base_wins)}"
    )

    print(
        f"Ties: "
        f"{len(ties)}"
    )

    print(
        f"\nRAG groundedness < 3: "
        f"{len(low_groundedness)}"
    )

    print(
        f"RAG groundedness >= 3: "
        f"{len(good_groundedness)}"
    )

    # ========================================================
    # PRINT RAG WINS
    # ========================================================

    print("\n")
    print("=" * 80)
    print("QUESTIONS WHERE RAG WINS")
    print("=" * 80)

    for item in rag_wins:

        print(
            f"\n{item['query_id']} "
            f"| Base={item['base_common_score']} "
            f"| RAG={item['rag_common_score']} "
            f"| Groundedness={item['rag_groundedness']}"
        )

        print(
            f"Question: "
            f"{item['question']}"
        )

    # ========================================================
    # PRINT BASE WINS
    # ========================================================

    print("\n")
    print("=" * 80)
    print("QUESTIONS WHERE BASE WINS")
    print("=" * 80)

    for item in base_wins:

        print(
            f"\n{item['query_id']} "
            f"| Base={item['base_common_score']} "
            f"| RAG={item['rag_common_score']} "
            f"| Groundedness={item['rag_groundedness']}"
        )

        print(
            f"Question: "
            f"{item['question']}"
        )

    # ========================================================
    # PRINT LOW GROUNDEDNESS
    # ========================================================

    print("\n")
    print("=" * 80)
    print("LOW GROUNDEDNESS QUESTIONS")
    print("=" * 80)

    for item in low_groundedness:

        print(
            f"\n{item['query_id']} "
            f"| Groundedness="
            f"{item['rag_groundedness']} "
            f"| RAG={item['rag_common_score']} "
            f"| Base={item['base_common_score']}"
        )

        print(
            f"Cases considered: "
            f"{item['cases_considered']}"
        )

        print(
            f"Evidence records: "
            f"{item['evidence_records']}"
        )

        print(
            f"Context characters: "
            f"{item['context_characters']}"
        )

        print(
            f"Truncated: "
            f"{item['context_truncated']}"
        )

        print(
            f"Cases: "
            f"{'; '.join(item['case_names'])}"
        )

        print(
            f"Reason: "
            f"{item['brief_reason']}"
        )

    # ========================================================
    # OUTPUT PATHS
    # ========================================================

    print("\n")
    print("=" * 80)
    print("FILES CREATED")
    print("=" * 80)

    print(
        f"\nJSON:"
    )

    print(
        analysis_file
    )

    print(
        f"\nCSV:"
    )

    print(
        csv_file
    )

    print(
        "\nFailure analysis complete."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()